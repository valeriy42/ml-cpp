/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include <sandbox/CPytorchInferenceSandboxPolicy.h>

#include <limits.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>

#ifdef SANDBOX2_AVAILABLE
#include <seccomp/CPytorchInferenceSyscallAllowlist.h>
#endif

#ifdef __linux__
#include <linux/futex.h>
#endif

namespace ml {
namespace sandbox {

namespace {

//! The only recognized path-bearing launch options. Adding or renaming one
//! requires a change here, a policy test, and an end-to-end Elasticsearch
//! invocation test (design.md §Typed argument extraction).
bool isPathOptionName(const std::string& name) {
    return name == "input" || name == "output" || name == "restore" || name == "logPipe";
}

//! Split a path into components, without resolving "." or "..".
std::vector<std::string> splitPathComponents(const std::string& path) {
    std::vector<std::string> components;
    std::string current;
    for (char c : path) {
        if (c == '/') {
            if (current.empty() == false) {
                components.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (current.empty() == false) {
        components.push_back(current);
    }
    return components;
}

bool containsDotDot(const std::vector<std::string>& components) {
    return std::find(components.begin(), components.end(), "..") != components.end();
}

//! realpath() requires the target to exist. The leaf FIFO/file may not
//! exist yet at validation time, but the native controller creates the
//! per-child ml-child-ipc/<child-id> directory before policy construction
//! (design.md §Required layout), so canonicalizing the *parent* directory
//! of the leaf is always meaningful.
bool canonicalize(const std::string& dir, std::string& canonicalOut) {
    char resolved[PATH_MAX];
    if (::realpath(dir.c_str(), resolved) == nullptr) {
        return false;
    }
    canonicalOut.assign(resolved);
    return true;
}

} // namespace

SChildIpcValidationResult validateChildIpcLaunchSpec(const std::string& trustedTmpDir,
                                                       const std::vector<std::string>& args) {
    SChildIpcValidationResult result;

    std::string trustedTmpDirCanonical;
    const bool trustedBaseResolved = canonicalize(trustedTmpDir, trustedTmpDirCanonical);

    std::vector<std::string> seenLiteralArgs;

    for (const std::string& arg : args) {
        const std::size_t eqPos = arg.find('=');
        if (eqPos == std::string::npos || eqPos + 1 >= arg.size()) {
            continue;
        }

        std::string optionName{arg.substr(0, eqPos)};
        while (optionName.empty() == false && optionName[0] == '-') {
            optionName.erase(0, 1);
        }

        if (isPathOptionName(optionName) == false) {
            continue;
        }

        const std::string value{arg.substr(eqPos + 1)};

        if (std::find(seenLiteralArgs.begin(), seenLiteralArgs.end(), arg) != seenLiteralArgs.end()) {
            result.s_Rejected.push_back({arg, EChildIpcPathRejection::E_Duplicate});
            continue;
        }
        seenLiteralArgs.push_back(arg);

        if (value.empty() || value[0] != '/') {
            result.s_Rejected.push_back({arg, EChildIpcPathRejection::E_NotAbsolute});
            continue;
        }

        const std::vector<std::string> components{splitPathComponents(value)};
        if (containsDotDot(components)) {
            result.s_Rejected.push_back({arg, EChildIpcPathRejection::E_ContainsDotDot});
            continue;
        }
        if (components.size() < 2) {
            // Fewer than two components below '/' means either the root
            // itself or a direct child of root - never a valid three-deep
            // $TMPDIR/ml-child-ipc/<child-id>/<leaf> path.
            result.s_Rejected.push_back({arg, EChildIpcPathRejection::E_RootLevelPath});
            continue;
        }

        const std::string leaf{components.back()};
        const std::size_t lastSlash = value.rfind('/');
        const std::string literalParent{value.substr(0, lastSlash)};

        if (trustedBaseResolved == false) {
            result.s_Rejected.push_back({arg, EChildIpcPathRejection::E_CanonicalizationFailed});
            continue;
        }

        std::string canonicalParent;
        if (canonicalize(literalParent, canonicalParent) == false) {
            result.s_Rejected.push_back({arg, EChildIpcPathRejection::E_CanonicalizationFailed});
            continue;
        }

        if (literalParent != canonicalParent) {
            // The literal path traverses a symlink (or other alias) before
            // reaching its parent directory. Accepting both forms - as the
            // pre-PR-C raw inference did - would let a mutable link widen
            // the mount after validation ran. Reject instead of mounting
            // either form.
            result.s_Rejected.push_back({arg, EChildIpcPathRejection::E_MutableSymlinkOrAlias});
            continue;
        }

        const std::vector<std::string> canonicalComponents{splitPathComponents(canonicalParent)};
        const std::vector<std::string> baseComponents{splitPathComponents(trustedTmpDirCanonical)};

        const bool underBase = canonicalComponents.size() == baseComponents.size() + 2 &&
                               std::equal(baseComponents.begin(), baseComponents.end(),
                                          canonicalComponents.begin());
        if (underBase == false) {
            const bool sharesBasePrefix =
                canonicalComponents.size() >= baseComponents.size() &&
                std::equal(baseComponents.begin(), baseComponents.end(), canonicalComponents.begin());
            result.s_Rejected.push_back(
                {arg, sharesBasePrefix ? EChildIpcPathRejection::E_WrongDepth
                                        : EChildIpcPathRejection::E_OutsideTrustedBase});
            continue;
        }

        const std::string intermediateDir{canonicalComponents[baseComponents.size()]};
        if (intermediateDir != "ml-child-ipc") {
            result.s_Rejected.push_back({arg, EChildIpcPathRejection::E_WrongDepth});
            continue;
        }

        const std::string childId{canonicalComponents.back()};
        if (result.s_Spec.s_ChildId.empty() == false && result.s_Spec.s_ChildId != childId) {
            result.s_Rejected.push_back({arg, EChildIpcPathRejection::E_ChildIdMismatch});
            continue;
        }

        result.s_Spec.s_ChildId = childId;
        result.s_Spec.s_ChildIpcRoot = canonicalParent;
        result.s_Spec.s_PipePaths.push_back(canonicalParent + "/" + leaf);
    }

    result.s_Ok = result.s_Rejected.empty() && result.s_Spec.s_ChildId.empty() == false;
    if (result.s_Ok == false) {
        // A rejected argument or an entirely absent path option both fail
        // the spawn; never return a partially-populated spec the caller
        // might build a policy from by mistake.
        result.s_Spec = SChildIpcLaunchSpec{};
    }
    return result;
}

#ifdef SANDBOX2_AVAILABLE

const std::vector<SFixedMountDecision>& fixedMountDecisions() {
    static const std::vector<SFixedMountDecision> DECISIONS{
        {"/lib", EFixedMountAction::E_MountReadOnlyDirectory,
         "Dynamic loader resolves libc/libgcc/libstdc++ from here at "
         "runtime; the set is unbounded and platform-dependent, so "
         "per-file allowlisting would duplicate the loader's own search "
         "logic."},
        {"/lib64", EFixedMountAction::E_MountReadOnlyDirectory,
         "Same reason as /lib, on the lib64 multilib path used by the "
         "64-bit dynamic loader on our supported Linux distributions."},
        {"/usr/lib", EFixedMountAction::E_MountReadOnlyDirectory,
         "Same reason as /lib: libtorch and its transitive shared-library "
         "dependencies resolve from here."},
        {"/usr/lib64", EFixedMountAction::E_MountReadOnlyDirectory,
         "Same reason as /lib64, for 64-bit multilib packages."},
        {"/etc", EFixedMountAction::E_Skip,
         "Whole /etc is never mounted; allowlistedEtcFiles() lists the "
         "individually justified files pytorch_inference/libtorch actually "
         "need instead."},
        {"/proc", EFixedMountAction::E_MountNamespacedProcfs,
         "Sandbox2 mounts a fresh procfs inside the sandbox's own PID "
         "namespace; binding the host's /proc would leak every other "
         "process's memory maps and command lines into the sandbox."},
        {"/sys", EFixedMountAction::E_MountNamespacedProcfs,
         "Same reason as /proc: nothing in this policy binds host /sys."},
    };
    return DECISIONS;
}

const std::vector<std::string>& allowlistedEtcFiles() {
    // NOTE: /etc/ssl/certs/ca-certificates.crt is the Debian/Ubuntu trust
    // bundle path; the ml-cpp CI build image is CentOS7/RHEL-based, whose
    // equivalent is /etc/pki/tls/certs/ca-bundle.crt. This list has not yet
    // been verified against the actual supported-distro trust bundle path -
    // tracked as an open item for PR C's Linux verification pass, not
    // resolved here.
    static const std::vector<std::string> FILES{
        "/etc/nsswitch.conf",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/localtime",
        "/etc/ld.so.cache",
    };
    return FILES;
}

sandbox2::PolicyBuilder buildPytorchInferenceFilesystemPolicy(const std::string& binDir,
                                                               const std::string& libDir,
                                                               const SChildIpcLaunchSpec& spec,
                                                               std::size_t tmpfsSizeBytes) {
    sandbox2::PolicyBuilder policyBuilder;

    policyBuilder.AllowDynamicStartup()
        .AllowExit()
        .AllowHandleSignals()
        .AllowGetPIDs()
        .AllowGetRandom()
        .AllowTcMalloc()
        .AllowMmap();

#ifdef __linux__
    // glibc/libtorch use futex for mutexes and condition variables; timed
    // waits and broadcast/requeue paths need more than plain WAIT/WAKE (see
    // the carry-forward note on d9a856d5f in
    // include/seccomp/CPytorchInferenceSyscallAllowlist.h).
    policyBuilder.AllowFutexOp(FUTEX_WAIT)
        .AllowFutexOp(FUTEX_WAKE)
        .AllowFutexOp(FUTEX_WAIT_BITSET)
        .AllowFutexOp(FUTEX_WAKE_BITSET)
        .AllowFutexOp(FUTEX_REQUEUE)
        .AllowFutexOp(FUTEX_CMP_REQUEUE)
        .AllowFutexOp(FUTEX_WAKE_OP);
#endif

    // Consume the one machine-readable syscall declaration shared with the
    // legacy in-process BPF filter instead of hand-maintaining a second
    // list (design.md "Locked design decisions": "One machine-readable
    // declaration generates the applied legacy BPF allowlist and Sandbox2
    // explicit grants").
    for (int syscallNr : seccomp::pytorch_inference::legacyBpfAllowedSyscalls()) {
        policyBuilder.AllowSyscall(syscallNr);
    }

    policyBuilder.AddDirectory(binDir, /*is_ro=*/true);
    policyBuilder.AddDirectory(libDir, /*is_ro=*/true);

    for (const SFixedMountDecision& decision : fixedMountDecisions()) {
        switch (decision.s_Action) {
        case EFixedMountAction::E_MountReadOnlyDirectory:
            policyBuilder.AddDirectory(decision.s_Path, /*is_ro=*/true);
            break;
        case EFixedMountAction::E_MountNamespacedProcfs:
        case EFixedMountAction::E_Skip:
            // Sandbox2 supplies its own namespaced procfs/sysfs
            // automatically; nothing to add here for either case, and
            // adding decision.s_Path would bind the host directory instead.
            break;
        }
    }

    for (const std::string& etcFile : allowlistedEtcFiles()) {
        policyBuilder.AddFile(etcFile, /*is_ro=*/true);
    }

    for (const std::string& devFile : {"/dev/null", "/dev/urandom", "/dev/random"}) {
        policyBuilder.AddFile(devFile, /*is_ro=*/devFile != std::string{"/dev/null"});
    }

    // Private, bounded tmpfs - never the host's shared /tmp.
    policyBuilder.AddTmpfs("/tmp", tmpfsSizeBytes);

    // The one per-child IPC root, mapped read-write to a fixed in-sandbox
    // path. spec must already be s_Ok (validateChildIpcLaunchSpec), so
    // s_ChildIpcRoot is exactly $TMPDIR/ml-child-ipc/<child-id> - never
    // ml-child-ipc itself, never a sibling child's directory.
    policyBuilder.AddDirectoryAt(spec.s_ChildIpcRoot, "/run/elastic/ml-ipc", /*is_ro=*/false);

    return policyBuilder;
}

#endif // SANDBOX2_AVAILABLE

} // namespace sandbox
} // namespace ml
