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
#include <seccomp/CLandlockFilesystemPolicy.h>

#include <core/CLogger.h>

#include <cstdint>
#include <cstring>

#include <errno.h>
#include <fcntl.h>
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace ml {
namespace seccomp {
namespace {

// Landlock UAPI, declared locally: linux/landlock.h is absent from the
// CentOS7-based CI build image. Same approach as the raw statx/rseq/clone3
// numbers in CMlLegacyBpfSyscallAllowlist.h - build against whatever headers
// exist, then negotiate capability at runtime against the live kernel.
//
// The three syscalls were added together in 5.13 through the generic syscall
// table, so x86_64 and aarch64 share these numbers.
#ifndef ML_NR_landlock_create_ruleset
#define ML_NR_landlock_create_ruleset 444
#endif
#ifndef ML_NR_landlock_add_rule
#define ML_NR_landlock_add_rule 445
#endif
#ifndef ML_NR_landlock_restrict_self
#define ML_NR_landlock_restrict_self 446
#endif

constexpr std::uint32_t ML_LANDLOCK_CREATE_RULESET_VERSION{1U << 0};
constexpr int ML_LANDLOCK_RULE_PATH_BENEATH{1};

// Access rights, by the ABI version that introduced them.
constexpr std::uint64_t ACCESS_FS_EXECUTE{1ULL << 0};
constexpr std::uint64_t ACCESS_FS_WRITE_FILE{1ULL << 1};
constexpr std::uint64_t ACCESS_FS_READ_FILE{1ULL << 2};
constexpr std::uint64_t ACCESS_FS_READ_DIR{1ULL << 3};
constexpr std::uint64_t ACCESS_FS_REMOVE_DIR{1ULL << 4};
constexpr std::uint64_t ACCESS_FS_REMOVE_FILE{1ULL << 5};
constexpr std::uint64_t ACCESS_FS_MAKE_CHAR{1ULL << 6};
constexpr std::uint64_t ACCESS_FS_MAKE_DIR{1ULL << 7};
constexpr std::uint64_t ACCESS_FS_MAKE_REG{1ULL << 8};
constexpr std::uint64_t ACCESS_FS_MAKE_SOCK{1ULL << 9};
constexpr std::uint64_t ACCESS_FS_MAKE_FIFO{1ULL << 10};
constexpr std::uint64_t ACCESS_FS_MAKE_BLOCK{1ULL << 11};
constexpr std::uint64_t ACCESS_FS_MAKE_SYM{1ULL << 12};
constexpr std::uint64_t ACCESS_FS_REFER{1ULL << 13};   // ABI 2
constexpr std::uint64_t ACCESS_FS_TRUNCATE{1ULL << 14}; // ABI 3
constexpr std::uint64_t ACCESS_FS_IOCTL_DEV{1ULL << 15}; // ABI 5

//! Only handled_access_fs is filled, and the size passed to the kernel is
//! this structure's size, which is the ABI 1 size the kernel still accepts
//! from a newer kernel's point of view. Declaring the later
//! handled_access_net field would mean passing a larger size that an ABI 1-3
//! kernel rejects.
struct SLandlockRulesetAttr {
    std::uint64_t s_HandledAccessFs;
};

struct SLandlockPathBeneathAttr {
    std::uint64_t s_AllowedAccess;
    std::int32_t s_ParentFd;
} __attribute__((packed));

long landlockCreateRuleset(const SLandlockRulesetAttr* attr, std::size_t size, std::uint32_t flags) {
    return ::syscall(ML_NR_landlock_create_ruleset, attr, size, flags);
}

long landlockAddRule(int rulesetFd, int ruleType, const void* attr, std::uint32_t flags) {
    return ::syscall(ML_NR_landlock_add_rule, rulesetFd, ruleType, attr, flags);
}

long landlockRestrictSelf(int rulesetFd, std::uint32_t flags) {
    return ::syscall(ML_NR_landlock_restrict_self, rulesetFd, flags);
}

//! Every right this build knows about, narrowed to those \p abi understands.
//! Rights the kernel does not handle must be cleared: passing an unknown bit
//! makes landlock_create_ruleset() fail with EINVAL, which would turn a
//! newer-build-on-older-kernel into a hard failure rather than a slightly
//! coarser ruleset.
std::uint64_t handledAccessForAbi(int abi) {
    std::uint64_t handled{ACCESS_FS_EXECUTE | ACCESS_FS_WRITE_FILE | ACCESS_FS_READ_FILE |
                          ACCESS_FS_READ_DIR | ACCESS_FS_REMOVE_DIR | ACCESS_FS_REMOVE_FILE |
                          ACCESS_FS_MAKE_CHAR | ACCESS_FS_MAKE_DIR | ACCESS_FS_MAKE_REG |
                          ACCESS_FS_MAKE_SOCK | ACCESS_FS_MAKE_FIFO | ACCESS_FS_MAKE_BLOCK |
                          ACCESS_FS_MAKE_SYM};
    if (abi >= 2) {
        handled |= ACCESS_FS_REFER;
    }
    if (abi >= 3) {
        handled |= ACCESS_FS_TRUNCATE;
    }
    if (abi >= 5) {
        handled |= ACCESS_FS_IOCTL_DEV;
    }
    return handled;
}

//! The subset of access rights that apply to a non-directory. Every other
//! right describes an operation performed *within* a directory, and naming
//! one on a file makes landlock_add_rule() fail with EINVAL.
constexpr std::uint64_t FILE_APPLICABLE_ACCESS{ACCESS_FS_EXECUTE | ACCESS_FS_WRITE_FILE |
                                               ACCESS_FS_READ_FILE | ACCESS_FS_TRUNCATE |
                                               ACCESS_FS_IOCTL_DEV};

//! Grant a path the rights it needs, intersected with what the ruleset
//! handles. A right granted but not handled is meaningless (Landlock only
//! restricts what it handles), and a right handled but not granted is what
//! actually denies access.
bool addPathRule(int rulesetFd, const std::string& path, std::uint64_t allowed, std::uint64_t handled) {
    // O_PATH so opening the directory itself needs no read permission and
    // triggers none of the side effects of a real open.
    const int pathFd{::open(path.c_str(), O_PATH | O_CLOEXEC)};
    if (pathFd < 0) {
        // A path that simply does not exist on this host is not an error:
        // the fixed list covers several distribution layouts (/lib64 exists
        // on RHEL-family, is a symlink or absent elsewhere), and a rule for
        // a missing path grants nothing anyway.
        LOG_DEBUG(<< "Landlock: skipping absent path " << path << ": " << ::strerror(errno));
        return true;
    }

    // Landlock rejects a rule (EINVAL) whose allowed_access names a
    // directory-only right when the file descriptor is not a directory, so a
    // single "read-only" mask cannot be applied to both /usr/lib and
    // /dev/urandom. Narrow to the rights that are meaningful for a file.
    std::uint64_t allowedForThisPath{allowed};
    struct stat pathStat {};
    if (::fstat(pathFd, &pathStat) == 0 && S_ISDIR(pathStat.st_mode) == false) {
        allowedForThisPath &= FILE_APPLICABLE_ACCESS;
    }

    SLandlockPathBeneathAttr attr{};
    attr.s_AllowedAccess = allowedForThisPath & handled;
    attr.s_ParentFd = pathFd;
    const bool ok{landlockAddRule(rulesetFd, ML_LANDLOCK_RULE_PATH_BENEATH, &attr, 0) == 0};
    if (ok == false) {
        LOG_ERROR(<< "Landlock: could not add rule for " << path << ": " << ::strerror(errno));
    }
    ::close(pathFd);
    return ok;
}

std::string parentDirectory(const std::string& path) {
    const std::size_t lastSlash{path.rfind('/')};
    if (lastSlash == std::string::npos || lastSlash == 0) {
        return "/";
    }
    return path.substr(0, lastSlash);
}

} // namespace

std::string describe(ELandlockOutcome outcome) {
    switch (outcome) {
    case ELandlockOutcome::E_Applied:
        return "applied";
    case ELandlockOutcome::E_Unsupported:
        return "unsupported on this kernel";
    case ELandlockOutcome::E_Failed:
        return "failed";
    }
    return "unrecognized outcome";
}

SLandlockPaths pytorchInferenceLandlockPaths(const std::string& ipcDirectory) {
    SLandlockPaths paths;

    // The binary's own directory and its sibling lib directory, matching the
    // <install>/bin + <install>/lib layout the Sandbox2 filesystem policy
    // assumes. Resolved from /proc/self/exe rather than argv[0] so a
    // relative launch path (the controller uses "./pytorch_inference")
    // resolves correctly.
    char exePath[PATH_MAX];
    const ssize_t exeLength{::readlink("/proc/self/exe", exePath, sizeof(exePath) - 1)};
    if (exeLength > 0) {
        exePath[exeLength] = '\0';
        const std::string binDir{parentDirectory(exePath)};
        paths.s_ReadOnly.push_back(binDir);
        paths.s_ReadOnly.push_back(parentDirectory(binDir) + "/lib");
    }

    // System library directories the dynamic loader resolves through. The
    // bundled libraries are found via RPATH=$ORIGIN in the lib directory
    // above, but libc/libstdc++/libgcc and anything dlopen()ed lazily - Intel
    // oneMKL's CPU-specific kernels in particular - still come from here.
    for (const char* systemDir : {"/lib", "/lib64", "/usr/lib", "/usr/lib64"}) {
        paths.s_ReadOnly.push_back(systemDir);
    }

    // libtorch and glibc read these. /proc/self is needed because oneMKL's
    // dispatcher reads /proc/self/exe and /proc/self/maps to locate itself
    // before dlopen()ing its kernels - the same requirement that made the
    // Sandbox2 policy mount a namespaced /proc. Granting /proc/self rather
    // than /proc keeps the rest of the host process table unreadable.
    paths.s_ReadOnly.push_back("/proc/self");
    paths.s_ReadOnly.push_back("/etc");
    paths.s_ReadOnly.push_back("/sys/devices/system/cpu");
    paths.s_ReadOnly.push_back("/dev/urandom");
    paths.s_ReadOnly.push_back("/dev/random");

    // The one directory the sandboxee may modify. pytorch_inference creates
    // its own log FIFO here, so this needs FIFO creation as well as
    // read/write.
    paths.s_ReadWrite.push_back(ipcDirectory);
    paths.s_ReadWrite.push_back("/dev/null");

    return paths;
}

ELandlockOutcome applyLandlockFilesystemPolicy(const SLandlockPaths& paths) {
    const long abi{landlockCreateRuleset(nullptr, 0, ML_LANDLOCK_CREATE_RULESET_VERSION)};
    if (abi < 0) {
        const int createErrno{errno};
        // Distinguish "this kernel cannot do Landlock" from "something
        // refused the call", exactly as the Sandbox2 capability probe
        // distinguishes its own denials: the two have completely different
        // remedies, and collapsing them into one outcome is what made the
        // Sandbox2 failures opaque in the first place. ENOSYS means a kernel
        // older than 5.13 or Landlock compiled out; EOPNOTSUPP means built
        // in but not enabled in the bootloader's lsm= list. Anything else -
        // EACCES/EPERM in particular - means a seccomp filter or LSM denied
        // the syscall on a kernel that does support it.
        if (createErrno == ENOSYS || createErrno == EOPNOTSUPP) {
            LOG_WARN(<< "Landlock unsupported by this kernel: " << ::strerror(createErrno));
            return ELandlockOutcome::E_Unsupported;
        }
        LOG_ERROR(<< "Landlock is supported but landlock_create_ruleset was denied: "
                  << ::strerror(createErrno)
                  << " - a seccomp filter or LSM policy is blocking syscall "
                  << ML_NR_landlock_create_ruleset);
        return ELandlockOutcome::E_Failed;
    }

    const std::uint64_t handled{handledAccessForAbi(static_cast<int>(abi))};
    SLandlockRulesetAttr rulesetAttr{};
    rulesetAttr.s_HandledAccessFs = handled;

    const long rulesetFd{landlockCreateRuleset(&rulesetAttr, sizeof(rulesetAttr), 0)};
    if (rulesetFd < 0) {
        LOG_ERROR(<< "Landlock: could not create ruleset: " << ::strerror(errno));
        return ELandlockOutcome::E_Failed;
    }

    const int fd{static_cast<int>(rulesetFd)};
    bool ok{true};

    const std::uint64_t readOnlyAccess{ACCESS_FS_EXECUTE | ACCESS_FS_READ_FILE | ACCESS_FS_READ_DIR};
    for (const std::string& path : paths.s_ReadOnly) {
        ok = addPathRule(fd, path, readOnlyAccess, handled) && ok;
    }

    // REFER is granted on the writable directory so a rename within it still
    // works once REFER is handled (ABI 2+); without it, any cross-directory
    // rename is denied outright even between two granted paths.
    const std::uint64_t readWriteAccess{
        ACCESS_FS_READ_FILE | ACCESS_FS_WRITE_FILE | ACCESS_FS_READ_DIR |
        ACCESS_FS_REMOVE_FILE | ACCESS_FS_REMOVE_DIR | ACCESS_FS_MAKE_REG |
        ACCESS_FS_MAKE_DIR | ACCESS_FS_MAKE_FIFO | ACCESS_FS_MAKE_SOCK |
        ACCESS_FS_MAKE_SYM | ACCESS_FS_REFER | ACCESS_FS_TRUNCATE | ACCESS_FS_IOCTL_DEV};
    for (const std::string& path : paths.s_ReadWrite) {
        ok = addPathRule(fd, path, readWriteAccess, handled) && ok;
    }

    if (ok == false) {
        ::close(fd);
        return ELandlockOutcome::E_Failed;
    }

    // Landlock requires no_new_privs of a caller without CAP_SYS_ADMIN, so
    // that a confined process cannot escape by exec'ing something setuid.
    if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        LOG_ERROR(<< "Landlock: could not set no_new_privs: " << ::strerror(errno));
        ::close(fd);
        return ELandlockOutcome::E_Failed;
    }

    if (landlockRestrictSelf(fd, 0) != 0) {
        LOG_ERROR(<< "Landlock: could not restrict self: " << ::strerror(errno));
        ::close(fd);
        return ELandlockOutcome::E_Failed;
    }

    ::close(fd);
    LOG_INFO(<< "{\"event\":\"landlock_applied\",\"abi\":" << abi
             << ",\"read_only_paths\":" << paths.s_ReadOnly.size()
             << ",\"read_write_paths\":" << paths.s_ReadWrite.size() << "}");
    return ELandlockOutcome::E_Applied;
}

} // namespace seccomp
} // namespace ml
