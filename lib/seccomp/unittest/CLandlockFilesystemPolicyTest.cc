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

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <string>
#include <vector>

#ifdef Linux
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

BOOST_AUTO_TEST_SUITE(CLandlockFilesystemPolicyTest)

BOOST_AUTO_TEST_CASE(testDescribeCoversEveryOutcome) {
    BOOST_TEST_REQUIRE(ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Applied).empty() == false);
    BOOST_TEST_REQUIRE(ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Unsupported).empty() == false);
    BOOST_TEST_REQUIRE(ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Failed).empty() == false);
    BOOST_REQUIRE(ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Applied) !=
                  ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Failed));
}

#ifdef Linux

namespace {

//! Exit codes a confined child uses to report what it observed. The ruleset
//! is irreversible, so every case below must run in its own forked child -
//! confining the test process itself would break every later test.
enum EChildExit : int {
    E_ChildOk = 0,
    E_ChildNotApplied = 20,
    E_ChildGrantedPathUnreadable = 21,
    E_ChildDeniedPathStillReadable = 22,
    E_ChildGrantedDirNotWritable = 23
};

//! Run \p body in a forked child and return its exit code, or -1 if the child
//! did not exit normally.
template<typename FUNC>
int runInChild(FUNC body) {
    const pid_t child{::fork()};
    if (child < 0) {
        return -1;
    }
    if (child == 0) {
        ::_exit(body());
    }
    int status{0};
    if (::waitpid(child, &status, 0) != child || WIFEXITED(status) == false) {
        return -1;
    }
    return WEXITSTATUS(status);
}

std::string makeScratchDirectory() {
    std::string path{"/tmp/ml-landlock-test-XXXXXX"};
    if (::mkdtemp(path.data()) == nullptr) {
        return std::string();
    }
    return path;
}

} // namespace

BOOST_AUTO_TEST_CASE(testRulesetGrantsTheAllowedPathAndDeniesEverythingElse) {
    // The point of the fallback is that it actually bounds the sandboxee, so
    // assert both halves: a granted directory stays usable, and a path that
    // was never granted becomes unreadable *even though the uid still owns
    // it*. Without the second half a vacuously permissive ruleset would
    // "pass".
    const std::string scratch{makeScratchDirectory()};
    BOOST_TEST_REQUIRE(scratch.empty() == false);

    const std::string outside{scratch + "/outside.txt"};
    const int outsideFd{::open(outside.c_str(), O_CREAT | O_WRONLY, 0600)};
    BOOST_TEST_REQUIRE(outsideFd >= 0);
    ::close(outsideFd);

    const std::string granted{scratch + "/granted"};
    BOOST_TEST_REQUIRE(::mkdir(granted.c_str(), 0700) == 0);

    const int childResult{runInChild([&] {
        ml::seccomp::SLandlockPaths paths;
        paths.s_ReadWrite.push_back(granted);

        const ml::seccomp::ELandlockOutcome outcome{
            ml::seccomp::applyLandlockFilesystemPolicy(paths)};
        if (outcome == ml::seccomp::ELandlockOutcome::E_Unsupported) {
            // Reported to the parent, which skips rather than fails - a
            // kernel without Landlock is a legitimate environment.
            return static_cast<int>(E_ChildNotApplied);
        }
        if (outcome != ml::seccomp::ELandlockOutcome::E_Applied) {
            return static_cast<int>(E_ChildGrantedPathUnreadable);
        }

        // The granted directory must still be writable.
        const std::string inside{granted + "/inside.txt"};
        const int insideFd{::open(inside.c_str(), O_CREAT | O_WRONLY, 0600)};
        if (insideFd < 0) {
            return static_cast<int>(E_ChildGrantedDirNotWritable);
        }
        ::close(insideFd);

        // The sibling file, owned by this very uid, must now be unreachable.
        const int deniedFd{::open(outside.c_str(), O_RDONLY)};
        if (deniedFd >= 0) {
            ::close(deniedFd);
            return static_cast<int>(E_ChildDeniedPathStillReadable);
        }

        return static_cast<int>(E_ChildOk);
    })};

    ::unlink(outside.c_str());
    ::rmdir(granted.c_str());
    ::rmdir(scratch.c_str());

    if (childResult == E_ChildNotApplied) {
        BOOST_TEST_MESSAGE("Landlock unsupported on this kernel - skipping enforcement assertions");
        return;
    }
    BOOST_REQUIRE_EQUAL(childResult, static_cast<int>(E_ChildOk));
}

BOOST_AUTO_TEST_CASE(testPolicyIsIrreversibleWithinTheConfinedProcess) {
    // Landlock rulesets stack and can only narrow. Applying an empty second
    // ruleset must not restore access the first one removed - otherwise a
    // malicious model could simply re-apply a permissive policy.
    const std::string scratch{makeScratchDirectory()};
    BOOST_TEST_REQUIRE(scratch.empty() == false);
    const std::string probeFile{scratch + "/probe.txt"};
    const int fd{::open(probeFile.c_str(), O_CREAT | O_WRONLY, 0600)};
    BOOST_TEST_REQUIRE(fd >= 0);
    ::close(fd);

    const int childResult{runInChild([&] {
        ml::seccomp::SLandlockPaths empty;
        if (ml::seccomp::applyLandlockFilesystemPolicy(empty) ==
            ml::seccomp::ELandlockOutcome::E_Unsupported) {
            return static_cast<int>(E_ChildNotApplied);
        }
        // Now grant the scratch directory in a second ruleset; Landlock
        // composes by intersection, so this must NOT re-open access.
        ml::seccomp::SLandlockPaths permissive;
        permissive.s_ReadWrite.push_back(scratch);
        ml::seccomp::applyLandlockFilesystemPolicy(permissive);

        const int reopened{::open(probeFile.c_str(), O_RDONLY)};
        if (reopened >= 0) {
            ::close(reopened);
            return static_cast<int>(E_ChildDeniedPathStillReadable);
        }
        return static_cast<int>(E_ChildOk);
    })};

    ::unlink(probeFile.c_str());
    ::rmdir(scratch.c_str());

    if (childResult == E_ChildNotApplied) {
        BOOST_TEST_MESSAGE("Landlock unsupported on this kernel - skipping");
        return;
    }
    BOOST_REQUIRE_EQUAL(childResult, static_cast<int>(E_ChildOk));
}

BOOST_AUTO_TEST_CASE(testPytorchInferencePathsIncludeTheIpcDirectoryAsWritable) {
    // The derived path set must make the per-child IPC directory writable -
    // pytorch_inference creates its own log FIFO there - and must not make
    // it merely readable, which would fail at startup rather than at load.
    const ml::seccomp::SLandlockPaths paths{
        ml::seccomp::pytorchInferenceLandlockPaths("/app/tmp/ml-child-ipc/dep-1")};

    const auto contains = [](const std::vector<std::string>& haystack, const std::string& needle) {
        return std::find(haystack.begin(), haystack.end(), needle) != haystack.end();
    };

    BOOST_TEST_REQUIRE(contains(paths.s_ReadWrite, "/app/tmp/ml-child-ipc/dep-1"));
    BOOST_TEST_REQUIRE(contains(paths.s_ReadOnly, "/app/tmp/ml-child-ipc/dep-1") == false);
    // /proc/self, not /proc: oneMKL needs its own exe/maps, and granting the
    // whole of /proc would leave the host process table readable.
    BOOST_TEST_REQUIRE(contains(paths.s_ReadOnly, "/proc/self"));
    BOOST_TEST_REQUIRE(contains(paths.s_ReadOnly, "/proc") == false);
}

BOOST_AUTO_TEST_CASE(testRealPytorchPolicyDeniesTheExploitTargetWrite) {
    // End-to-end at the policy level, using the exact ruleset
    // pytorch_inference installs on the Landlock fallback route - not a
    // synthetic one. The attack-defense exploit model
    // (test/evil_model_generator.py) writes an -agentpath payload to
    // /usr/share/elasticsearch/config/jvm.options.d/gc.options; that path is
    // outside every grant pytorchInferenceLandlockPaths() produces, so the
    // real policy must deny a write there, while the per-child IPC directory
    // it does grant stays writable (pytorch_inference creates its own log
    // FIFO in it). This is the same boundary the harness's ROP exploit
    // exercises, proven deterministically without a build-fragile ROP chain.
    const std::string scratch{makeScratchDirectory()};
    BOOST_TEST_REQUIRE(scratch.empty() == false);

    // A stand-in for the operator TMPDIR, with the per-child IPC directory
    // laid out as the controller creates it: <tmp>/ml-child-ipc/<child-id>.
    const std::string ipcDir{scratch + "/ml-child-ipc/dep-e2e"};
    BOOST_TEST_REQUIRE(::system(("mkdir -p " + ipcDir).c_str()) == 0);

    // The exploit's hard-coded target, created here so the difference the
    // test observes is Landlock denying the write - not the parent directory
    // being absent. Its parent is deliberately outside every grant.
    const std::string forbiddenDir{scratch + "/config/jvm.options.d"};
    BOOST_TEST_REQUIRE(::system(("mkdir -p " + forbiddenDir).c_str()) == 0);
    const std::string forbiddenTarget{forbiddenDir + "/gc.options"};

    const int childResult{runInChild([&] {
        ml::seccomp::SLandlockPaths paths{
            ml::seccomp::pytorchInferenceLandlockPaths(ipcDir)};
        // Point the "config" grant nowhere near forbiddenTarget: the real
        // policy grants /etc etc., none of which cover this scratch config
        // path, so no extra removal is needed - forbiddenTarget is already
        // outside paths. Apply the real ruleset unchanged.
        const ml::seccomp::ELandlockOutcome outcome{
            ml::seccomp::applyLandlockFilesystemPolicy(paths)};
        if (outcome == ml::seccomp::ELandlockOutcome::E_Unsupported) {
            return static_cast<int>(E_ChildNotApplied);
        }
        if (outcome != ml::seccomp::ELandlockOutcome::E_Applied) {
            return static_cast<int>(E_ChildGrantedPathUnreadable);
        }

        // The per-child IPC directory the real policy grants must stay
        // writable - pytorch_inference creates its log FIFO there.
        const std::string fifoStandin{ipcDir + "/logPipe.test"};
        const int okFd{::open(fifoStandin.c_str(), O_CREAT | O_WRONLY, 0600)};
        if (okFd < 0) {
            return static_cast<int>(E_ChildGrantedDirNotWritable);
        }
        ::close(okFd);

        // The exploit's target write must be denied by the real policy.
        const int deniedFd{::open(forbiddenTarget.c_str(), O_CREAT | O_WRONLY, 0600)};
        if (deniedFd >= 0) {
            ::close(deniedFd);
            return static_cast<int>(E_ChildDeniedPathStillReadable);
        }
        return static_cast<int>(E_ChildOk);
    })};

    ::system(("rm -rf " + scratch).c_str());

    if (childResult == E_ChildNotApplied) {
        BOOST_TEST_MESSAGE("Landlock unsupported on this kernel - skipping");
        return;
    }
    BOOST_REQUIRE_EQUAL(childResult, static_cast<int>(E_ChildOk));
}

#endif // Linux

BOOST_AUTO_TEST_SUITE_END()
