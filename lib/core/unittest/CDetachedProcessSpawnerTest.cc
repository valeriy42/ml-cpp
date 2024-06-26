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

#include <core/CDetachedProcessSpawner.h>
#include <core/COsFileFuncs.h>
#include <core/CStringUtils.h>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

BOOST_AUTO_TEST_SUITE(CDetachedProcessSpawnerTest)

namespace {
const std::string OUTPUT_FILE("withNs.xml");
#ifdef Windows
// Unlike Windows NT system calls, copy's command line cannot cope with
// forward slash path separators
const std::string INPUT_FILE("testfiles\\withNs.xml");
// File size is different on Windows due to CRLF line endings
const size_t EXPECTED_FILE_SIZE(585);
const char* winDir(std::getenv("windir"));
const std::string PROCESS_PATH1(winDir != 0 ? std::string(winDir) + "\\System32\\cmd"
                                            : std::string("C:\\Windows\\System32\\cmd"));
const std::string PROCESS_ARGS1[] = {"/C", "copy " + INPUT_FILE + " ."};
const std::string& PROCESS_PATH2 = PROCESS_PATH1;
const std::string PROCESS_ARGS2[] = {"/C", "ping 127.0.0.1 -n 11"};
#else
const std::string INPUT_FILE("testfiles/withNs.xml");
const size_t EXPECTED_FILE_SIZE(563);
const std::string PROCESS_PATH1("/bin/dd");
const std::string PROCESS_ARGS1[] = {
    "if=" + INPUT_FILE, "of=" + OUTPUT_FILE, "bs=1",
    "count=" + ml::core::CStringUtils::typeToString(EXPECTED_FILE_SIZE)};
const std::string PROCESS_PATH2("/bin/sleep");
const std::string PROCESS_ARGS2[] = {"10"};
#endif
}

BOOST_AUTO_TEST_CASE(testSpawn) {
    // The intention of this test is to copy a file by spawning an external
    // program and then make sure the file has been copied

    // Remove any output file left behind by a previous failed test, but don't
    // check the return code as this will usually fail
    std::remove(OUTPUT_FILE.c_str());

    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args(
        PROCESS_ARGS1, PROCESS_ARGS1 + std::size(PROCESS_ARGS1));

    BOOST_TEST_REQUIRE(spawner.spawn(PROCESS_PATH1, args));

    // Expect the copy to complete in less than 1 second
    std::this_thread::sleep_for(std::chrono::seconds(1));

    ml::core::COsFileFuncs::TStat statBuf;
    BOOST_REQUIRE_EQUAL(0, ml::core::COsFileFuncs::stat(OUTPUT_FILE.c_str(), &statBuf));
    BOOST_REQUIRE_EQUAL(EXPECTED_FILE_SIZE, static_cast<size_t>(statBuf.st_size));

    BOOST_REQUIRE_EQUAL(0, std::remove(OUTPUT_FILE.c_str()));
}

BOOST_AUTO_TEST_CASE(testKill) {
    // The intention of this test is to spawn a process that sleeps for 10
    // seconds, but kill it before it exits by itself and prove that its death
    // has been detected

    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH2);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    ml::core::CDetachedProcessSpawner::TStrVec args(
        PROCESS_ARGS2, PROCESS_ARGS2 + std::size(PROCESS_ARGS2));

    ml::core::CProcess::TPid childPid = 0;
    BOOST_TEST_REQUIRE(spawner.spawn(PROCESS_PATH2, args, childPid));

    BOOST_TEST_REQUIRE(spawner.hasChild(childPid));
    BOOST_TEST_REQUIRE(spawner.terminateChild(childPid));

    // The spawner should detect the death of the process within half a second
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    BOOST_TEST_REQUIRE(!spawner.hasChild(childPid));

    // We shouldn't be able to kill an already killed process
    BOOST_TEST_REQUIRE(!spawner.terminateChild(childPid));

    // We shouldn't be able to kill processes we didn't start
    BOOST_TEST_REQUIRE(!spawner.terminateChild(1));
    BOOST_TEST_REQUIRE(!spawner.terminateChild(0));
    BOOST_TEST_REQUIRE(!spawner.terminateChild(static_cast<ml::core::CProcess::TPid>(-1)));
}

BOOST_AUTO_TEST_CASE(testPermitted) {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, PROCESS_PATH1);
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    // Should fail as ml_test is not on the permitted processes list
    BOOST_TEST_REQUIRE(
        !spawner.spawn("./ml_test", ml::core::CDetachedProcessSpawner::TStrVec()));
}

BOOST_AUTO_TEST_CASE(testNonExistent) {
    ml::core::CDetachedProcessSpawner::TStrVec permittedPaths(1, "./does_not_exist");
    ml::core::CDetachedProcessSpawner spawner(permittedPaths);

    // Should fail as even though it's a permitted process as the file doesn't exist
    BOOST_TEST_REQUIRE(!spawner.spawn(
        "./does_not_exist", ml::core::CDetachedProcessSpawner::TStrVec()));
}

BOOST_AUTO_TEST_SUITE_END()
