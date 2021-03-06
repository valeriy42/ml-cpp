#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# The non-Windows part of ML C++ CI does the following:
#
# 1. If this is not a PR build nor a debug build, obtain credentials from Vault
#    for the accessing S3
# 2. Build and unit test the C++ on the native architecture
# 3. For Linux PR builds, also run some Java integration tests using the newly
#    built C++ code
# 4. If this is not a PR build nor a debug build, upload the builds to the
#    artifacts directory on S3 that subsequent Java builds will download the C++
#    components from
#
# On Linux all steps run in Docker containers that ensure OS dependencies
# are appropriate given the support matrix.
#
# On macOS the build runs on the native machine, but downloads dependencies
# that were previously built on a reference build server.  However, care still
# needs to be taken that the machines running this script are set up
# appropriately for generating builds for redistribution.

: "${HOME:?Need to set HOME to a non-empty value.}"
: "${WORKSPACE:?Need to set WORKSPACE to a non-empty value.}"

set +x

# Change directory to the directory containing this script
cd "$(dirname $0)"

# If this isn't a PR build or a debug build then obtain credentials from Vault
if [[ -z "$PR_AUTHOR" && -z "$ML_DEBUG" ]] ; then
    . ./aws_creds_from_vault.sh
fi

set -e

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

# Default to running tests
if [ -z "$RUN_TESTS" ] ; then
    RUN_TESTS=true
fi

VERSION=$(cat ../gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo)
HARDWARE_ARCH=$(uname -m | sed 's/arm64/aarch64/')

# Jenkins sets BUILD_SNAPSHOT, but our Docker scripts require SNAPSHOT
if [ "$BUILD_SNAPSHOT" = false ] ; then
    export SNAPSHOT=no
else
    export SNAPSHOT=yes
    VERSION=${VERSION}-SNAPSHOT
fi

# Version qualifier shouldn't be used in PR builds
if [[ -n "$PR_AUTHOR" && -n "$VERSION_QUALIFIER" ]] ; then
    echo "VERSION_QUALIFIER should not be set in PR builds: was $VERSION_QUALIFIER"
    exit 2
fi

# Tests must be run in PR builds
if [[ -n "$PR_AUTHOR" && "$RUN_TESTS" = false ]] ; then
    echo "RUN_TESTS should not be false PR builds"
    exit 3
fi

# Remove any old builds
rm -rf ../builds

# Disassociate from reference repo
git repack -a -d
readonly GIT_TOPLEVEL=$(git rev-parse --show-toplevel 2> /dev/null)
rm -f "${GIT_TOPLEVEL}/.git/objects/info/alternates"

case `uname` in

    Darwin)
        # For macOS, build directly on the machine
        ./download_macos_deps.sh
        if [ -z "$PR_AUTHOR" ] ; then
            if [ "$RUN_TESTS" = false ] ; then
                TASKS="clean buildZip buildZipSymbols"
            else
                TASKS="clean buildZip buildZipSymbols check"
            fi
        else
            TASKS="clean buildZip check"
        fi
        (cd .. && ./gradlew --info -Dbuild.version_qualifier=$VERSION_QUALIFIER -Dbuild.snapshot=$BUILD_SNAPSHOT -Dbuild.ml_debug=$ML_DEBUG $TASKS)
        ;;

    Linux)
        # On Linux build using Docker

        # The Docker version is helpful to identify version-specific Docker bugs
        docker --version

        # Build and test the native Linux architecture
        if [ "$HARDWARE_ARCH" = x86_64 ] ; then

            if [ "$RUN_TESTS" = false ] ; then
                ./docker_build.sh linux
            else
                ./docker_test.sh linux
            fi

        elif [ "$HARDWARE_ARCH" = aarch64 ] ; then
            if [ "$RUN_TESTS" = false ] ; then
                ./docker_build.sh linux_aarch64_native
            else
                ./docker_test.sh linux_aarch64_native
            fi
        fi

        # If this is a PR build then run some Java integration tests
        if [ -n "$PR_AUTHOR" ] ; then
            IVY_REPO="${GIT_TOPLEVEL}/../ivy"
            mkdir -p "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION"
            cp "../build/distributions/ml-cpp-$VERSION-linux-$HARDWARE_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION.zip"
            ./run_es_tests.sh "${GIT_TOPLEVEL}/.." "$(cd "${IVY_REPO}" && pwd)"
        fi
        ;;

    *)
        echo `uname 2>&1` "- unsupported operating system"
        exit 4
        ;;
esac

# If this isn't a PR build and isn't a debug build then upload the artifacts
if [[ -z "$PR_AUTHOR" && -z "$ML_DEBUG" ]] ; then
    (cd .. && ./gradlew --info -b upload.gradle -Dbuild.version_qualifier=$VERSION_QUALIFIER -Dbuild.snapshot=$BUILD_SNAPSHOT upload)
fi

