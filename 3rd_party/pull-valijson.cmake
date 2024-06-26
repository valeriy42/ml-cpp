#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

# Script to get the latest version of Valijson, if not already present.
#
# Valijson must only be used in test code, _not_ any form of redistributable code.

# This cmake script is expected to be called from a target or custom command with WORKING_DIRECTORY set to this file's location

if ( NOT EXISTS valijson )
  execute_process( COMMAND git -c advice.detachedHead=false clone --depth=1 --branch=v1.0.2 https://github.com/tristanpenman/valijson.git WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
endif()
