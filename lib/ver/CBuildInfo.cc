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
#include <ver/CBuildInfo.h>

#include <core/CProgName.h>
#include <core/CStringUtils.h>

namespace ml {
namespace ver {

// Initialise static strings
// Variables are supplied by command line macro definitions
#ifdef DEV_BUILD
const std::string CBuildInfo::VERSION_NUMBER("based on " STRINGIFY_MACRO(PRODUCT_VERSION));
const std::string CBuildInfo::BUILD_NUMBER("DEVELOPMENT BUILD by " STRINGIFY_MACRO(ML_USER));
#else
const std::string CBuildInfo::VERSION_NUMBER(STRINGIFY_MACRO(PRODUCT_VERSION));
const std::string CBuildInfo::BUILD_NUMBER(STRINGIFY_MACRO(ML_BUILD_STR));
#endif
const std::string CBuildInfo::COPYRIGHT("Copyright (c) " STRINGIFY_MACRO(BUILD_YEAR) " Elasticsearch BV");

const std::string& CBuildInfo::versionNumber() {
    return VERSION_NUMBER;
}

const std::string& CBuildInfo::buildNumber() {
    return BUILD_NUMBER;
}

const std::string& CBuildInfo::copyright() {
    return COPYRIGHT;
}

std::string CBuildInfo::fullInfo() {
    static const size_t BITS_PER_BYTE(8);

    return core::CProgName::progName() + " (" +
           core::CStringUtils::typeToString(sizeof(void*) * BITS_PER_BYTE) +
           " bit): Version " + VERSION_NUMBER + " (Build " + BUILD_NUMBER + ") " + COPYRIGHT;
}
}
}
