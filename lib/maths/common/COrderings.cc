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

#include <maths/common/COrderings.h>

namespace ml {
namespace maths {
namespace common {
const std::less<> COrderings::SDerefLess::s_Less;
const std::greater<> COrderings::SDerefGreater::s_Greater;
const std::less<> COrderings::SReferenceLess::s_Less;
const std::greater<> COrderings::SReferenceGreater::s_Greater;
}
}
}
