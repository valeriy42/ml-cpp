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

#ifndef INCLUDED_ml_maths_common_CMathsFuncsForMatrixAndVectorTypes_h
#define INCLUDED_ml_maths_common_CMathsFuncsForMatrixAndVectorTypes_h

#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CMathsFuncs.h>

#include <algorithm>

namespace ml {
namespace maths {
namespace common {
template<typename VECTOR, typename F>
bool CMathsFuncs::aComponent(const F& f, const VECTOR& val) {
    for (std::size_t i = 0; i < val.dimension(); ++i) {
        if (f(val(i))) {
            return true;
        }
    }
    return false;
}

template<typename VECTOR, typename F>
bool CMathsFuncs::everyComponent(const F& f, const VECTOR& val) {
    for (std::size_t i = 0; i < val.dimension(); ++i) {
        if (!f(val(i))) {
            return false;
        }
    }
    return true;
}

template<typename SYMMETRIC_MATRIX, typename F>
bool CMathsFuncs::anElement(const F& f, const SYMMETRIC_MATRIX& val) {
    for (std::size_t i = 0; i < val.rows(); ++i) {
        for (std::size_t j = i; j < val.columns(); ++j) {
            if (f(val(i, j))) {
                return true;
            }
        }
    }
    return false;
}

template<typename SYMMETRIC_MATRIX, typename F>
bool CMathsFuncs::everyElement(const F& f, const SYMMETRIC_MATRIX& val) {
    for (std::size_t i = 0; i < val.rows(); ++i) {
        for (std::size_t j = i; j < val.columns(); ++j) {
            if (!f(val(i, j))) {
                return false;
            }
        }
    }
    return true;
}

template<std::size_t N>
bool CMathsFuncs::isNan(const CVectorNx1<double, N>& val) {
    return aComponent(static_cast<bool (*)(double)>(&isNan), val);
}

template<std::size_t N>
bool CMathsFuncs::isNan(const CSymmetricMatrixNxN<double, N>& val) {
    return anElement(static_cast<bool (*)(double)>(&isNan), val);
}

template<typename T, std::size_t N>
bool CMathsFuncs::isNan(const std::array<T, N>& val) {
    return std::any_of(val.begin(), val.end(),
                       [](const auto& x) { return isNan(x); });
}

template<std::size_t N>
bool CMathsFuncs::isInf(const CVectorNx1<double, N>& val) {
    return aComponent(static_cast<bool (*)(double)>(&isInf), val);
}

template<std::size_t N>
bool CMathsFuncs::isInf(const CSymmetricMatrixNxN<double, N>& val) {
    return anElement(static_cast<bool (*)(double)>(&isInf), val);
}

template<typename T, std::size_t N>
bool CMathsFuncs::isInf(const std::array<T, N>& val) {
    return std::any_of(val.begin(), val.end(),
                       [](const auto& x) { return isInf(x); });
}

template<std::size_t N>
bool CMathsFuncs::isFinite(const CVectorNx1<double, N>& val) {
    return everyComponent(static_cast<bool (*)(double)>(&isFinite), val);
}

template<std::size_t N>
bool CMathsFuncs::isFinite(const CSymmetricMatrixNxN<double, N>& val) {
    return everyElement(static_cast<bool (*)(double)>(&isFinite), val);
}

template<typename T, std::size_t N>
bool CMathsFuncs::isFinite(const std::array<T, N>& val) {
    return std::all_of(val.begin(), val.end(),
                       [](const auto& x) { return isFinite(x); });
}
}
}
}

#endif // INCLUDED_ml_maths_common_CMathsFuncsForMatrixAndVectorTypes_h
