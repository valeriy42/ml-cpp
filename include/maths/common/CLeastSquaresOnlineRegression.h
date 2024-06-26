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

#ifndef INCLUDED_ml_maths_common_CLeastSquaresOnlineRegression_h
#define INCLUDED_ml_maths_common_CLeastSquaresOnlineRegression_h

#include <core/CoreTypes.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CCategoricalTools.h>
#include <maths/common/CLinearAlgebra.h>
#include <maths/common/CLinearAlgebraTools.h>
#include <maths/common/CTools.h>
#include <maths/common/ImportExport.h>

#include <boost/operators.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace CTimeSeriesDecompositionTest {
class CNanInjector;
}

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace common {
namespace least_squares_online_regression_detail {

//! Used for getting the default maximum condition number to use
//! when computing parameters.
template<typename T>
struct CMaxCondition {
    static constexpr double VALUE{1e15};
};

//! Used for getting the default maximum condition number to use
//! when computing parameters.
template<>
struct MATHS_COMMON_EXPORT CMaxCondition<CFloatStorage> {
    static const double VALUE;
};

//! The number of statistics needed by the regression model.
constexpr std::size_t numberStatistics(std::size_t n, bool r2) {
    return r2 ? 3 * n : 3 * n - 1;
}
}

//! DESCRIPTION:\n
//! A very lightweight online weighted least squares regression to
//! fit degree N polynomials to a collection of points \f$\{(x_i, y_i)\}\f$,
//! i.e. to find the \f$y = c_0 + c_1 x + ... + c_N x^N\f$ s.t. the
//! weighted sum of the square residuals is minimized. Formally, we
//! are looking for \f$\theta^*\f$ defined as
//! <pre class="fragment">
//!   \f$\theta^* = arg\min_{\theta}{(y - X\theta)^tDiag(w)(y - X\theta)}\f$
//! </pre>
//! Here, \f$X\f$ denotes the design matrix and for a polynomial
//! takes the form \f$[X]_{ij} = x_i^{j-1}\f$. This is solved using
//! the Moore-Penrose pseudo-inverse.
//!
//! We are able to maintain \f$2N-1\f$ sufficient statistics to
//! construct \f$X^tDiag(w)X\f$ and also the \f$N\f$ components of
//! the vector \f$X^tDiag(w)y\f$ online.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This uses float storage and requires \f$3(N+1)\f$ floats where
//! \f$N\f$ is the polynomial order. In total this therefore uses
//! \f$12(N+1)\f$ bytes.
//!
//! Note that this constructs the Gramian \f$X^tDiag(w)X\f$ of the
//! design matrix when computing the least squares solution. This
//! is because holding sufficient statistics for constructing this
//! matrix is the most space efficient representation to compute
//! online. However, the condition of this matrix is the square of
//! the condition of the design matrix and so this approach doesn't
//! have good numerics.
//!
//! A much more robust scheme is to use incremental QR factorization
//! and for large problems that approach should be used in preference.
//! However, much can be done by using an affine transformation of
//! \f$x_i\f$ to improve the numerics of this approach and the intention
//! is that it is used for the case where \f$N\f$ is small and space
//! is at a premium.
//!
//! \tparam N_ The degree of the polynomial.
//! \tparam T The storage type used for the statistics. Note that since
//! we can't avoid losing precision in our formulation be careful using
//! single precision if the polynomial degree is high.
//! \tparam R_2 If this is true you can additionally compute R^2 and
//! the residual statistics, but it adds sizeof(T) to the memory usage.
// clang-format off
template<std::size_t N_, typename T = CFloatStorage, bool R_2 = false>
class CLeastSquaresOnlineRegression : boost::addable<CLeastSquaresOnlineRegression<N_, T>,
                                      boost::subtractable<CLeastSquaresOnlineRegression<N_, T>>> {
    // clang-format on
public:
    static const std::size_t N = N_ + 1;
    using TArray = std::array<double, N>;
    using TVector =
        CVectorNx1<T, least_squares_online_regression_detail::numberStatistics(N, R_2)>;
    using TMatrix = CSymmetricMatrixNxN<double, N>;
    using TVectorMeanAccumulator = typename CBasicStatistics::SSampleMean<TVector>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

public:
    static const core::TPersistenceTag STATISTIC_TAG;
    static const T MAX_CONDITION;

public:
    // This is purposely not explicit to allow type coercion.
    CLeastSquaresOnlineRegression() = default;
    template<typename U>
    CLeastSquaresOnlineRegression(const CLeastSquaresOnlineRegression<N_, U, R_2>& other)
        : m_S{other.statistic()} {}

    //! Restore by traversing a state document.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Add in the point \f$(x, y(x))\f$ with weight \p weight.
    //!
    //! \param[in] x The abscissa of the point.
    //! \param[in] y The ordinate of the point.
    //! \param[in] weight The residual weight at the point.
    void add(double x, double y, double weight = 1.0) {
        TVector d;
        double xi{1.0};
        for (std::size_t i = 0; i < N; ++i, xi *= x) {
            d(i) = xi;
            d(i + 2 * N - 1) = xi * y;
        }
        for (std::size_t i = N; i < 2 * N - 1; ++i, xi *= x) {
            d(i) = xi;
        }
        if constexpr (R_2) {
            d(3 * N - 1) = y * y;
        }
        m_S.add(d, weight);
    }

    //! Set the statistics from \p rhs.
    template<typename U>
    CLeastSquaresOnlineRegression&
    operator=(const CLeastSquaresOnlineRegression<N_, U, R_2>& rhs) {
        m_S = rhs.statistic();
        return *this;
    }

    //! Differences two regressions.
    //!
    //! This creates a regression which is fit on just the points
    //! add to this and not \p rhs.
    //!
    //! \param[in] rhs The regression fit to combine.
    //! \note This is only meaningful if they have the same time
    //! origin and the values added to \p rhs are a subset of the
    //! values add to this.
    template<typename U>
    const CLeastSquaresOnlineRegression&
    operator-=(const CLeastSquaresOnlineRegression<N_, U, R_2>& rhs) {
        m_S -= rhs.statistic();
        return *this;
    }

    //! Combines two regressions.
    //!
    //! This creates the regression fit on the points fit with
    //! \p rhs and the points fit with this regression.
    //!
    //! \param[in] rhs The regression fit to combine.
    //! \note This is only meaningful if they have the same time
    //! origin.
    template<typename U>
    const CLeastSquaresOnlineRegression&
    operator+=(const CLeastSquaresOnlineRegression<N_, U, R_2>& rhs) {
        m_S += rhs.statistic();
        return *this;
    }

    //! In order to get reasonable accuracy, one typically needs to
    //! use an affine transform of the abscissa.
    //!
    //! In particular, one will typically use \f$x \mapsto x - b\f$
    //! rather than \f$x\f$ directly, since \f$b\f$ can be adjusted
    //! to improve the condition of the Gramian.
    //!
    //! If this is running online, then as x increases one wants to
    //! allow the shift \f$b\f$ to increase. This function computes
    //! the impact of a change in \f$b\f$ on the stored statistics.
    //!
    //! \param[in] dx The shift that will subsequently be applied to
    //! the abscissa.
    void shiftAbscissa(double dx);

    //! Translate the ordinates by \p dy.
    //!
    //! \param[in] dy The shift that will subsequently be applied to
    //! the ordinates.
    void shiftOrdinate(double dy) {
        if (CBasicStatistics::count(m_S) > 0.0) {
            const TVector& s{CBasicStatistics::mean(m_S)};
            for (std::size_t i = 0; i < N; ++i) {
                CBasicStatistics::moment<0>(m_S)(i + 2 * N - 1) += s(i) * dy;
            }
        }
    }

    //! Shift the gradient by \p dydx.
    //!
    //! \param[in] dydx The shift that will subsequently be applied to
    //! the derivative of the regression w.r.t. the abscissa.
    void shiftGradient(double dydx) {
        if (CBasicStatistics::count(m_S) > 0.0) {
            const TVector& s{CBasicStatistics::mean(m_S)};
            for (std::size_t i = 0; i < N; ++i) {
                CBasicStatistics::moment<0>(m_S)(i + 2 * N - 1) += s(i + 1) * dydx;
            }
        }
    }

    //! Linearly scale the regression model.
    //!
    //! i.e. apply a transform such that each regression parameter maps
    //! to \p scale times its current value.
    //!
    //! \param[in] scale The scale to apply to the regression parameters.
    void linearScale(double scale) {
        if (CBasicStatistics::count(m_S) > 0.0) {
            for (std::size_t i = 0; i < N; ++i) {
                CBasicStatistics::moment<0>(m_S)(i + 2 * N - 1) *= scale;
            }
            if constexpr (R_2) {
                CBasicStatistics::moment<0>(m_S)(3 * N - 1) *= scale * scale;
            }
        }
    }

    //! Multiply the statistics' count by \p scale.
    CLeastSquaresOnlineRegression scaled(double scale) const {
        CLeastSquaresOnlineRegression result(*this);
        return result.scale(scale);
    }

    //! Scale the statistics' count by \p scale.
    const CLeastSquaresOnlineRegression& scale(double scale) {
        CBasicStatistics::count(m_S) *= scale;
        return *this;
    }

    //! Get the predicted value of the regression model with parameters
    //! \p params at \p x.
    static double predict(const TArray& params, double x) {
        double result{params[0]};
        double xi{x};
        for (std::size_t i = 1; i < params.size(); ++i, xi *= x) {
            result += params[i] * xi;
        }
        return result;
    }

    //! Get the predicted value at \p x.
    double predict(double x, double maxCondition = MAX_CONDITION) const {
        return this->predict(N, x, maxCondition);
    }

    //! Get the predicted value at \p x for the order \p n model.
    double predict(std::size_t n, double x, double maxCondition = MAX_CONDITION) const {
        TArray params;
        this->parameters(n, params, maxCondition);
        return predict(params, x);
    }

    //! Get the regression model R^2 if it can be computed.
    //!
    //! If created with R_2 true then the regression model R^2 can
    //! be computed from the statistics maintained.
    //! \param[in] maxCondition The maximum condition number for
    //! the Gramian this will consider solving. If the condition
    //! is worse than this it'll fit a lower order polynomial.
    //! \param[out] result Filled in with the regression model R^2.
    bool r2(double& result, double maxCondition = MAX_CONDITION) const {

        result = 0.0;

        if constexpr (R_2 == false) {
            return false;
        }
        if (CBasicStatistics::count(m_S) == T{0}) {
            return true;
        }

        TMeanVarAccumulator residualMoments;
        if (this->residualMoments(residualMoments, maxCondition) == false) {
            return false;
        }

        const auto& s = CBasicStatistics::mean(m_S);
        double variance{s(3 * N - 1) - CTools::pow2(s(2 * N - 1))};
        double residualVariance{CBasicStatistics::maximumLikelihoodVariance(residualMoments)};
        result = CTools::truncate(1.0 - residualVariance / variance, 0.0, 1.0);
        return true;
    }

    //! Compute the residual moments.
    //!
    //! \param[in] maxCondition The maximum condition number for
    //! the Gramian this will consider solving. If the condition
    //! is worse than this it'll fit a lower order polynomial.
    //! \param[out] result Filled in with the residual moments.
    //! \note This is only possible if R_2 is true.
    bool residualMoments(TMeanVarAccumulator& result, double maxCondition = MAX_CONDITION) const;

    //! Get the regression parameters.
    //!
    //! i.e. The intercept, slope, curvature, etc.
    //!
    //! \param[in] maxCondition The maximum condition number for
    //! the Gramian this will consider solving. If the condition
    //! is worse than this it'll fit a lower order polynomial.
    //! \param[out] result Filled in with the regression parameters.
    bool parameters(TArray& result, double maxCondition = MAX_CONDITION) const;

    //! Get the regression parameters for a model of order n (<= N).
    bool parameters(std::size_t n, TArray& result, double maxCondition = MAX_CONDITION) const;

    //! Get the predicted value of the regression parameters at \p x.
    //!
    //! \note Returns array of zeros if getting the parameters fails.
    TArray parameters(double x, double maxCondition = MAX_CONDITION) const {
        TArray result;
        TArray params;
        if (this->parameters(params, maxCondition)) {
            std::ptrdiff_t n{static_cast<std::ptrdiff_t>(params.size())};
            for (std::ptrdiff_t i = n - 1; i >= 0; --i) {
                result[i] = params[i];
                for (std::ptrdiff_t j = i + 1; j < n; ++j) {
                    params[j] *= static_cast<double>(i + 1) /
                                 static_cast<double>(j - i) * x;
                    result[i] += params[j];
                }
            }
        }
        return result;
    }

    //! Get the covariance matrix of the regression parameters.
    //!
    //! To compute this assume the data to fit are described by
    //! \f$y_i = \sum_{j=0}{N} c_j x_i^j + Y_i\f$ where \f$Y_i\f$
    //! are IID and \f$N(0, \sigma)\f$ whence
    //! <pre class="fragment">
    //!   \f$C = (X^t X)^{-1}X^t E[YY^t] X (X^t X)^{-1}\f$
    //! </pre>
    //!
    //! Since \f$E[YY^t] = \sigma^2 I\f$ it follows that
    //! <pre class="fragment">
    //!   \f$C = \sigma^2 (X^t X)^{-1}\f$
    //! </pre>
    //!
    //! \param[in] variance The variance of the data residuals.
    //! \param[in] maxCondition The maximum condition number for
    //! the Gramian this will consider solving. If the condition
    //! is worse than this it'll fit a lower order polynomial.
    //! \param[out] result Filled in with the covariance matrix.
    bool covariances(double variance, TMatrix& result, double maxCondition = MAX_CONDITION) const;

    //! Get the parameter covariance matrix for a model of order \p n (<= N).
    bool covariances(std::size_t n, double variance, TMatrix& result, double maxCondition = MAX_CONDITION) const;

    //! Get the safe prediction horizon based on the spread
    //! of the abscissa added to the model so far.
    double range() const {
        // The magic 12 comes from assuming the independent
        // variable X is uniform over the range (for our uses
        // it typically is). We maintain mean X^2 and X. For
        // a uniform variable on a range [a, b] we have that
        // E[(X - E(X))^2] = E[X^2] - E[X]^2 = (b - a)^2 / 12.

        double x1{CBasicStatistics::mean(m_S)(1)};
        double x2{CBasicStatistics::mean(m_S)(2)};
        return std::sqrt(12.0 * std::max(x2 - x1 * x1, 0.0));
    }

    //! Age out the old points.
    void age(double factor, double meanRevertFactor = 0.0) {
        if (meanRevertFactor > 0.0) {
            TVector& s{CBasicStatistics::moment<0>(m_S)};
            double alpha{1.0 - meanRevertFactor * (1.0 - factor)};
            double beta{1.0 - alpha};
            for (std::size_t i = 1; i < N; ++i) {
                s(i + 2 * N - 1) = alpha * s(i + 2 * N - 1) + beta * s(i) * s(2 * N - 1);
            }
        }
        m_S.age(factor);
    }

    //! Get the effective number of points being fitted.
    double count() const { return CBasicStatistics::count(m_S); }

    //! Get the mean value of the ordinates.
    double mean() const { return CBasicStatistics::mean(m_S)(2 * N - 1); }

    //! Get the mean in the interval [\p a, \p b].
    double mean(double a, double b) const {
        double result{0.0};

        double interval{b - a};

        TArray params;
        this->parameters(params);

        if (interval == 0.0) {
            result = params[0];
            double xi{a};
            for (std::size_t i = 1; i < params.size(); ++i, xi *= a) {
                result += params[i] * xi;
            }
            return result;
        }

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                result += CCategoricalTools::binomialCoefficient(i + 1, j + 1) *
                          params[i] / static_cast<double>(i + 1) *
                          std::pow(a, static_cast<double>(i - j)) *
                          std::pow(interval, static_cast<double>(j + 1));
            }
        }

        return result / interval;
    }

    //! Get the vector statistic.
    const TVectorMeanAccumulator& statistic() const { return m_S; }

    //! Get a checksum for this object.
    std::uint64_t checksum() const { return m_S.checksum(); }

    //! Print this regression out for debug.
    std::string print() const;

private:
    //! Compute the residual variance for an order \p n regression model.
    template<typename MATRIX, typename VECTOR>
    bool residualMoments(std::size_t n,
                         MATRIX& x,
                         VECTOR& y,
                         VECTOR& z,
                         double maxCondition,
                         TMeanVarAccumulator& result) const;

    //! Get the first \p n regression parameters.
    template<typename MATRIX, typename VECTOR>
    bool parameters(std::size_t n, MATRIX& x, VECTOR& y, double maxCondition, TArray& result) const;

    //! Compute the covariance matrix of the regression parameters.
    template<typename MATRIX>
    bool covariances(std::size_t n, MATRIX& x, double variance, double maxCondition, TMatrix& result) const;

    //! Get the gramian of the design matrix.
    template<typename MATRIX>
    void gramian(std::size_t n, MATRIX& x) const {
        for (std::size_t i = 0; i < n; ++i) {
            x(i, i) = CBasicStatistics::mean(m_S)(i + i);
            for (std::size_t j = i + 1; j < n; ++j) {
                x(i, j) = CBasicStatistics::mean(m_S)(i + j);
            }
        }
    }

private:
    //! Sufficient statistics for computing the least squares regression.
    //! There are 3N - 1 (or 3N if computing R^2) in total, for the distinct
    //! values in the design matrix and vector.
    TVectorMeanAccumulator m_S;

    //! Befriend a helper class used by the unit tests.
    friend class CTimeSeriesDecompositionTest::CNanInjector;
};

template<std::size_t N_, typename T, bool R_2>
const core::TPersistenceTag
    CLeastSquaresOnlineRegression<N_, T, R_2>::STATISTIC_TAG("a", "statistic");
template<std::size_t N_, typename T, bool R_2>
const T CLeastSquaresOnlineRegression<N_, T, R_2>::MAX_CONDITION{
    least_squares_online_regression_detail::CMaxCondition<T>::VALUE};
}
}
}

#endif // INCLUDED_ml_maths_common_CLeastSquaresOnlineRegression_h
