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

#include <maths/common/CMultinomialConjugate.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/FunctionTypes.h>
#include <model/ModelTypes.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CAnnotatedProbabilityBuilderTest)

using namespace ml;
using namespace model;

namespace {
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble4Vec = core::CSmallVector<double, 4>;
using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;
using TOptionalStr = std::optional<std::string>;
using TOptionalStr1Vec = core::CSmallVector<TOptionalStr, 1>;

const TOptionalStr1Vec NO_CORRELATED_ATTRIBUTES;
const TSizeDoublePr1Vec NO_CORRELATES;

class CAnnotatedProbabilityBuilderForTest : public CAnnotatedProbabilityBuilder {
public:
    explicit CAnnotatedProbabilityBuilderForTest(SAnnotatedProbability& annotatedProbability)
        : CAnnotatedProbabilityBuilder(annotatedProbability) {}

    CAnnotatedProbabilityBuilderForTest(SAnnotatedProbability& annotatedProbability,
                                        std::size_t numberAttributeProbabilities,
                                        function_t::EFunction function)
        : CAnnotatedProbabilityBuilder(annotatedProbability, numberAttributeProbabilities, function) {
    }
};
}

BOOST_AUTO_TEST_CASE(testProbability) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result);

    builder.probability(0.42);
    BOOST_REQUIRE_EQUAL(0.42, result.s_Probability);

    builder.probability(0.99);
    BOOST_REQUIRE_EQUAL(0.99, result.s_Probability);
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenIndividualCount) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 1, function_t::E_IndividualCount);

    builder.addAttributeProbability(0, "", 0.68, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualCountByBucketAndPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(1, result.s_AttributeProbabilities.size());
    BOOST_REQUIRE_EQUAL("", *result.s_AttributeProbabilities[0].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.68, result.s_AttributeProbabilities[0].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualCountByBucketAndPerson,
                        result.s_AttributeProbabilities[0].s_Feature);
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenPopulationCount) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 3, function_t::E_PopulationCount);

    builder.addAttributeProbability(0, "", 0.09, model_t::CResultType::E_Unconditional,
                                    model_t::E_PopulationCountByBucketPersonAndAttribute,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(1, "c1", 0.05, model_t::CResultType::E_Unconditional,
                                    model_t::E_PopulationCountByBucketPersonAndAttribute,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(2, "c2", 0.04, model_t::CResultType::E_Unconditional,
                                    model_t::E_PopulationCountByBucketPersonAndAttribute,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(3, "c3", 0.06, model_t::CResultType::E_Unconditional,
                                    model_t::E_PopulationCountByBucketPersonAndAttribute,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(2, result.s_AttributeProbabilities.size());

    BOOST_REQUIRE_EQUAL("c2", *result.s_AttributeProbabilities[0].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.04, result.s_AttributeProbabilities[0].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_PopulationCountByBucketPersonAndAttribute,
                        result.s_AttributeProbabilities[0].s_Feature);

    BOOST_REQUIRE_EQUAL("c1", *result.s_AttributeProbabilities[1].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.05, result.s_AttributeProbabilities[1].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_PopulationCountByBucketPersonAndAttribute,
                        result.s_AttributeProbabilities[1].s_Feature);
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenIndividualRare) {
    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 1, function_t::E_IndividualRare);

    builder.addAttributeProbability(0, "", 0.68, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(1, result.s_AttributeProbabilities.size());
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenPopulationRare) {
    maths::common::CMultinomialConjugate attributePrior(
        maths::common::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1; i <= 4; ++i) {
        TDouble1Vec samples(i, static_cast<double>(i));
        maths_t::TDoubleWeightsAry1Vec weights(i, maths_t::CUnitWeights::UNIT);
        attributePrior.addSamples(samples, weights);
    }

    maths::common::CMultinomialConjugate personAttributePrior(
        maths::common::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1; i <= 4; ++i) {
        TDouble1Vec samples(2 * i, static_cast<double>(i));
        maths_t::TDoubleWeightsAry1Vec weights(2 * i, maths_t::CUnitWeights::UNIT);
        personAttributePrior.addSamples(samples, weights);
    }

    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 2, function_t::E_PopulationRare);
    builder.attributeProbabilityPrior(&attributePrior);
    builder.personAttributeProbabilityPrior(&personAttributePrior);

    builder.addAttributeProbability(1, "c1", 0.02, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(2, "c2", 0.06, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(3, "c3", 0.01, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(4, "c4", 0.03, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(2, result.s_AttributeProbabilities.size());
    BOOST_REQUIRE_EQUAL("c3", *result.s_AttributeProbabilities[0].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.01, result.s_AttributeProbabilities[0].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                        result.s_AttributeProbabilities[0].s_Feature);
    BOOST_REQUIRE_EQUAL("c1", *result.s_AttributeProbabilities[1].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.02, result.s_AttributeProbabilities[1].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                        result.s_AttributeProbabilities[1].s_Feature);
}

BOOST_AUTO_TEST_CASE(testAddAttributeProbabilityGivenPopulationFreqRare) {
    maths::common::CMultinomialConjugate attributePrior(
        maths::common::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1; i <= 4; ++i) {
        TDouble1Vec samples(i, static_cast<double>(i));
        maths_t::TDoubleWeightsAry1Vec weights(i, maths_t::CUnitWeights::UNIT);
        attributePrior.addSamples(samples, weights);
    }

    maths::common::CMultinomialConjugate personAttributePrior(
        maths::common::CMultinomialConjugate::nonInformativePrior(4u));
    for (std::size_t i = 1; i <= 4; ++i) {
        TDouble1Vec samples(2 * i, static_cast<double>(i));
        maths_t::TDoubleWeightsAry1Vec weights(2 * i, maths_t::CUnitWeights::UNIT);
        personAttributePrior.addSamples(samples, weights);
    }

    SAnnotatedProbability result;
    CAnnotatedProbabilityBuilderForTest builder(result, 2, function_t::E_PopulationFreqRare);
    builder.attributeProbabilityPrior(&attributePrior);
    builder.personAttributeProbabilityPrior(&personAttributePrior);

    builder.addAttributeProbability(1, "c1", 0.02, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(2, "c2", 0.06, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(3, "c3", 0.01, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.addAttributeProbability(4, "c4", 0.03, model_t::CResultType::E_Unconditional,
                                    model_t::E_IndividualIndicatorOfBucketPerson,
                                    NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
    builder.build();

    BOOST_REQUIRE_EQUAL(2, result.s_AttributeProbabilities.size());
    BOOST_REQUIRE_EQUAL("c3", *result.s_AttributeProbabilities[0].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.01, result.s_AttributeProbabilities[0].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                        result.s_AttributeProbabilities[0].s_Feature);
    BOOST_REQUIRE_EQUAL("c1", *result.s_AttributeProbabilities[1].s_Attribute);
    BOOST_REQUIRE_EQUAL(0.02, result.s_AttributeProbabilities[1].s_Probability);
    BOOST_REQUIRE_EQUAL(model_t::E_IndividualIndicatorOfBucketPerson,
                        result.s_AttributeProbabilities[1].s_Feature);
}

BOOST_AUTO_TEST_SUITE_END()
