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

#include <model/CTokenListDataCategorizerBase.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CTokenListDataCategorizerBaseTest)

BOOST_AUTO_TEST_CASE(testMinMatchingWeights) {
    BOOST_REQUIRE_EQUAL(
        0, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(0, 0.7));
    BOOST_REQUIRE_EQUAL(
        1, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(1, 0.7));
    BOOST_REQUIRE_EQUAL(
        2, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(2, 0.7));
    BOOST_REQUIRE_EQUAL(
        3, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(3, 0.7));
    BOOST_REQUIRE_EQUAL(
        3, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(4, 0.7));
    BOOST_REQUIRE_EQUAL(
        4, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(5, 0.7));
    BOOST_REQUIRE_EQUAL(
        5, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(6, 0.7));
    BOOST_REQUIRE_EQUAL(
        5, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(7, 0.7));
    BOOST_REQUIRE_EQUAL(
        6, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(8, 0.7));
    BOOST_REQUIRE_EQUAL(
        7, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(9, 0.7));
    BOOST_REQUIRE_EQUAL(
        8, ml::model::CTokenListDataCategorizerBase::minMatchingWeight(10, 0.7));
}

BOOST_AUTO_TEST_CASE(testMaxMatchingWeights) {
    BOOST_REQUIRE_EQUAL(
        0, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(0, 0.7));
    BOOST_REQUIRE_EQUAL(
        1, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(1, 0.7));
    BOOST_REQUIRE_EQUAL(
        2, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(2, 0.7));
    BOOST_REQUIRE_EQUAL(
        4, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(3, 0.7));
    BOOST_REQUIRE_EQUAL(
        5, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(4, 0.7));
    BOOST_REQUIRE_EQUAL(
        7, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(5, 0.7));
    BOOST_REQUIRE_EQUAL(
        8, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(6, 0.7));
    BOOST_REQUIRE_EQUAL(
        9, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(7, 0.7));
    BOOST_REQUIRE_EQUAL(
        11, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(8, 0.7));
    BOOST_REQUIRE_EQUAL(
        12, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(9, 0.7));
    BOOST_REQUIRE_EQUAL(
        14, ml::model::CTokenListDataCategorizerBase::maxMatchingWeight(10, 0.7));
}

BOOST_AUTO_TEST_CASE(testCalculateCategorizationStatus) {
    BOOST_REQUIRE_EQUAL(ml::model_t::E_CategorizationStatusOk,
                        ml::model::CTokenListDataCategorizerBase::calculateCategorizationStatus(
                            99, 99, 0, 99, 0));

    BOOST_REQUIRE_EQUAL(ml::model_t::E_CategorizationStatusWarn,
                        ml::model::CTokenListDataCategorizerBase::calculateCategorizationStatus(
                            1000, 1, 1, 0, 0));

    BOOST_REQUIRE_EQUAL(ml::model_t::E_CategorizationStatusWarn,
                        ml::model::CTokenListDataCategorizerBase::calculateCategorizationStatus(
                            1000, 100, 3, 91, 1));

    BOOST_REQUIRE_EQUAL(ml::model_t::E_CategorizationStatusWarn,
                        ml::model::CTokenListDataCategorizerBase::calculateCategorizationStatus(
                            1000, 501, 1, 99, 0));

    BOOST_REQUIRE_EQUAL(ml::model_t::E_CategorizationStatusWarn,
                        ml::model::CTokenListDataCategorizerBase::calculateCategorizationStatus(
                            1000, 200, 0, 20, 0));

    BOOST_REQUIRE_EQUAL(ml::model_t::E_CategorizationStatusWarn,
                        ml::model::CTokenListDataCategorizerBase::calculateCategorizationStatus(
                            1000, 300, 2, 50, 151));

    BOOST_REQUIRE_EQUAL(ml::model_t::E_CategorizationStatusOk,
                        ml::model::CTokenListDataCategorizerBase::calculateCategorizationStatus(
                            1000, 120, 20, 40, 1));
}

BOOST_AUTO_TEST_SUITE_END()
