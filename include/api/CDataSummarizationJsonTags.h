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

#ifndef INCLUDED_ml_api_CDataSummarizationJsonTags_h
#define INCLUDED_ml_api_CDataSummarizationJsonTags_h

#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace api {

//! \brief Shared tags used by the JSON data summarization object.
struct API_EXPORT CDataSummarizationJsonTags {
    static const std::string JSON_CATEGORICAL_COLUMN_VALUES_TAG;
    static const std::string JSON_COLUMN_IS_CATEGORICAL_TAG;
    static const std::string JSON_COLUMN_NAMES_TAG;
    static const std::string JSON_COMPRESSED_DATA_SUMMARIZATION_TAG;
    static const std::string JSON_DATA_SUMMARIZATION_TAG;
    static const std::string JSON_DATA_TAG;
    static const std::string JSON_ENCODING_NAME_INDEX_MAP_KEY_TAG;
    static const std::string JSON_ENCODING_NAME_INDEX_MAP_TAG;
    static const std::string JSON_ENCODING_NAME_INDEX_MAP_VALUE_TAG;
    static const std::string JSON_ENCODINGS_TAG;
    static const std::string JSON_NUM_COLUMNS_TAG;
    static const std::string JSON_NUM_ROWS_TAG;
};
}
}

#endif // INCLUDED_ml_api_CDataSummarizationJsonTags_h
