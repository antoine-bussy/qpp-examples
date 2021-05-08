#pragma once
/*!
@file
Macros for matrix types  comparison with googletest.
 */

#include <gtest/gtest.h>


//! @brief Matrix equality GTest macro
#define EXPECT_MATRIX_EQ(actual, expected) \
    EXPECT_PRED2([](auto const& actual_, auto const& expected_) \
    { \
        return actual_.rows() == expected_.rows() \
            && actual_.cols() == expected_.cols() \
            && actual_ == expected_; \
    } \
    , actual, expected)

//! @brief Matrix closeness GTest macro
#define EXPECT_MATRIX_CLOSE(actual, expected, precision) \
    EXPECT_PRED3([](auto const& actual_, auto const& expected_, auto const& precision_) \
    { \
        return actual_.rows() == expected_.rows() \
            && actual_.cols() == expected_.cols() \
            && actual_.isApprox(expected_, precision_); \
    } \
    , actual, expected, precision)
