#pragma once
/*!
@file
Macros for types comparison with googletest.
 */

#include <gtest/gtest.h>
#include "compare.hpp"


//! @brief Matrix equality GTest macro
#define EXPECT_MATRIX_EQ(actual, expected) EXPECT_PRED2(qpp_e::maths::matrix_equal_l, actual, expected)

//! @brief Matrix closeness GTest macro
#define EXPECT_MATRIX_CLOSE(actual, expected, precision) EXPECT_PRED3(qpp_e::maths::matrix_close_l, actual, expected, precision)

//! @brief Matrix non closeness GTest macro
#define EXPECT_MATRIX_NOT_CLOSE(actual, expected, precision) EXPECT_PRED3(qpp_e::maths::matrix_not_close_l, actual, expected, precision)

//! @brief Complex closeness GTest macro
#define EXPECT_COMPLEX_CLOSE(actual, expected, precision) EXPECT_PRED3(qpp_e::maths::complex_close_l, actual, expected, precision)

//! @brief Complex non closeness GTest macro
#define EXPECT_COMPLEX_NOT_CLOSE(actual, expected, precision) EXPECT_PRED3(qpp_e::maths::complex_not_close_l, actual, expected, precision)

//! @brief Collinearity GTest macro
#define EXPECT_COLLINEAR(actual, expected, precision) EXPECT_PRED3(qpp_e::maths::collinear_l, actual, expected, precision)
