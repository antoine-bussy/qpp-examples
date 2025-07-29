#pragma once
/*!
@file
Macros for types comparison with googletest.
 */

#include <gtest/gtest.h>
#include "compare.hpp"


//! @brief Matrix equality GTest macro
#define EXPECT_MATRIX_EQ(actual, expected) EXPECT_PRED2(qube::maths::matrix_equal_l, actual, expected)

//! @brief Matrix closeness GTest macro
#define EXPECT_MATRIX_CLOSE(actual, expected, precision) EXPECT_PRED3(qube::maths::matrix_close_l, actual, expected, precision)

//! @brief Matrix non closeness GTest macro
#define EXPECT_MATRIX_NOT_CLOSE(actual, expected, precision) EXPECT_PRED3(qube::maths::matrix_not_close_l, actual, expected, precision)

//! @brief Complex closeness GTest macro
#define EXPECT_COMPLEX_CLOSE(actual, expected, precision) EXPECT_PRED3(qube::maths::complex_close_l, actual, expected, precision)

//! @brief Complex non closeness GTest macro
#define EXPECT_COMPLEX_NOT_CLOSE(actual, expected, precision) EXPECT_PRED3(qube::maths::complex_not_close_l, actual, expected, precision)

//! @brief Collinearity GTest macro
#define EXPECT_COLLINEAR(actual, expected, precision) EXPECT_PRED3(qube::maths::collinear_l, actual, expected, precision)

//! @brief Phase factor collinearity GTest macro
#define EXPECT_PHASE_COLLINEAR(actual, expected, precision) EXPECT_PRED3(qube::maths::phase_collinear_l, actual, expected, precision)

//! @brief Matrix closeness GTest macro, up to factor
#define EXPECT_MATRIX_CLOSE_UP_TO_FACTOR(actual, expected, precision) EXPECT_PRED3(qube::maths::matrix_close_up_to_factor_l, actual, expected, precision)

//! @brief Matrix closeness GTest macro, up to phase factor
#define EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(actual, expected, precision) EXPECT_PRED3(qube::maths::matrix_close_up_to_phase_factor_l, actual, expected, precision)
