#pragma once
/*!
@file
Matrix comparison functions.
 */

#include "concepts.hpp"

namespace qpp_e::maths
{

    template < Matrix Actual, Matrix Expected >
    auto matrix_equal(Actual const& actual, Expected const& expected)
    {
        return actual.rows() == expected.rows()
            && actual.cols() == expected.cols()
            && actual == expected;
    }
    auto constexpr matrix_equal_l = [](auto const& actual, auto const& expected) { return matrix_equal(actual, expected); };

    template < Matrix Actual, Matrix Expected, RealNumber Precision >
    auto matrix_close(Actual const& actual, Expected const& expected, Precision const& precision)
    {
        return actual.rows() == expected.rows()
            && actual.cols() == expected.cols()
            && actual.isApprox(expected, precision);
    }
    auto constexpr matrix_close_l = [](auto const& actual, auto const& expected, auto const& precision) { return matrix_close(actual, expected, precision); };

    template < ComplexNumber Actual, ComplexNumber Expected, RealNumber Precision >
    auto complex_close(Actual const& actual, Expected const& expected, Precision const& precision)
    {
        return std::norm(actual - expected) <= std::pow(precision, 2) * std::min(std::norm(actual), std::norm(expected));
    }
    auto constexpr complex_close_l = [](auto const& actual, auto const& expected, auto const& precision) { return complex_close(actual, expected, precision); };

} /* namespace qpp_e::maths */
