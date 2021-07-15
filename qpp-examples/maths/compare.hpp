#pragma once
/*!
@file
Matrix comparison functions.
 */

#include "concepts.hpp"

namespace qpp_e::maths
{

    auto matrix_equal(Matrix auto const& actual, Matrix auto const& expected)
    {
        return actual.rows() == expected.rows()
            && actual.cols() == expected.cols()
            && actual == expected;
    }
    auto constexpr matrix_equal_l = [](auto const& actual, auto const& expected) { return matrix_equal(actual, expected); };

    auto matrix_close(Matrix auto const& actual, Matrix auto const& expected, RealNumber auto const& precision)
    {
        return actual.rows() == expected.rows()
            && actual.cols() == expected.cols()
            && actual.isApprox(expected, precision);
    }
    auto constexpr matrix_close_l = [](auto const& actual, auto const& expected, auto const& precision) { return matrix_close(actual, expected, precision); };
    auto constexpr matrix_not_close_l = [](auto const& actual, auto const& expected, auto const& precision) { return !matrix_close(actual, expected, precision); };

    auto complex_close(ComplexNumber auto const& actual, ComplexNumber auto const& expected, RealNumber auto const& precision)
    {
        return std::norm(actual - expected) <= precision * std::min(std::norm(actual), std::norm(expected));
    }
    auto constexpr complex_close_l = [](auto const& actual, auto const& expected, auto const& precision) { return complex_close(actual, expected, precision); };

    auto collinear(Matrix auto const& actual, Matrix auto const& expected, RealNumber auto const& precision)
    {
        auto const n2 = actual.squaredNorm() * expected.squaredNorm();
        return n2 - std::norm(actual.dot(expected)) < precision * n2;
    }
    auto constexpr collinear_l = [](auto&& actual, auto&& expected, auto&& precision) { return collinear(actual, expected, precision); };

} /* namespace qpp_e::maths */
