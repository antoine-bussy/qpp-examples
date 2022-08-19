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
    auto constexpr complex_not_close_l = [](auto const& actual, auto const& expected, auto const& precision) { return !complex_close(actual, expected, precision); };

    auto collinear(Matrix auto const& actual, Matrix auto const& expected, RealNumber auto const& precision, bool phase_factor = false)
    {
        auto const n_actual = actual.squaredNorm();
        auto const n_expected = expected.squaredNorm();

        if(phase_factor && !complex_close(n_actual, n_expected, precision))
            return false;

        auto const n2 = n_actual * n_expected;
        return n2 - std::norm(actual.dot(expected)) < precision * n2;
    }
    auto constexpr collinear_l = [](auto&& actual, auto&& expected, auto&& precision) { return collinear(actual, expected, precision); };
    auto constexpr phase_collinear_l = [](auto&& actual, auto&& expected, auto&& precision) { return collinear(actual, expected, precision, true); };

    auto matrix_close_up_to_factor(Matrix auto const& actual, Matrix auto const& expected, RealNumber auto const& precision, bool phase_factor = false)
    {
        if(actual.rows() != expected.rows()
            || actual.cols() != expected.cols())
            return false;

        if(phase_factor)
        {
            auto const n_actual = actual.colwise().squaredNorm().eval();
            auto const n_expected = expected.colwise().squaredNorm().eval();

            if(!matrix_close(n_actual, n_expected, precision))
                return false;
        }

        for(auto i = 0u; i < actual.cols(); ++i)
        {
            if(!collinear(actual.col(i), expected.col(i), precision))
                return false;
        }

        return true;
    }
    auto constexpr matrix_close_up_to_factor_l = [](auto const& actual, auto const& expected, auto const& precision) { return matrix_close_up_to_factor(actual, expected, precision); };
    auto constexpr matrix_close_up_to_phase_factor_l = [](auto const& actual, auto const& expected, auto const& precision) { return matrix_close_up_to_factor(actual, expected, precision, true); };

} /* namespace qpp_e::maths */
