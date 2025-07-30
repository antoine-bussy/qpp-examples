#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ranges>

#include <qpp/qpp.hpp>
#include <qube/debug.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/norm.hpp>
#include <qube/maths/random.hpp>

using namespace qube::stream;

//! @brief Equation 4.61
TEST(chapter4_5, operator_error)
{
    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr n = qube::maths::pow(2ul, nq);

    auto const U = qpp::randU(n);
    auto const V = qpp::randU(n);
    auto const W = (U - V).eval();

    auto const error = qube::maths::operator_norm_2(W);
    auto const psi_max = W.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeFullV).matrixV().col(0).eval();
    auto const e_max = qpp::norm(W * psi_max);
    auto constexpr epsilon = 1e-12;
    EXPECT_LT(e_max, error + epsilon);

    for ([[maybe_unused]] auto&& i: std::views::iota(0, 50))
    {
        auto const psi = qpp::randket(n);
        auto const e = (W * psi).norm();
        EXPECT_LT(e, error + epsilon);
    }
}

//! @brief Equation 4.62, and equations 4.64 through 4.68
TEST(chapter4_5_2, probability_error_bound)
{
    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr n = qube::maths::pow(2ul, nq);
    auto constexpr m = 5ul;

    auto const U = qpp::randU(n);
    auto const V = qpp::randU(n);
    auto const error = qube::maths::operator_norm_2(U - V);

    auto const Ks = qpp::randkraus(m, n);
    auto const M = (Ks[2].adjoint() * Ks[2]).eval();

    auto constexpr epsilon = 1e-12;

    for([[maybe_unused]] auto&& i: std::views::iota(0, 50))
    {
        auto const psi = qpp::randket(n);
        auto const [P_U, P_U_imag] = (U * psi).dot(M * (U * psi));
        auto const [P_V, P_V_imag] = (V * psi).dot(M * (V * psi));

        // Check that P_U and P_V are probabilities
        EXPECT_GT(P_U, -epsilon);
        EXPECT_LT(P_U, 1. + epsilon);
        EXPECT_LT(std::abs(P_U_imag), epsilon);
        EXPECT_GT(P_V, -epsilon);
        EXPECT_LT(P_V, 1. + epsilon);
        EXPECT_LT(std::abs(P_V_imag), epsilon);

        EXPECT_LT(std::abs(P_U - P_V), 2 * error + epsilon);
    }

}
