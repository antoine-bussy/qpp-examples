#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ranges>

#include <qpp/qpp.hpp>
#include <qube/debug.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/gtest_macros.hpp>
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

//! @brief Equation 4.63, and equations 4.69 through 4.73
TEST(chapter4_5, operator_error_composition)
{
    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr n = qube::maths::pow(2ul, nq);
    auto constexpr m = 5;

    auto composed_error = 0.;
    auto U = Eigen::MatrixXcd::Identity(n, n).eval();
    auto V = Eigen::MatrixXcd::Identity(n, n).eval();

    for ([[maybe_unused]] auto&& i: std::views::iota(0, m))
    {
        auto const U_i = qpp::randU(n);
        auto const V_i = qpp::randU(n);
        composed_error += qube::maths::operator_norm_2(U_i - V_i);
        U = U_i * U;
        V = V_i * V;
    }

    auto const error = qube::maths::operator_norm_2(U - V);
    auto constexpr epsilon = 1e-12;
    EXPECT_LT(error, composed_error + epsilon);
}

//! @brief Equations 4.74 and 4.75
TEST(chapter4_5, H_T_phase_CNOT_universality)
{
    using namespace std::complex_literals;

    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto const T = (c * qpp::gt.Id2 - 1.i * s * qpp::gt.Z).eval();
    EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(T, qpp::gt.T, 1e-12);
    auto const HTH = (c * qpp::gt.Id2 - 1.i * s * qpp::gt.X).eval();
    EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(HTH, qpp::gt.H * qpp::gt.T * qpp::gt.H, 1e-12);

    auto const THTH = (c * c * qpp::gt.Id2 - 1.i * s * (c * (qpp::gt.X + qpp::gt.Z) + s * qpp::gt.Y)).eval();
    EXPECT_MATRIX_CLOSE(T * HTH, THTH, 1e-12);

    auto n = std::array{c, s, c};
    Eigen::Vector3d::Map(n.data()).normalize();
    auto constexpr theta = 2. * std::acos(c*c);
    auto const Rn_theta = qpp::gt.Rn(theta, n).eval();
    EXPECT_MATRIX_CLOSE(Rn_theta, THTH, 1e-12);
}

//! @brief Exercise 4.40
TEST(chapter4_5, H_T_phase_CNOT_universality_2)
{
    using namespace std::complex_literals;

    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto n = std::array{c, s, c};
    Eigen::Vector3d::Map(n.data()).normalize();

    auto const alpha = qpp::rand(0., 2. * pi);
    auto const beta = qpp::rand(0., 2. * pi);

    auto const error = qube::maths::operator_norm_2(qpp::gt.Rn(alpha, n) - qpp::gt.Rn(alpha + beta, n));
    auto const expected_error = std::abs(1. - std::exp(0.5i * beta));
    EXPECT_NEAR(error, expected_error, 1e-12);
}

//! @brief Equations 4.76 and 4.77
TEST(chapter4_5, H_T_phase_CNOT_universality_3)
{
    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto n = std::array{c, s, c};
    Eigen::Vector3d::Map(n.data()).normalize();
    auto constexpr theta = 2. * std::acos(c*c);

    auto alpha = qpp::rand(0., 2. * pi);
    // Don't ask for too high a precision, otherwise the test will reach floating-point precision limits
    auto constexpr epsilon = 1e-4;

    auto constexpr beta = 4. * std::asin(epsilon / 6.);
    auto constexpr delta = beta;
    auto constexpr N = static_cast<unsigned long int>(std::ceil(2. * pi / delta)) + 2ul;

    auto const R = std::views::iota(1ul, N) |
        std::views::transform([&](auto&& k)
        {
            return std::abs(std::fmod(k * theta, 2. * pi));
        });
    auto const k = 1ul + static_cast<unsigned long int>(std::ranges::distance(R.cbegin(), std::ranges::min_element(R)));
    auto const theta_k = std::fmod(k * theta, 2. * pi);

    EXPECT_LT(std::abs(theta_k), delta);
    EXPECT_NE(theta_k, 0.);

    if (theta_k < 0.)
        alpha -= 2. * pi;

    auto const m = static_cast<unsigned long int>(std::floor(alpha/theta_k));
    EXPECT_GE(m, 0ul);

    auto const Rn_alpha = qpp::gt.Rn(alpha, n).eval();
    auto const alpha_approx = std::fmod(m * k * theta, 2. * pi);
    auto const Rn_approx = qpp::gt.Rn(alpha_approx, n).eval();
    auto const error = qube::maths::operator_norm_2(Rn_alpha - Rn_approx);

    EXPECT_LT(error, epsilon / 3.);
}
