#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ranges>

#include <qpp/qpp.hpp>
#include <qube/debug.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/gtest_macros.hpp>
#include <qube/maths/norm.hpp>
#include <qube/maths/random.hpp>
#include <qube/approximations.hpp>
#include <qube/decompositions.hpp>

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

//! @brief Equations 4.78 and 4.79
TEST(chapter4_5, H_T_phase_CNOT_universality_4)
{
    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto n = std::array{c, s, c};
    Eigen::Vector3d::Map(n.data()).normalize();
    auto m = std::array{c,-s, c};
    Eigen::Vector3d::Map(m.data()).normalize();

    auto const alpha = qpp::rand(0., 2. * std::numbers::pi);
    auto const& H = qpp::gt.H;

    auto const expected_Rm = qpp::gt.Rn(alpha, m).eval();
    auto const Rm = (H * qpp::gt.Rn(alpha, n) * H).eval();
    EXPECT_MATRIX_CLOSE(Rm, expected_Rm, 1e-12);

    debug() << ">> Rm:\n" << qpp::disp(Rm) << "\n\n";
    debug() << ">> expected_Rm:\n" << qpp::disp(expected_Rm) << "\n\n";
}

//! @brief Equations 4.80 and 4.81
//! @details Equation 4.80 is not verified in general.
//! @see Errata of N&C: https://michaelnielsen.org/qcqi/errata/errata/errata.html, for the correct formula
//! @details However, it holds on some necessary and sufficient condition: @see generalized_euler_decomposition_exists
TEST(chapter4_5, H_T_phase_CNOT_universality_5)
{
    using namespace std::complex_literals;

    qube::maths::seed(1754132672u);

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto n = std::array{c, s, c};
    auto n_v = Eigen::Vector3d::Map(n.data());
    n_v.normalize();
    auto m = std::array{c,-s, c};
    auto m_v = Eigen::Vector3d::Map(m.data());
    m_v.normalize();

    auto const U = qpp::randU();
    auto const [phase_U, theta_U, n_U] = qube::unitary_to_rotation(U);
    auto alpha = qube::generalized_euler_decomposition(phase_U, theta_U, n_U, n_v, m_v, n_v);

    auto const computed_U = (std::exp(1.i * alpha[0])
        * qpp::gt.Rn(alpha[1], n)
        * qpp::gt.Rn(alpha[2], m)
        * qpp::gt.Rn(alpha[3], n)).eval();
    EXPECT_MATRIX_CLOSE(computed_U, U, 1e-12);

    auto constexpr theta = 2. * std::acos(c*c);
    // Don't ask for too high a precision, otherwise the test will reach floating-point precision limits
    auto constexpr epsilon = 1e-4;

    auto constexpr beta = 4. * std::asin(epsilon / 6.);
    auto constexpr delta = beta;

    auto alpha_approx = Eigen::Vector3d::Zero().eval();
    auto mk = Eigen::Vector3<unsigned long int>::Zero().eval();
    auto theta_k = Eigen::Vector3d::Zero().eval();

    for (auto&& i: std::views::iota(0, 3))
    {
        debug() << "\n";
        alpha[i+1] = std::fmod(alpha[i+1], 2. * pi);
        if (alpha[i+1] < 0.)
        {
            alpha[i+1] += 2. * pi;
            // Adding 2pi to alpha[i+1] multiplies by the corresponding rotation matrix by -1.
            // This equivalent to multiplying the global phase by -1. To compensate, we add pi to alpha[0].
            alpha[0] += pi;
        }
        std::tie(alpha_approx[i], mk[i], theta_k[i]) = qube::angle_approximation(alpha[i+1], theta, delta);
        EXPECT_LT(std::abs(theta_k[i]), delta);
        EXPECT_NE(theta_k[i], 0.);
    }

    auto const& H = qpp::gt.H;

    for (auto&& i: std::views::iota(0, 3))
    {
        auto R_approx = qpp::gt.Rn(alpha_approx[i], n).eval();
        if (i == 1)
            R_approx = H * R_approx * H;
        auto const R = qpp::gt.Rn(alpha[i+1], (i == 1 ? m : n)).eval();
        auto const error = qube::maths::operator_norm_2(R - R_approx);
        EXPECT_LT(error, epsilon / 3.);

        debug() << ">> R_approx:\n" << qpp::disp(R_approx) << "\n\n";
        debug() << ">> R:\n" << qpp::disp(R) << "\n\n";
        debug() << ">> error: " << error << "\n";
    }

    auto const U_approx = (std::exp(1.i * alpha[0])
        * qpp::gt.Rn(alpha_approx[0], n)
        * H * qpp::gt.Rn(alpha_approx[1], n) * H
        * qpp::gt.Rn(alpha_approx[2], n)).eval();

    auto const error = qube::maths::operator_norm_2(U - U_approx);
    EXPECT_LT(error, epsilon);

    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
    debug() << ">> U_approx:\n" << qpp::disp(U_approx) << "\n\n";
    debug() << ">> error: " << error << "\n";
}
