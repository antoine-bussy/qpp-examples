#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>

#include <numbers>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Equations 1.8 through 1.12
TEST(chapter1_3, not_gate)
{
    auto const state = qpp::randket(2).eval();
    auto const not_state = (qpp::gt.X * state).eval();

    EXPECT_MATRIX_EQ(not_state, state.reverse());
    EXPECT_MATRIX_EQ(qpp::gt.X, Eigen::Matrix2cd::Identity().rowwise().reverse());

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> NOT Gate:\n" << qpp::disp(qpp::gt.X) << '\n';
        std::cerr << ">> NOT State:\n" << qpp::disp(not_state) << '\n';
    }
}

//! @brief Equation 1.13
TEST(chapter1_3, z_gate)
{
    auto const state = qpp::randket(2).eval();
    auto const z_state = (qpp::gt.Z * state).eval();

    EXPECT_EQ(z_state[0], state[0]);
    EXPECT_EQ(z_state[1],-state[1]);
    EXPECT_MATRIX_EQ(qpp::gt.Z, Eigen::Vector2cd(1, -1).asDiagonal().toDenseMatrix());

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> Z Gate:\n" << qpp::disp(qpp::gt.Z) << '\n';
        std::cerr << ">> Z State:\n" << qpp::disp(z_state) << '\n';
    }
}

//! @brief Equation 1.14
TEST(chapter1_3, hadamard_gate)
{
    auto const state = qpp::randket(2).eval();
    auto const h_state = (qpp::gt.H * state).eval();

    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    EXPECT_COMPLEX_CLOSE(h_state[0], (state[0] + state[1]) * inv_sqrt2, 1e-12);
    EXPECT_COMPLEX_CLOSE(h_state[1], (state[0] - state[1]) * inv_sqrt2, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> H Gate:\n" << qpp::disp(qpp::gt.H) << '\n';
        std::cerr << ">> H State:\n" << qpp::disp(h_state) << '\n';
    }
}

//! @brief Equations 1.15 and 1.16, and Block 1.1
TEST(chapter1_3, general_single_qubit_gate)
{
    auto constexpr alpha = 0.57;
    auto constexpr beta  = -1.3;
    auto constexpr gamma = -0.86;
    auto constexpr delta = 1.02;

    auto const phase_shift = std::polar(1., alpha);
    auto const rot_z_beta = Eigen::Vector2cd(std::polar(1., -beta), std::polar(1., beta)).asDiagonal().toDenseMatrix();
    auto const rot_gamma = Eigen::Rotation2Dd(0.5 * gamma);
    auto const rot_z_delta = Eigen::Vector2cd(std::polar(1., -delta), std::polar(1., delta)).asDiagonal().toDenseMatrix();

    auto const U = (phase_shift * rot_z_beta * rot_gamma.cast<std::complex<double>>() * rot_z_delta).eval();
    EXPECT_MATRIX_CLOSE(U * U.adjoint(), Eigen::Matrix2cd::Identity(), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> Phase shift:\n" << phase_shift << '\n';
        std::cerr << ">> Rotation-Z Beta:\n" << rot_z_beta << '\n';
        std::cerr << ">> Rotation Gamma:\n" << rot_gamma.toRotationMatrix() << '\n';
        std::cerr << ">> Rotation-Z Delta:\n" << rot_z_delta << '\n';
        std::cerr << ">> General single qubit gate:\n" << U << '\n';
    }
}

//! @brief Equation 1.18 and Figure 1.6
TEST(chapter1_3, cnot_gate)
{
    using namespace qpp::literals;

    auto cnot_matrix = Eigen::Matrix4cd::Zero().eval();
    cnot_matrix(0, 0) = 1.;
    cnot_matrix(1, 1) = 1.;
    cnot_matrix(2, 3) = 1.;
    cnot_matrix(3, 2) = 1.;

    EXPECT_MATRIX_EQ(qpp::gt.CNOT, cnot_matrix);

    EXPECT_MATRIX_EQ(qpp::gt.CNOT * 00_ket, 00_ket);
    EXPECT_MATRIX_EQ(qpp::gt.CNOT * 01_ket, 01_ket);
    EXPECT_MATRIX_EQ(qpp::gt.CNOT * 10_ket, 11_ket);
    EXPECT_MATRIX_EQ(qpp::gt.CNOT * 11_ket, 10_ket);

    EXPECT_MATRIX_EQ(qpp::gt.CNOT * qpp::gt.CNOT.adjoint(), Eigen::Matrix4cd::Identity());

    if constexpr (print_text)
        std::cerr << ">> CNOT Gate:\n" << qpp::disp(qpp::gt.CNOT) << '\n';
}
