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
    EXPECT_MATRIX_CLOSE(qpp::gt.RZ(beta), Eigen::Vector2cd(std::polar(1., -0.5 * beta), std::polar(1., 0.5 * beta)).asDiagonal().toDenseMatrix(), 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::gt.RY(gamma), Eigen::Rotation2Dd(0.5 * gamma).toRotationMatrix().cast<std::complex<double>>(), 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::gt.RZ(delta), Eigen::Vector2cd(std::polar(1., -0.5 * delta), std::polar(1., 0.5 * delta)).asDiagonal().toDenseMatrix(), 1e-12);

    // Note: it looks like Euler angles
    auto const U = (phase_shift * qpp::gt.RZ(beta) * qpp::gt.RY(gamma) * qpp::gt.RZ(delta)).eval();
    EXPECT_MATRIX_CLOSE(U * U.adjoint(), Eigen::Matrix2cd::Identity(), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> Phase shift:\n" << phase_shift << '\n';
        std::cerr << ">> Rotation-Z Beta:\n" << qpp::disp(qpp::gt.RZ(beta)) << '\n';
        std::cerr << ">> Rotation Gamma:\n" << qpp::disp(qpp::gt.RY(gamma)) << '\n';
        std::cerr << ">> Rotation-Z Delta:\n" << qpp::disp(qpp::gt.RZ(delta)) << '\n';
        std::cerr << ">> General single qubit gate:\n" << qpp::disp(U) << '\n';
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

//! @brief Equation 1.19
TEST(chapter1_3, plus_minus_states)
{
    using namespace qpp::literals;

    auto const plus_ket = qpp::st.plus();
    auto const minus_ket = qpp::st.minus();

    EXPECT_MATRIX_CLOSE(plus_ket, (0_ket + 1_ket).normalized(), 1e-12);
    EXPECT_MATRIX_CLOSE(minus_ket, (0_ket - 1_ket).normalized(), 1e-12);

    auto const state = qpp::randket().eval();
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;
    EXPECT_MATRIX_CLOSE(state, ((state[0] + state[1]) * plus_ket + (state[0] - state[1]) * minus_ket) * inv_sqrt2, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> |+> State:\n" << qpp::disp(plus_ket) << '\n';
        std::cerr << ">> |-> State:\n" << qpp::disp(minus_ket) << '\n';
    }
}

//! @brief Equation 1.19
TEST(chapter1_3, plus_minus_states_measure)
{
    using namespace qpp::literals;
    auto const plus_minus_basis = (Eigen::Matrix2cd() << qpp::st.plus(), qpp::st.minus()).finished();
    EXPECT_MATRIX_EQ(plus_minus_basis, qpp::gt.H);

    auto const state = qpp::randket().eval();
    auto const [result, probabilities, resulting_state] = qpp::measure(state, qpp::gt.H);

    EXPECT_THAT((std::array{ 0, 1 }), testing::Contains(result));

    auto constexpr collinear2D = [](auto&& actual, auto&& expected, auto&& precision)
    {
        using complex_t = std::complex<double>;
        auto const det = (Eigen::Matrix2<complex_t>() << actual.template cast<complex_t>(), expected.template cast<complex_t>()).finished().determinant();
        return std::norm(det) < precision * precision;
    };
    EXPECT_PRED3(collinear2D, resulting_state[0], qpp::st.plus(), 1e-12);
    EXPECT_PRED3(collinear2D, resulting_state[1], qpp::st.minus(), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state)
            std::cerr << qpp::disp(it) << "\n\n";
    }
}
