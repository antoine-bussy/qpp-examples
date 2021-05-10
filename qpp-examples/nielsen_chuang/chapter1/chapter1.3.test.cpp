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
    using namespace qpp::literals;
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
    using namespace qpp::literals;
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
    using namespace qpp::literals;
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
