#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/arithmetic.hpp>
#include <execution>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Equations 2.92 through 2.95
TEST(chapter2_2, measurement_operators)
{
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const state = qpp::randket(_2_pow_n).eval();
    auto const measurement_operators = qpp::randkraus(_2_pow_n, _2_pow_n);

    auto const completeness = std::transform_reduce(policy, measurement_operators.cbegin(), measurement_operators.cend()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& Mm)
    {
        return (Mm.adjoint() * Mm).eval();
    });
    EXPECT_MATRIX_CLOSE(completeness, Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);

    auto const [result, probabilities, resulting_state] = qpp::measure(state, measurement_operators);

    for(auto&& m : range)
    {
        auto const& Mm = measurement_operators[m];
        auto const pm = state.dot(Mm.adjoint() * Mm * state);
        EXPECT_NEAR(pm.real(), probabilities[m], 1e-12);
        EXPECT_NEAR(pm.imag(), 0., 1e-12);
        auto const post_measurement_state_m = (Mm * state / std::sqrt(pm.real())).eval();
        EXPECT_MATRIX_CLOSE(post_measurement_state_m, resulting_state[m], 1e-12);
    }

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> Measurement operators:\n";
        for (auto&& op : measurement_operators)
            std::cerr << qpp::disp(op) << "\n\n";
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& st : resulting_state)
            std::cerr << qpp::disp(st) << "\n\n";
    }
}

//! @brief Equations 2.96 through 2.98
TEST(chapter2_2, measurement_operators_one_qubit)
{
    using namespace qpp::literals;

    auto const state = qpp::randket().eval();

    auto const M0 = 0_prj;
    EXPECT_MATRIX_EQ(M0, 0_ket * (0_ket).adjoint());
    EXPECT_MATRIX_EQ(M0 * M0, M0);
    EXPECT_MATRIX_EQ(M0.adjoint(), M0);

    auto const M1 = 1_prj;
    EXPECT_MATRIX_EQ(M1, 1_ket * (1_ket).adjoint());
    EXPECT_MATRIX_EQ(M1 * M1, M1);
    EXPECT_MATRIX_EQ(M1.adjoint(), M1);

    auto const completeness = (M0.adjoint() * M0 + M1.adjoint() * M1).eval();
    EXPECT_MATRIX_EQ(completeness, Eigen::Matrix2cd::Identity());

    auto const [result, probabilities, resulting_state] = qpp::measure(state, { M0, M1 });

    auto const p0 = state.dot(M0.adjoint() * M0 * state);
    EXPECT_NEAR(p0.real(), probabilities[0], 1e-12);
    EXPECT_NEAR(p0.real(), std::norm(state[0]), 1e-12);
    EXPECT_NEAR(p0.imag(), 0., 1e-12);
    auto const post_measurement_state_0 = (M0 * state / std::abs(state[0])).eval();
    EXPECT_MATRIX_CLOSE(post_measurement_state_0, resulting_state[0], 1e-12);
    EXPECT_MATRIX_CLOSE(post_measurement_state_0, state[0] / std::abs(state[0]) * 0_ket, 1e-12);

    auto const p1 = state.dot(M1.adjoint() * M1 * state);
    EXPECT_NEAR(p1.real(), probabilities[1], 1e-12);
    EXPECT_NEAR(p1.real(), std::norm(state[1]), 1e-12);
    EXPECT_NEAR(p1.imag(), 0., 1e-12);
    auto const post_measurement_state_1 = (M1 * state / std::abs(state[1])).eval();
    EXPECT_MATRIX_CLOSE(post_measurement_state_1, resulting_state[1], 1e-12);
    EXPECT_MATRIX_CLOSE(post_measurement_state_1, state[1] / std::abs(state[1]) * 1_ket, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> M0:\n" << qpp::disp(M0) << "\n\n";
        std::cerr << ">> M1:\n" << qpp::disp(M1) << "\n\n";
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& st : resulting_state)
            std::cerr << qpp::disp(st) << "\n\n";
    }
}

//! @brief Exercise 2.57
TEST(chapter2_2, cascade_measurement_operators)
{
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const L = qpp::randkraus(_2_pow_n, _2_pow_n);
    auto const M = qpp::randkraus(_2_pow_n, _2_pow_n);

    auto const state = qpp::randket(_2_pow_n).eval();

    auto const [result_L, probabilities_L, resulting_state_L] = qpp::measure(state, L);

    auto ML = std::vector<qpp::cmat>(L.size() * M.size());
    auto probabilities_ML = std::vector<double>(L.size() * M.size());
    auto resulting_state_ML = std::vector<qpp::ket>(L.size() * M.size());

    std::for_each(policy, range.begin(), range.end(), [&](auto&& l)
    {
        auto const [result_M, probabilities_M, resulting_state_M] = qpp::measure(resulting_state_L[l], M);
        for (auto&& m : range)
        {
            auto const ml = l * _2_pow_n + m;
            ML[ml] = M[m] * L[l];
            probabilities_ML[ml] = probabilities_M[m] * probabilities_L[l];
            resulting_state_ML[ml] = resulting_state_M[m];
        }
    });

    auto const [result_ML_single, probabilities_ML_single, resulting_state_ML_single] = qpp::measure(state, ML);

    for (auto&& ml : std::views::iota(0u, ML.size()))
    {
        EXPECT_NEAR(probabilities_ML_single[ml], probabilities_ML[ml], 1e-12);
        EXPECT_MATRIX_CLOSE(resulting_state_ML_single[ml], resulting_state_ML_single[ml], 1e-12);
    }

    auto const completeness = std::transform_reduce(policy, ML.cbegin(), ML.cend()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& Nlm)
    {
        return (Nlm.adjoint() * Nlm).eval();
    });
    EXPECT_MATRIX_CLOSE(completeness, Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);
}
