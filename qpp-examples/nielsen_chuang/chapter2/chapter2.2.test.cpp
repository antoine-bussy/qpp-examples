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
