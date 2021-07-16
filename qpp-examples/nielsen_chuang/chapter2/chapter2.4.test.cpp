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

//! @brief Equations 2.138 and 2.139
TEST(chapter2_4, density_operator_transformation)
{
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr I = 7u;
    auto constexpr range = std::views::iota(0u, I) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const p = Eigen::VectorXd::Map(qpp::randprob(I).data(), I).eval();
    EXPECT_NEAR(p.sum(), 1., 1e-12);

    auto const rho = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        auto const psi_i = qpp::randket(_2_pow_n);
        return (p[i] * qpp::prj(psi_i)).eval();
    });

    auto target = qpp::qram(n);
    std::iota(target.begin(), target.end(), 0u);

    auto const U = qpp::randU(_2_pow_n);
    auto const rho_out = qpp::apply(rho, U, target);
    EXPECT_MATRIX_CLOSE(rho_out, U * rho * U.adjoint(), 1e-12);
}

//! @brief Equation 2.139
TEST(chapter2_4, density_operator_transformation_2)
{
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);

    auto const rho = qpp::randrho(_2_pow_n);

    auto target = qpp::qram(n);
    std::iota(target.begin(), target.end(), 0u);

    auto const U = qpp::randU(_2_pow_n);
    auto const rho_out = qpp::apply(rho, U, target);
    EXPECT_MATRIX_CLOSE(rho_out, U * rho * U.adjoint(), 1e-12);
}

//! @brief Equations 2.140 through 2.147
TEST(chapter2_4, density_operator_measure)
{
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr MM = 7u;

    auto const rho = qpp::randrho(_2_pow_n);
    auto const M = qpp::randkraus(MM, _2_pow_n);

    auto const [result, probabilities, resulting_state] = qpp::measure(rho, M);

    for (auto&& m : std::views::iota(0u, MM))
    {
        auto const pm = (M[m].adjoint() * M[m] * rho).trace();
        EXPECT_COMPLEX_CLOSE(pm, probabilities[m], 1e-12);

        auto const rho_m = ((M[m] * rho * M[m].adjoint()) / pm).eval();
        EXPECT_MATRIX_CLOSE(rho_m, resulting_state[m], 1e-12);
    }
}
