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

namespace
{
    //! @brief Characterization of density operator from theorem 2.5
    auto expect_density_operator(qpp_e::maths::Matrix auto const& rho, qpp_e::maths::RealNumber auto const& precision)
    {
        EXPECT_COMPLEX_CLOSE(rho.trace(), 1., precision);
        EXPECT_MATRIX_CLOSE(rho.adjoint(), rho, precision);
        EXPECT_GE(qpp::hevals(rho).minCoeff(), -precision);
    }
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

    expect_density_operator(rho, 1e-12);

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
    expect_density_operator(rho, 1e-12);

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

//! @brief Equations 2.148 through 2.152
TEST(chapter2_4, density_operator_measure_output)
{
    std::srand(0u);
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr MM = 7u;
    auto constexpr range = std::views::iota(0u, MM) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const rho = qpp::randrho(_2_pow_n);
    expect_density_operator(rho, 1e-12);
    auto const M = qpp::randkraus(MM, _2_pow_n);

    auto const [result, probabilities, resulting_state] = qpp::measure(rho, M);

    auto const rho_out = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& m)
    {
        return (M[m] * rho * M[m].adjoint()).eval();
    });
    EXPECT_MATRIX_NOT_CLOSE(rho_out, rho, 1e-1);
    expect_density_operator(rho_out, 1e-12);
}

//! @brief Theorem 2.5 and equations 2.153 through 2.157
TEST(chapter2_4, density_operator_characterization)
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

    expect_density_operator(rho, 1e-12);

    auto const phi = qpp::randket(_2_pow_n);
    auto const positivity = phi.dot(rho * phi);
    EXPECT_COMPLEX_CLOSE(positivity, positivity.real(), 1e-12);
    EXPECT_GE(positivity.real(), -1e-12);
}

//! @brief Exercise 2.71
TEST(chapter2_4, mixed_state_criterion)
{
    std::srand(0u);
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);

    auto const rho_mixed = qpp::randrho(_2_pow_n);
    expect_density_operator(rho_mixed, 1e-12);
    auto const trace2_mixed = (rho_mixed * rho_mixed).trace();
    EXPECT_COMPLEX_CLOSE(trace2_mixed, trace2_mixed.real(), 1e-12);
    EXPECT_LT(trace2_mixed.real(), 1.);

    auto const rho_pure = qpp::prj(qpp::randket(_2_pow_n));
    expect_density_operator(rho_pure, 1e-12);
    auto const trace2_pure = (rho_pure * rho_pure).trace();
    EXPECT_COMPLEX_CLOSE(trace2_pure, 1., 1e-12);
}
