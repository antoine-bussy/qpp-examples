#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ranges>

#include <qpp/qpp.hpp>
#include <qube/debug.hpp>
#include <qube/introspection.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/random.hpp>
#include <qube/maths/gtest_macros.hpp>

using namespace qube::stream;

//! @brief Equations 4.58 and 4.59, figure 4.16
TEST(chapter4_5, single_qubit_and_cnot_universality)
{
    using namespace qpp::literals;

    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr n = qube::maths::pow(2ul, nq);

    auto const U_tilde = qpp::randU(2);
    auto U_expected = Eigen::MatrixXcd::Identity(n, n).eval();
    U_expected({ 0ul, n-1 }, { 0ul, n-1 }) = U_tilde;

    auto const& X = qpp::gt.X;
    auto const circuit = qpp::QCircuit{ nq }
        .CTRL(X, {0,1}, 2, std::vector{1ul,1ul})
        .CTRL(X, {0,2}, 1, std::vector{1ul,0ul})
        .CTRL(U_tilde, {1,2}, 0)
        .CTRL(X, {0,2}, 1, std::vector{1ul,0ul})
        .CTRL(X, {0,1}, 2, std::vector{1ul,1ul})
        ;
    auto const U = qube::extract_matrix<8>(circuit);

    EXPECT_MATRIX_CLOSE(U, U_expected, 1e-12);
    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
    debug() << ">> U_expected:\n" << qpp::disp(U_expected) << "\n\n";
}

//! @brief Exercise 4.39
TEST(chapter4_5, single_qubit_and_cnot_universality_2)
{
    using namespace qpp::literals;

    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr n = qube::maths::pow(2ul, nq);

    auto const U_tilde = qpp::randU(2);
    auto U_expected = Eigen::MatrixXcd::Identity(n, n).eval();
    U_expected({ 2ul, n-1 }, { 2ul, n-1 }) = U_tilde;

    auto const& X = qpp::gt.X;
    auto const circuit = qpp::QCircuit{ nq }
        .CTRL(X, {0,1}, 2, std::vector{1ul,0ul})
        .CTRL(U_tilde, {1,2}, 0)
        .CTRL(X, {0,1}, 2, std::vector{1ul,0ul})
        ;
    auto const U = qube::extract_matrix<8>(circuit);

    EXPECT_MATRIX_CLOSE(U, U_expected, 1e-12);
    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
    debug() << ">> U_expected:\n" << qpp::disp(U_expected) << "\n\n";
}
