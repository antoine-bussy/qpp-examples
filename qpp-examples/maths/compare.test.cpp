#include <gtest/gtest.h>

#include "compare.hpp"

#include <complex>

TEST(compare_test, matrix_equal)
{
    EXPECT_TRUE(qpp_e::maths::matrix_equal(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 1.}));
    EXPECT_TRUE(qpp_e::maths::matrix_equal_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 1.}));
    EXPECT_FALSE(qpp_e::maths::matrix_equal(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 7.}));
    EXPECT_FALSE(qpp_e::maths::matrix_equal_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 7.}));
}

TEST(compare_test, matrix_close)
{
    EXPECT_TRUE(qpp_e::maths::matrix_close(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 1.01}, 0.01));
    EXPECT_TRUE(qpp_e::maths::matrix_close_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 1.01}, 0.01));
    EXPECT_FALSE(qpp_e::maths::matrix_close(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 1.1}, 0.01));
    EXPECT_FALSE(qpp_e::maths::matrix_close_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 1.1}, 0.01));
}

TEST(compare_test, complex_close)
{
    using namespace std::complex_literals;
    EXPECT_TRUE(qpp_e::maths::complex_close(1. + 1i, 1. + 1.01i, 0.001));
    EXPECT_TRUE(qpp_e::maths::complex_close_l(1. + 1i, 1. + 1.01i, 0.001));
    EXPECT_FALSE(qpp_e::maths::complex_close(1. + 1i, 1. + 1.1i, 0.001));
    EXPECT_FALSE(qpp_e::maths::complex_close_l(1. + 1i, 1. + 1.1i, 0.001));
}

TEST(compare_test, collinear)
{
    EXPECT_TRUE(qpp_e::maths::collinear(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 2.01 }, 0.01));
    EXPECT_TRUE(qpp_e::maths::collinear_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 2.01 }, 0.01));
    EXPECT_FALSE(qpp_e::maths::collinear(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 2.1 }, 1e-6));
    EXPECT_FALSE(qpp_e::maths::collinear_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 2.1 }, 1e-6));
    EXPECT_FALSE(qpp_e::maths::collinear(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 5.1 }, 0.01));
    EXPECT_FALSE(qpp_e::maths::collinear_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 5.1 }, 0.01));
}

TEST(compare_test, phase_collinear)
{
    EXPECT_TRUE(qpp_e::maths::collinear(Eigen::Vector2d::Ones(), Eigen::Vector2d{ -1., -1.01 }, 0.01, true));
    EXPECT_TRUE(qpp_e::maths::phase_collinear_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ -1., -1.01 }, 0.01));
    EXPECT_FALSE(qpp_e::maths::collinear(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 2.01 }, 0.01, true));
    EXPECT_FALSE(qpp_e::maths::phase_collinear_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 2.01 }, 0.01));
    EXPECT_FALSE(qpp_e::maths::collinear(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 2.1 }, 0.01, true));
    EXPECT_FALSE(qpp_e::maths::phase_collinear_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 2.1 }, 0.01));
    EXPECT_FALSE(qpp_e::maths::collinear(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 5.1 }, 0.01, true));
    EXPECT_FALSE(qpp_e::maths::phase_collinear_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 2., 5.1 }, 0.01));
}

TEST(compare_test, matrix_close_up_to_factor)
{
    using namespace std::literals::complex_literals;

    auto const M = Eigen::Matrix2cd{{1., 3.i}, {7.i, 5.i}};
    auto const N = Eigen::Matrix2cd{{6.i, 9.i}, {7., 8.i}};
    auto const lambda = std::exp(5.14i);

    EXPECT_FALSE(qpp_e::maths::matrix_close_up_to_factor(M, N, 0.01));
    EXPECT_FALSE(qpp_e::maths::matrix_close_up_to_factor_l(M, N, 0.01));

    EXPECT_TRUE(qpp_e::maths::matrix_close_up_to_factor(M, (5. * M + Eigen::Matrix2cd::Constant(0.01)).eval(), 0.01));
    EXPECT_TRUE(qpp_e::maths::matrix_close_up_to_factor_l(M, (5. * M + Eigen::Matrix2cd::Constant(0.01)).eval(), 0.01));

    EXPECT_TRUE(qpp_e::maths::matrix_close_up_to_factor(M, (lambda * M + Eigen::Matrix2cd::Constant(0.01)).eval(), 0.01));
    EXPECT_TRUE(qpp_e::maths::matrix_close_up_to_factor_l(M, (lambda * M + Eigen::Matrix2cd::Constant(0.01)).eval(), 0.01));
}

TEST(compare_test, matrix_close_up_to_phase_factor)
{
    using namespace std::literals::complex_literals;

    auto const M = Eigen::Matrix2cd{{1., 3.i}, {7.i, 5.i}};
    auto const N = Eigen::Matrix2cd{{6.i, 9.i}, {7., 8.i}};
    auto const lambda = std::exp(5.14i);

    EXPECT_FALSE(qpp_e::maths::matrix_close_up_to_factor(M, N, 0.01, true));
    EXPECT_FALSE(qpp_e::maths::matrix_close_up_to_phase_factor_l(M, N, 0.01));

    EXPECT_FALSE(qpp_e::maths::matrix_close_up_to_factor(M, (5. * M + Eigen::Matrix2cd::Constant(0.01)).eval(), 0.01, true));
    EXPECT_FALSE(qpp_e::maths::matrix_close_up_to_phase_factor_l(M, (5. * M + Eigen::Matrix2cd::Constant(0.01)).eval(), 0.01));

    EXPECT_TRUE(qpp_e::maths::matrix_close_up_to_factor(M, (lambda * M + Eigen::Matrix2cd::Constant(0.01)).eval(), 0.01, true));
    EXPECT_TRUE(qpp_e::maths::matrix_close_up_to_phase_factor_l(M, (lambda * M + Eigen::Matrix2cd::Constant(0.01)).eval(), 0.01));
}

