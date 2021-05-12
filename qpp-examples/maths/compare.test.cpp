#include <gtest/gtest.h>

#include "compare.hpp"

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
    EXPECT_TRUE(qpp_e::maths::matrix_close_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 1.}, 0.01));
    EXPECT_FALSE(qpp_e::maths::matrix_close(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 1.1}, 0.01));
    EXPECT_FALSE(qpp_e::maths::matrix_close_l(Eigen::Vector2d::Ones(), Eigen::Vector2d{ 1., 1.1}, 0.01));
}

TEST(compare_test, complex_close)
{
    using namespace std::complex_literals;
    EXPECT_TRUE(qpp_e::maths::complex_close(1. + 1i, 1. + 1.01i, 0.01));
    EXPECT_TRUE(qpp_e::maths::complex_close_l(1. + 1i, 1. + 1.01i, 0.01));
    EXPECT_FALSE(qpp_e::maths::complex_close(1. + 1i, 1. + 1.1i, 0.01));
    EXPECT_FALSE(qpp_e::maths::complex_close_l(1. + 1i, 1. + 1.1i, 0.01));
}

