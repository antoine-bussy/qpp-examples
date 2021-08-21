#include <gtest/gtest.h>

#include "concepts.hpp"

TEST(concepts_test, matrix)
{
    EXPECT_TRUE(qpp_e::maths::Matrix<Eigen::Vector2d>);
    EXPECT_FALSE(qpp_e::maths::Matrix<Eigen::Array2d>);
    EXPECT_FALSE(qpp_e::maths::Matrix<float>);
}

TEST(concepts_test, real_number)
{
    EXPECT_TRUE(qpp_e::maths::RealNumber<int>);
    EXPECT_TRUE(qpp_e::maths::RealNumber<float>);
    EXPECT_TRUE(qpp_e::maths::RealNumber<double>);
    EXPECT_FALSE(qpp_e::maths::RealNumber<std::complex<float>>);
    EXPECT_FALSE(qpp_e::maths::RealNumber<char*>);
}

TEST(concepts_test, complex_number)
{
    EXPECT_TRUE(qpp_e::maths::ComplexNumber<int>);
    EXPECT_TRUE(qpp_e::maths::ComplexNumber<float>);
    EXPECT_TRUE(qpp_e::maths::ComplexNumber<double>);
    EXPECT_TRUE(qpp_e::maths::ComplexNumber<std::complex<float>>);
    EXPECT_FALSE(qpp_e::maths::ComplexNumber<char*>);
}

TEST(concepts_test, matrix_vector_block)
{
    using vector_t = Eigen::Vector4f;
    auto const A = vector_t{};
    using vector_block_t = std::decay_t<decltype(A.head<2>())>;

    EXPECT_TRUE(qpp_e::maths::Matrix<vector_block_t>);
}
