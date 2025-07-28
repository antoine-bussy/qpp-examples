#include <gtest/gtest.h>

#include "concepts.hpp"

TEST(concepts_test, matrix)
{
    EXPECT_TRUE(qube::maths::Matrix<Eigen::Vector2d>);
    EXPECT_FALSE(qube::maths::Matrix<Eigen::Array2d>);
    EXPECT_FALSE(qube::maths::Matrix<float>);
}

TEST(concepts_test, real_number)
{
    EXPECT_TRUE(qube::maths::RealNumber<int>);
    EXPECT_TRUE(qube::maths::RealNumber<float>);
    EXPECT_TRUE(qube::maths::RealNumber<double>);
    EXPECT_FALSE(qube::maths::RealNumber<std::complex<float>>);
    EXPECT_FALSE(qube::maths::RealNumber<char*>);
}

TEST(concepts_test, complex_number)
{
    EXPECT_TRUE(qube::maths::ComplexNumber<int>);
    EXPECT_TRUE(qube::maths::ComplexNumber<float>);
    EXPECT_TRUE(qube::maths::ComplexNumber<double>);
    EXPECT_TRUE(qube::maths::ComplexNumber<std::complex<float>>);
    EXPECT_FALSE(qube::maths::ComplexNumber<char*>);
}

TEST(concepts_test, matrix_vector_block)
{
    using vector_t = Eigen::Vector4f;
    auto const A = vector_t{};
    using vector_block_t = std::decay_t<decltype(A.head<2>())>;

    EXPECT_TRUE(qube::maths::Matrix<vector_block_t>);
}
