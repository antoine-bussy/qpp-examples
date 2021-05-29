#include <gtest/gtest.h>

#include "arithmetic.hpp"

TEST(arithmetic_test, matrix_equal)
{
    EXPECT_EQ(qpp_e::maths::pow(2, 4), 16);
    EXPECT_EQ(qpp_e::maths::pow(-1, 4), 1);
    EXPECT_EQ(qpp_e::maths::pow(-1, 3), -1);
    EXPECT_EQ(qpp_e::maths::pow(3, 3), 27);
}
