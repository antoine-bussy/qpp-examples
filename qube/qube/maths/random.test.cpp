#include <gtest/gtest.h>

#include "random.hpp"

TEST(random_test, seed)
{
    auto constexpr seed = 0u;

    qube::maths::seed(seed);
    auto const i0 = qpp::randidx();

    std::srand(seed);
    auto const i1 = qpp::randidx();
    EXPECT_NE(i1, i0);

    qube::maths::seed(seed);
    auto const i2 = qpp::randidx();
    EXPECT_EQ(i2, i0);
}
