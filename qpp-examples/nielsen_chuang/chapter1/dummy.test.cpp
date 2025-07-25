#include <gtest/gtest.h>

#include <qpp/qpp.hpp>

TEST(dummy_test, add)
{
    EXPECT_EQ(3 + 4, 7);
}

TEST(dummy_test, qpp)
{
    using namespace qpp::literals;
    EXPECT_EQ(0_ket, Eigen::Vector2cd(1, 0));
}
