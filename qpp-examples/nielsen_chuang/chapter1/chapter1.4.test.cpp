#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>

namespace
{
    auto constexpr print_text = true;
}

//! @brief Test of Q++ mket function
TEST(chapter1_4, mket)
{
    using namespace qpp::literals;
    
    EXPECT_MATRIX_EQ(qpp::mket({0, 0, 0}), 000_ket);
    EXPECT_MATRIX_EQ(qpp::mket({0, 0, 1}), 001_ket);
    EXPECT_MATRIX_EQ(qpp::mket({0, 1, 0}), 010_ket);
    EXPECT_MATRIX_EQ(qpp::mket({0, 1, 1}), 011_ket);

    EXPECT_MATRIX_EQ(qpp::mket({1, 0, 0}), 100_ket);
    EXPECT_MATRIX_EQ(qpp::mket({1, 0, 1}), 101_ket);
    EXPECT_MATRIX_EQ(qpp::mket({1, 1, 0}), 110_ket);
    EXPECT_MATRIX_EQ(qpp::mket({1, 1, 1}), 111_ket);
}

//! @brief Figure 1.14
TEST(chapter1_4, toffoli_gate)
{
    using namespace qpp::literals;
    
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 000_ket, 000_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 001_ket, 001_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 010_ket, 010_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 011_ket, 011_ket);

    EXPECT_MATRIX_EQ(qpp::gt.TOF * 100_ket, 100_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 101_ket, 101_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 110_ket, 111_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 111_ket, 110_ket);

    for(auto&& a : { 0u, 1u })
        for(auto&& b : { 0u, 1u })
            for(auto&& c : { 0u, 1u })
                EXPECT_MATRIX_EQ(qpp::gt.TOF * qpp::mket({a, b, c}), qpp::mket({a, b, (c + a * b) % 2}));
}
