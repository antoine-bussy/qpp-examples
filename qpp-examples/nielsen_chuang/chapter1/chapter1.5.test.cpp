#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>

#include <numbers>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Figure 1.22 and equations 1.56 through 1.59
TEST(chapter1_5, stern_gerlach)
{
    using namespace qpp::literals;

    auto const oven = (0_ket + 1_ket).normalized().eval();

    auto const [result1, probabilities1, resulting_state1] = qpp::measure(oven, qpp::gt.Z);

    EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities1.data()), Eigen::Vector2d::Constant(0.5), 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state1[0], 0_ket, 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state1[1], 1_ket, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> Oven:\n" << qpp::disp(oven) << '\n';
        std::cerr << ">> Measurement result: " << result1 << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities1, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state1)
            std::cerr << qpp::disp(it) << "\n\n";
    }
}

//! @brief Figure 1.23
TEST(chapter1_5, cascaded_stern_gerlach)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    auto const oven = (0_ket + 1_ket).normalized().eval();

    auto const [result1, probabilities1, resulting_state1] = qpp::measure(oven, qpp::gt.Z);
    auto const [result2, probabilities2, resulting_state2] = qpp::measure(resulting_state1[0], qpp::gt.H);

    EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities2.data()), Eigen::Vector2d::Constant(0.5), 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state2[0], (0_ket + 1_ket) * inv_sqrt2, 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state2[1], (0_ket - 1_ket) * inv_sqrt2, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> Oven:\n" << qpp::disp(oven) << '\n';
        std::cerr << ">> |+Z>:\n" << qpp::disp(resulting_state1[0]) << '\n';
        std::cerr << ">> Measurement result: " << result2 << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities2, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state2)
            std::cerr << qpp::disp(it) << "\n\n";
    }
}
