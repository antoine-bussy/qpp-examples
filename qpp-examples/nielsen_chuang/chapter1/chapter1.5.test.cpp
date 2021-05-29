#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Figure 1.22
TEST(chapter1_5, stern_gerlach)
{
    using namespace qpp::literals;

    auto const oven = (0_ket + 1_ket).normalized().eval();

    auto const [result, probabilities, resulting_state] = qpp::measure(oven, qpp::gt.Z);
    EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities.data()), Eigen::Vector2d::Constant(0.5), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(oven) << '\n';
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state)
            std::cerr << qpp::disp(it) << "\n\n";
    }
}
