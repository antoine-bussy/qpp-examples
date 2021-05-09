#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ranges>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Equations 1.1 and 1.2
TEST(chapter1, superposition)
{
    using namespace qpp::literals;
    auto const state = (0_ket + 1_ket).normalized().eval();
    EXPECT_EQ(state, Eigen::Vector2cd(1, 1).normalized());
}

//! @brief Measure of equation 1.2
TEST(chapter1, measure)
{
    using namespace qpp::literals;
    auto const state = (0_ket + 1_ket).normalized().eval();
    auto const [result, probabilities, resulting_state] = qpp::measure(state, qpp::gt.Id2);

    EXPECT_THAT((std::array{ 0, 1 }), testing::Contains(result));
    EXPECT_THAT(probabilities, testing::ElementsAreArray({ testing::DoubleEq(0.5), testing::DoubleEq(0.5) }));
    EXPECT_THAT(resulting_state, testing::ElementsAreArray({ 0_ket, 1_ket }));

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state)
            std::cerr << qpp::disp(it) << "\n\n";
    }
}

//! @brief Equation 1.5
TEST(chapter1, two_qubits)
{
    using namespace qpp::literals;
    auto const state = (00_ket + 01_ket + 10_ket + 11_ket).normalized().eval();
    EXPECT_MATRIX_EQ(state, Eigen::Vector4cd(1, 1, 1, 1).normalized());

    auto const state00 = 00_ket;
    EXPECT_MATRIX_EQ(state00, Eigen::Vector4cd(1, 0, 0, 0));

    auto const state01 = 01_ket;
    EXPECT_MATRIX_EQ(state01, Eigen::Vector4cd(0, 1, 0, 0));

    auto const state10 = 10_ket;
    EXPECT_MATRIX_EQ(state10, Eigen::Vector4cd(0, 0, 1, 0));

    auto const state11 = 11_ket;
    EXPECT_MATRIX_EQ(state11, Eigen::Vector4cd(0, 0, 0, 1));
}

//! @brief Measure of equation 1.5
TEST(chapter1, two_qubits_measure)
{
    using namespace qpp::literals;
    auto const state = qpp::randket(4).normalized().eval();

    auto const [result, probabilities, resulting_state] = qpp::measure(state, Eigen::Matrix4cd::Identity());

    EXPECT_THAT((std::array{ 0, 1, 2, 3 }), testing::Contains(result));
    EXPECT_MATRIX_CLOSE(Eigen::Vector4d::Map(probabilities.data()), state.cwiseAbs2(), 1e-12);
    for (auto&& i : std::views::iota(0, 4))
        EXPECT_MATRIX_EQ(resulting_state[i], (state[i] * Eigen::Vector4cd::Unit(i)).normalized());

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state)
            std::cerr << qpp::disp(it) << "\n\n";
    }
}

//! @brief Equation 1.6
TEST(chapter1, simple_two_qubits_measure_on_first_qubit)
{
    using namespace qpp::literals;
    auto const state = (00_ket + 01_ket).normalized().eval();

    auto const [result, probabilities, resulting_state] = qpp::measure(state, qpp::gt.Id2, { 0 });

    EXPECT_EQ(result, 0);
    EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities.data()), Eigen::Vector2d::UnitX(), 1e-12);
    EXPECT_MATRIX_EQ(resulting_state[0], Eigen::Vector2cd(1, 1).normalized());
    EXPECT_MATRIX_EQ(resulting_state[1], Eigen::Vector2cd::Zero());

    auto const resulting_state_0_4 = Eigen::Vector4cd(resulting_state[0](0, 0), resulting_state[0](1, 0), 0, 0).eval();
    EXPECT_MATRIX_CLOSE(resulting_state_0_4, Eigen::Vector4cd(state[0], state[1], 0, 0).normalized(), 1e-12);

    auto const resulting_state_1_4 = Eigen::Vector4cd(0, 0, resulting_state[1](0, 0), resulting_state[1](1, 0)).eval();
    EXPECT_MATRIX_CLOSE(resulting_state_1_4, Eigen::Vector4cd(0, 0, state[2], state[3]).normalized(), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state)
            std::cerr << qpp::disp(it) << "\n\n";
    }
}

//! @brief Equation 1.6
TEST(chapter1, simple_two_qubits_measure_on_first_qubit_2)
{
    using namespace qpp::literals;
    auto const state = (00_ket + 01_ket).normalized().eval();

    auto const [result, probabilities, resulting_state] = qpp::measure(state, qpp::gt.Id2, { 1 });

    EXPECT_THAT((std::array{ 0, 1 }), testing::Contains(result));
    EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities.data()), Eigen::Vector2d::Constant(0.5), 1e-12);
    EXPECT_MATRIX_EQ(resulting_state[0], Eigen::Vector2cd::UnitX());
    EXPECT_MATRIX_EQ(resulting_state[1], Eigen::Vector2cd::UnitX());

    auto const resulting_state_0_4 = Eigen::Vector4cd(resulting_state[0](0, 0), 0, resulting_state[0](1, 0), 0).eval();
    EXPECT_MATRIX_CLOSE(resulting_state_0_4, Eigen::Vector4cd(state[0], 0, state[2], 0).normalized(), 1e-12);

    auto const resulting_state_1_4 = Eigen::Vector4cd(0, resulting_state[1](0, 0), resulting_state[1](1, 0), 0).eval();
    EXPECT_MATRIX_CLOSE(resulting_state_1_4, Eigen::Vector4cd(0, state[1], 0, state[3]).normalized(), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state)
            std::cerr << qpp::disp(it) << "\n\n";
    }
}

//! @brief Equation 1.6
TEST(chapter1, two_qubits_measure_on_first_qubit)
{
    using namespace qpp::literals;
    auto const state = qpp::randket(4).normalized().eval();

    auto const [result, probabilities, resulting_state] = qpp::measure(state, qpp::gt.Id2, { 0 });

    EXPECT_THAT((std::array{ 0, 1 }), testing::Contains(result));
    EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities.data()), Eigen::Vector2d(state.head<2>().squaredNorm(), state.tail<2>().squaredNorm()), 1e-12);

    EXPECT_MATRIX_CLOSE(resulting_state[0], state.head<2>().normalized(), 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state[1], state.tail<2>().normalized(), 1e-12);

    auto const resulting_state_0_4 = Eigen::Vector4cd(resulting_state[0](0, 0), resulting_state[0](1, 0), 0, 0).eval();
    EXPECT_MATRIX_CLOSE(resulting_state_0_4, Eigen::Vector4cd(state[0], state[1], 0, 0).normalized(), 1e-12);

    auto const resulting_state_1_4 = Eigen::Vector4cd(0, 0, resulting_state[1](0, 0), resulting_state[1](1, 0)).eval();
    EXPECT_MATRIX_CLOSE(resulting_state_1_4, Eigen::Vector4cd(0, 0, state[2], state[3]).normalized(), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state)
            std::cerr << qpp::disp(it) << "\n\n";
    }
}

//! @brief Equation 1.7
TEST(chapter1, bell_state)
{
    using namespace qpp::literals;
    auto const bell_state = (00_ket + 11_ket).normalized().eval();
    EXPECT_MATRIX_EQ(bell_state, Eigen::Vector4cd(1, 0, 0, 1).normalized());

    if constexpr (print_text)
        std::cerr << ">> State:\n" << qpp::disp(bell_state) << '\n';
}
