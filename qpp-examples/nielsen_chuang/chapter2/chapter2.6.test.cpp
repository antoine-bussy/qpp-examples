#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>

#include <numbers>

namespace
{
    auto constexpr print_text = false;
}

namespace
{
    //! @brief Compute mean and variance
    auto statistics(qpp_e::maths::Matrix auto const& M, qpp_e::maths::Matrix auto const& state)
    {
        auto const mean = state.dot(M * state);
        EXPECT_NEAR(mean.imag(), 0., 1e-12);
        auto const variance = state.dot(M * M * state) - mean * mean;
        EXPECT_GE(variance.real(), -1e-12);
        EXPECT_NEAR(variance.imag(), 0., 1e-12);
        return std::tuple{ mean.real(), variance.real(), std::sqrt(std::max(variance.real(), 0.)) };
    }
}

//! @brief Box 2.7 and equations 2.213 through 2.217
TEST(chapter2_6, anti_correlations)
{
    qpp_e::maths::seed();

    auto constexpr sqrt2 = std::numbers::sqrt2;
    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto const state = qpp::st.b11;

    auto const v = Eigen::Vector3d::Random().normalized().eval();
    auto const v_dot_sigma = (v[0] * qpp::gt.X + v[1] * qpp::gt.Y + v[2] * qpp::gt.Z).eval();
    auto const v_dot_sigma_hvects = qpp::hevects(v_dot_sigma);

    auto const [result, probabilities, resulting_state] = qpp::measure(state, v_dot_sigma_hvects, { 0 });
    EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities.data()), Eigen::Vector2d::Constant(0.5), 1e-12);


    if constexpr (print_text)
    {
        std::cerr << ">> state:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> v . sigma:\n" << qpp::disp(v_dot_sigma) << "\n\n";

        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& st : resulting_state)
            std::cerr << qpp::disp(st) << "\n\n";
    }

    for (auto&& i : { 0u, 1u })
    {
        auto const j = 1u - i;
        auto const [result2, probabilities2, resulting_state2] = qpp::measure(resulting_state[i], v_dot_sigma_hvects);

        EXPECT_EQ(result2, j);
        EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities2.data()), Eigen::Vector2d::Unit(j), 1e-12);
        EXPECT_MATRIX_CLOSE(resulting_state2[j], resulting_state[i], 1e-12);

        if constexpr (print_text)
        {
            std::cerr << ">> Measurement result: " << result2 << '\n';
            std::cerr << ">> Probabilities: ";
            std::cerr << qpp::disp(probabilities2, ", ") << '\n';
            std::cerr << ">> Resulting states:\n";
            for (auto&& stt : resulting_state2)
                std::cerr << qpp::disp(stt) << "\n\n";
        }
    }

    EXPECT_MATRIX_CLOSE(v_dot_sigma_hvects * v_dot_sigma_hvects.adjoint(), Eigen::Matrix2cd::Identity(), 1e-12);
    auto const v_dot_sigma_hvects_inv = v_dot_sigma_hvects.adjoint();

    auto const a = v_dot_sigma_hvects.col(0);
    auto const b = v_dot_sigma_hvects.col(1);
    auto const det = v_dot_sigma_hvects_inv.determinant();

    auto const ab_state = (inv_sqrt2 * (qpp::kron(a, b) - qpp::kron(b, a))).eval();
    EXPECT_MATRIX_CLOSE(det * ab_state, state, 1e-12);

    EXPECT_NEAR(std::norm(det), 1., 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> state    : " << qpp::disp(state.transpose()) << '\n';
        std::cerr << ">> ab_state : " << qpp::disp(ab_state.transpose()) << '\n';
        std::cerr << ">> alt_state: " << qpp::disp(det * ab_state.transpose()) << '\n';
        std::cerr << ">> det: " << det << ", norm: " << std::norm(det) << "\n";
    }
}

//! @brief Equations 2.226 through 2.230
TEST(chapter2_6, bell_inequality)
{
    auto constexpr sqrt2 = std::numbers::sqrt2;
    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto const state = qpp::st.b11;

    auto const Q = qpp::gt.Z;
    auto const R = qpp::gt.X;
    auto const S = (-inv_sqrt2 * (qpp::gt.Z + qpp::gt.X)).eval();
    auto const T = ( inv_sqrt2 * (qpp::gt.Z - qpp::gt.X)).eval();

    auto const [mean_QS, variance_QS, std_QS] = statistics(qpp::kron(Q,S), state);
    auto const [mean_RT, variance_RT, std_RT] = statistics(qpp::kron(R,T), state);
    auto const [mean_RS, variance_RS, std_RS] = statistics(qpp::kron(R,S), state);
    auto const [mean_QT, variance_QT, std_QT] = statistics(qpp::kron(Q,T), state);

    EXPECT_NEAR(mean_QS, inv_sqrt2, 1e-12);
    EXPECT_NEAR(mean_RS, inv_sqrt2, 1e-12);
    EXPECT_NEAR(mean_RT, inv_sqrt2, 1e-12);
    EXPECT_NEAR(mean_QT,-inv_sqrt2, 1e-12);

    EXPECT_NEAR(mean_QS + mean_RS + mean_RT - mean_QT, 2 * sqrt2, 1e-12);
}
