#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/arithmetic.hpp>

#include <execution>
#include <numbers>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Equations 2.92 through 2.95
TEST(chapter2_2, measurement_operators)
{
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const state = qpp::randket(_2_pow_n).eval();
    auto const measurement_operators = qpp::randkraus(_2_pow_n, _2_pow_n);

    auto const completeness = std::transform_reduce(policy, measurement_operators.cbegin(), measurement_operators.cend()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& Mm)
    {
        return (Mm.adjoint() * Mm).eval();
    });
    EXPECT_MATRIX_CLOSE(completeness, Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);

    auto const [result, probabilities, resulting_state] = qpp::measure(state, measurement_operators);

    for(auto&& m : range)
    {
        auto const& Mm = measurement_operators[m];
        auto const pm = state.dot(Mm.adjoint() * Mm * state);
        EXPECT_NEAR(pm.real(), probabilities[m], 1e-12);
        EXPECT_NEAR(pm.imag(), 0., 1e-12);
        auto const post_measurement_state_m = (Mm * state / std::sqrt(pm.real())).eval();
        EXPECT_MATRIX_CLOSE(post_measurement_state_m, resulting_state[m], 1e-12);
    }

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> Measurement operators:\n";
        for (auto&& op : measurement_operators)
            std::cerr << qpp::disp(op) << "\n\n";
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& st : resulting_state)
            std::cerr << qpp::disp(st) << "\n\n";
    }
}

//! @brief Equations 2.96 through 2.98
TEST(chapter2_2, measurement_operators_one_qubit)
{
    using namespace qpp::literals;

    auto const state = qpp::randket().eval();

    auto const M0 = 0_prj;
    EXPECT_MATRIX_EQ(M0, 0_ket * (0_ket).adjoint());
    EXPECT_MATRIX_EQ(M0 * M0, M0);
    EXPECT_MATRIX_EQ(M0.adjoint(), M0);

    auto const M1 = 1_prj;
    EXPECT_MATRIX_EQ(M1, 1_ket * (1_ket).adjoint());
    EXPECT_MATRIX_EQ(M1 * M1, M1);
    EXPECT_MATRIX_EQ(M1.adjoint(), M1);

    auto const completeness = (M0.adjoint() * M0 + M1.adjoint() * M1).eval();
    EXPECT_MATRIX_EQ(completeness, Eigen::Matrix2cd::Identity());

    auto const [result, probabilities, resulting_state] = qpp::measure(state, { M0, M1 });

    auto const p0 = state.dot(M0.adjoint() * M0 * state);
    EXPECT_NEAR(p0.real(), probabilities[0], 1e-12);
    EXPECT_NEAR(p0.real(), std::norm(state[0]), 1e-12);
    EXPECT_NEAR(p0.imag(), 0., 1e-12);
    auto const post_measurement_state_0 = (M0 * state / std::abs(state[0])).eval();
    EXPECT_MATRIX_CLOSE(post_measurement_state_0, resulting_state[0], 1e-12);
    EXPECT_MATRIX_CLOSE(post_measurement_state_0, state[0] / std::abs(state[0]) * 0_ket, 1e-12);

    auto const p1 = state.dot(M1.adjoint() * M1 * state);
    EXPECT_NEAR(p1.real(), probabilities[1], 1e-12);
    EXPECT_NEAR(p1.real(), std::norm(state[1]), 1e-12);
    EXPECT_NEAR(p1.imag(), 0., 1e-12);
    auto const post_measurement_state_1 = (M1 * state / std::abs(state[1])).eval();
    EXPECT_MATRIX_CLOSE(post_measurement_state_1, resulting_state[1], 1e-12);
    EXPECT_MATRIX_CLOSE(post_measurement_state_1, state[1] / std::abs(state[1]) * 1_ket, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> State:\n" << qpp::disp(state) << '\n';
        std::cerr << ">> M0:\n" << qpp::disp(M0) << "\n\n";
        std::cerr << ">> M1:\n" << qpp::disp(M1) << "\n\n";
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& st : resulting_state)
            std::cerr << qpp::disp(st) << "\n\n";
    }
}

//! @brief Exercise 2.57
TEST(chapter2_2, cascade_measurement_operators)
{
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const L = qpp::randkraus(_2_pow_n, _2_pow_n);
    auto const M = qpp::randkraus(_2_pow_n, _2_pow_n);

    auto const state = qpp::randket(_2_pow_n).eval();

    auto const [result_L, probabilities_L, resulting_state_L] = qpp::measure(state, L);

    auto ML = std::vector<qpp::cmat>(L.size() * M.size());
    auto probabilities_ML = std::vector<double>(L.size() * M.size());
    auto resulting_state_ML = std::vector<qpp::ket>(L.size() * M.size());

    std::for_each(policy, range.begin(), range.end(), [&](auto&& l)
    {
        auto const [result_M, probabilities_M, resulting_state_M] = qpp::measure(resulting_state_L[l], M);
        for (auto&& m : range)
        {
            auto const ml = l * _2_pow_n + m;
            ML[ml] = M[m] * L[l];
            probabilities_ML[ml] = probabilities_M[m] * probabilities_L[l];
            resulting_state_ML[ml] = resulting_state_M[m];
        }
    });

    auto const [result_ML_single, probabilities_ML_single, resulting_state_ML_single] = qpp::measure(state, ML);

    for (auto&& ml : std::views::iota(0u, ML.size()))
    {
        EXPECT_NEAR(probabilities_ML_single[ml], probabilities_ML[ml], 1e-12);
        EXPECT_MATRIX_CLOSE(resulting_state_ML_single[ml], resulting_state_ML_single[ml], 1e-12);
    }

    auto const completeness = std::transform_reduce(policy, ML.cbegin(), ML.cend()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& Nlm)
    {
        return (Nlm.adjoint() * Nlm).eval();
    });
    EXPECT_MATRIX_CLOSE(completeness, Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);
}

//! @brief Box 2.3 and equations 2.99 through 2.101
//! @details Instead of reproducing the proof and its equations as is, we show by example that measurements operators
//! on a non orthogonal basis, e.g. (|0>, |+>), don't satisfy the completeness relation.
TEST(chapter2_2, non_completeness)
{
   using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    EXPECT_MATRIX_EQ(qpp::st.pz0, 0_prj);
    EXPECT_MATRIX_EQ(qpp::st.pz1, 1_prj);
    EXPECT_MATRIX_CLOSE(qpp::st.px0, qpp::prj(0_ket + 1_ket), 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::st.px0, qpp::prj(inv_sqrt2 * (0_ket + 1_ket)), 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::st.px1, qpp::prj(0_ket - 1_ket), 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::st.px1, qpp::prj(inv_sqrt2 * (0_ket - 1_ket)), 1e-12);

    auto constexpr completeness = [](auto&& M0, auto&& M1) { return (M0.adjoint() * M0 + M1.adjoint() * M1).eval(); };

    EXPECT_TRUE(completeness(qpp::st.pz0, qpp::st.pz1).isIdentity(1e-12));
    EXPECT_TRUE(completeness(qpp::st.px0, qpp::st.px1).isIdentity(1e-12));
    EXPECT_FALSE(completeness(qpp::st.px0, qpp::st.pz0).isIdentity(1e-1));
    EXPECT_FALSE(completeness(qpp::st.px0, qpp::st.pz1).isIdentity(1e-1));
    EXPECT_FALSE(completeness(qpp::st.px1, qpp::st.pz0).isIdentity(1e-1));
    EXPECT_FALSE(completeness(qpp::st.px1, qpp::st.pz1).isIdentity(1e-1));

    if constexpr (print_text)
    {
        std::cerr << ">> PZ0:\n" << qpp::disp(qpp::st.pz0) << "\n\n";
        std::cerr << ">> PZ1:\n" << qpp::disp(qpp::st.pz1) << "\n\n";
        std::cerr << ">> PX0:\n" << qpp::disp(qpp::st.px0) << "\n\n";
        std::cerr << ">> PX1:\n" << qpp::disp(qpp::st.px1) << "\n\n";
        std::cerr << ">> Completeness Z:\n" << qpp::disp(completeness(qpp::st.pz0, qpp::st.pz1)) << "\n\n";
        std::cerr << ">> Completeness X:\n" << qpp::disp(completeness(qpp::st.px0, qpp::st.px1)) << "\n\n";
        std::cerr << ">> Completeness X0Z0:\n" << qpp::disp(completeness(qpp::st.px0, qpp::st.pz0)) << "\n\n";
        std::cerr << ">> Completeness X0Z1:\n" << qpp::disp(completeness(qpp::st.px0, qpp::st.pz1)) << "\n\n";
        std::cerr << ">> Completeness X1Z0:\n" << qpp::disp(completeness(qpp::st.px1, qpp::st.pz0)) << "\n\n";
        std::cerr << ">> Completeness X1Z1:\n" << qpp::disp(completeness(qpp::st.px1, qpp::st.pz1)) << "\n\n";
    }
}

//! @brief Measurements operators are not necessarily projective measurements
TEST(chapter2_2, not_projective_measurements)
{
    std::srand(0);
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto const measurement_operators = qpp::randkraus(_2_pow_n, _2_pow_n);

    for (auto&& m : std::views::iota(0u, _2_pow_n))
    {
        auto const& Mm = measurement_operators[m];
        EXPECT_FALSE(Mm.adjoint().isApprox(Mm, 1e-1));
        EXPECT_FALSE(Mm.isApprox(Mm * Mm, 1e-1));
        for (auto&& l : std::views::iota(m + 1, _2_pow_n))
        {
            auto const& Ml = measurement_operators[l];
            EXPECT_FALSE((Ml * Mm).isZero(1e-2));
            EXPECT_FALSE((Mm * Ml).isZero(1e-2));
        }
    }
}

//! @brief Equations 2.102 through 2.115
TEST(chapter2_2, projective_measurements)
{
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const M = qpp::randH(_2_pow_n);
    EXPECT_MATRIX_EQ(M.adjoint(), M);
    auto const [lambda, X] = qpp::heig(M);
    EXPECT_TRUE((X.adjoint() * X).isIdentity(1e-12));

    auto P = std::vector<qpp::cmat>(X.cols());
    for (auto&& m : range)
        P[m] = X.col(m) * X.col(m).adjoint();

    auto const M_check = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& m)
    {
        return (lambda[m] * P[m]).eval();
    });
    EXPECT_MATRIX_CLOSE(M_check, M, 1e-12);

    auto const state = qpp::randket(_2_pow_n);
    auto const [result, probabilities, resulting_state] = qpp::measure(state, P);
    for (auto&& m : range)
    {
        auto const& Pm = P[m];
        auto const pm = state.dot(Pm * state);
        EXPECT_NEAR(pm.real(), probabilities[m], 1e-12);
        EXPECT_NEAR(pm.imag(), 0., 1e-12);
        auto const post_measurement_state_m = (Pm * state / std::sqrt(pm.real())).eval();
        EXPECT_MATRIX_CLOSE(post_measurement_state_m, resulting_state[m], 1e-12);
    }

    auto const mean_M = std::transform_reduce(policy, range.begin(), range.end()
        , 0.
        , std::plus<>{}
        , [&](auto&& m)
    {
        return lambda[m] * probabilities[m];
    });
    auto const mean_Mb = state.dot(M * state);
    EXPECT_NEAR(mean_M, mean_Mb.real(), 1e-12);
    EXPECT_NEAR(mean_Mb.imag(), 0., 1e-12);

    auto const variance_M = std::transform_reduce(policy, range.begin(), range.end()
        , 0.
        , std::plus<>{}
        , [&](auto&& m)
    {
        auto const dx = lambda[m] - mean_M;
        return probabilities[m] * dx * dx ;
    });
    auto const variance_Mb = state.dot(M * M * state) - mean_M * mean_M;
    EXPECT_NEAR(variance_M, variance_Mb.real(), 1e-12);
    EXPECT_NEAR(variance_Mb.imag(), 0., 1e-12);
}
