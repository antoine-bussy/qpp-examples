#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.hpp>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>
#include <qpp-examples/qube/debug.hpp>

#include <execution>
#include <numbers>
#include <ranges>

using namespace qpp_e::qube::stream;

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
        EXPECT_COMPLEX_CLOSE(pm, probabilities[m], 1e-12);
        auto const post_measurement_state_m = (Mm * state / std::sqrt(probabilities[m])).eval();
        EXPECT_MATRIX_CLOSE(post_measurement_state_m, resulting_state[m], 1e-12);
    }

    debug() << ">> State:\n" << qpp::disp(state) << '\n';
    debug() << ">> Measurement operators:\n";
    for (auto&& op : measurement_operators)
        debug() << qpp::disp(op) << "\n\n";
    debug() << ">> Measurement result: " << result << '\n';
    debug() << ">> Probabilities: ";
    debug() << qpp::disp(probabilities, {", "}) << '\n';
    debug() << ">> Resulting states:\n";
    for (auto&& st : resulting_state)
        debug() << qpp::disp(st) << "\n\n";
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
    EXPECT_COMPLEX_CLOSE(p0, probabilities[0], 1e-12);
    EXPECT_COMPLEX_CLOSE(p0, std::norm(state[0]), 1e-12);
    auto const post_measurement_state_0 = (M0 * state / std::abs(state[0])).eval();
    EXPECT_MATRIX_CLOSE(post_measurement_state_0, resulting_state[0], 1e-12);
    EXPECT_MATRIX_CLOSE(post_measurement_state_0, state[0] / std::abs(state[0]) * 0_ket, 1e-12);

    auto const p1 = state.dot(M1.adjoint() * M1 * state);
    EXPECT_COMPLEX_CLOSE(p1, probabilities[1], 1e-12);
    EXPECT_COMPLEX_CLOSE(p1, std::norm(state[1]), 1e-12);
    auto const post_measurement_state_1 = (M1 * state / std::abs(state[1])).eval();
    EXPECT_MATRIX_CLOSE(post_measurement_state_1, resulting_state[1], 1e-12);
    EXPECT_MATRIX_CLOSE(post_measurement_state_1, state[1] / std::abs(state[1]) * 1_ket, 1e-12);

    debug() << ">> State:\n" << qpp::disp(state) << '\n';
    debug() << ">> M0:\n" << qpp::disp(M0) << "\n\n";
    debug() << ">> M1:\n" << qpp::disp(M1) << "\n\n";
    debug() << ">> Measurement result: " << result << '\n';
    debug() << ">> Probabilities: ";
    debug() << qpp::disp(probabilities, {", "}) << '\n';
    debug() << ">> Resulting states:\n";
    for (auto&& st : resulting_state)
        debug() << qpp::disp(st) << "\n\n";
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

    debug() << ">> PZ0:\n" << qpp::disp(qpp::st.pz0) << "\n\n";
    debug() << ">> PZ1:\n" << qpp::disp(qpp::st.pz1) << "\n\n";
    debug() << ">> PX0:\n" << qpp::disp(qpp::st.px0) << "\n\n";
    debug() << ">> PX1:\n" << qpp::disp(qpp::st.px1) << "\n\n";
    debug() << ">> Completeness Z:\n" << qpp::disp(completeness(qpp::st.pz0, qpp::st.pz1)) << "\n\n";
    debug() << ">> Completeness X:\n" << qpp::disp(completeness(qpp::st.px0, qpp::st.px1)) << "\n\n";
    debug() << ">> Completeness X0Z0:\n" << qpp::disp(completeness(qpp::st.px0, qpp::st.pz0)) << "\n\n";
    debug() << ">> Completeness X0Z1:\n" << qpp::disp(completeness(qpp::st.px0, qpp::st.pz1)) << "\n\n";
    debug() << ">> Completeness X1Z0:\n" << qpp::disp(completeness(qpp::st.px1, qpp::st.pz0)) << "\n\n";
    debug() << ">> Completeness X1Z1:\n" << qpp::disp(completeness(qpp::st.px1, qpp::st.pz1)) << "\n\n";
}

//! @brief Measurements operators are not necessarily projective measurements
TEST(chapter2_2, not_projective_measurements)
{
    qpp_e::maths::seed(0);
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

//! @brief Equations 2.102 through 2.104 and 2.110 through 2.115
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
        EXPECT_COMPLEX_CLOSE(pm, probabilities[m], 1e-12);
        auto const post_measurement_state_m = (Pm * state / std::sqrt(probabilities[m])).eval();
        EXPECT_MATRIX_CLOSE(post_measurement_state_m, resulting_state[m], 1e-12);
    }

    auto const mean_M = std::transform_reduce(policy, range.begin(), range.end()
        , 0.
        , std::plus<>{}
        , [&](auto&& m)
    {
        return lambda[m] * probabilities[m];
    });
    auto const variance_M = std::transform_reduce(policy, range.begin(), range.end()
        , 0.
        , std::plus<>{}
        , [&](auto&& m)
    {
        auto const dx = lambda[m] - mean_M;
        return probabilities[m] * dx * dx ;
    });
    auto const [mean_Mb, variance_Mb, std_Mb] = statistics(M, state);
    EXPECT_COMPLEX_CLOSE(mean_M, mean_Mb, 1e-12);
    EXPECT_COMPLEX_CLOSE(variance_M, variance_Mb, 1e-12);
}

//! @brief Exercise 2.58
TEST(chapter2_2, projective_measurements_with_eigenstate)
{
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;

    auto const M = qpp::randH(_2_pow_n);
    auto const [lambda, X] = qpp::heig(M);

    for (auto&& m : range)
    {
        auto const state = X.col(m);
        auto const [mean_M, variance_M, std_M] = statistics(M, state);
        EXPECT_COMPLEX_CLOSE(mean_M, lambda[m], 1e-12);
        EXPECT_NEAR(std::abs(variance_M), 0., 1e-12);
    }
}

//! @brief Box 2.4 and equations 2.105 through 2.108
TEST(chapter2_2, heisenberg_uncertainty_principle)
{
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);

    auto const state = qpp::randket(_2_pow_n);
    auto const C = qpp::randH(_2_pow_n);
    auto const D = qpp::randH(_2_pow_n);

    auto const [mean_C, variance_C, std_C] = statistics(C, state);
    auto const [mean_D, variance_D, std_D] = statistics(D, state);

    EXPECT_GE(std_C * std_D, 0.5 * std::abs(state.dot(qpp::comm(C, D) * state)));
}

//! @brief Box 2.4 and equation 2.109
TEST(chapter2_2, heisenberg_uncertainty_principle_pauli)
{
    using namespace std::complex_literals;
    using namespace qpp::literals;

    EXPECT_MATRIX_CLOSE(qpp::comm(qpp::gt.X, qpp::gt.Y), 2i * qpp::gt.Z, 1e-12);

    auto const [mean_X, variance_X, std_X] = statistics(qpp::gt.X, 0_ket);
    auto const [mean_Y, variance_Y, std_Y] = statistics(qpp::gt.Y, 0_ket);

    EXPECT_COMPLEX_CLOSE((0_ket).dot(qpp::gt.Z * 0_ket), 1., 1e-12);
    EXPECT_GE(std_X * std_Y, 1.);
    EXPECT_GE(std_X, 0.);
    EXPECT_GE(std_Y, 0.);
}

//! @brief Equation 2.116
TEST(chapter2_2, spin_axis_measurement)
{
    auto const v = Eigen::Vector3d::Random().normalized().eval();
    auto const v_dot_sigma = (v[0] * qpp::gt.X + v[1] * qpp::gt.Y + v[2] * qpp::gt.Z).eval();
    EXPECT_MATRIX_EQ(v_dot_sigma.adjoint(), v_dot_sigma);
}

//! @brief Exercise 2.59
TEST(chapter2_2, observable_X)
{
    using namespace qpp::literals;

    auto const [mean, variance, standard_deviation] = statistics(qpp::gt.X, 0_ket);
    EXPECT_NEAR(mean, 0., 1e-12);
    EXPECT_NEAR(standard_deviation, 1., 1e-12);
}

//! @brief Exercise 2.60
TEST(chapter2_2, spin_axis_measurement_eigen)
{
    auto const v = Eigen::Vector3d::Random().normalized().eval();
    auto const v_dot_sigma = (v[0] * qpp::gt.X + v[1] * qpp::gt.Y + v[2] * qpp::gt.Z).eval();
    auto [eigen_values, eigen_vectors] = qpp::heig(v_dot_sigma);
    EXPECT_NEAR(eigen_values[0], -1., 1e-12);
    EXPECT_NEAR(eigen_values[1],  1., 1e-12);

    auto const P_minus = (eigen_vectors.col(0) * eigen_vectors.col(0).adjoint());
    auto const P_plus  = (eigen_vectors.col(1) * eigen_vectors.col(1).adjoint());
    EXPECT_MATRIX_CLOSE(P_minus, 0.5 * (qpp::gt.Id2 - v_dot_sigma), 1e-12);
    EXPECT_MATRIX_CLOSE(P_plus,  0.5 * (qpp::gt.Id2 + v_dot_sigma), 1e-12);
}

//! @brief Exercise 2.61
TEST(chapter2_2, spin_axis_measure)
{
    using namespace std::complex_literals;
    using namespace qpp::literals;

    auto const v = Eigen::Vector3d::Random().normalized().eval();
    auto const v_dot_sigma = (v[0] * qpp::gt.X + v[1] * qpp::gt.Y + v[2] * qpp::gt.Z).eval();

    auto const [result, probabilities, resulting_state] = qpp::measure(0_ket, qpp::hevects(v_dot_sigma));

    EXPECT_NEAR(probabilities[1], 0.5 * (1. + v[2]), 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state[1], std::sqrt(0.5 * (1. + v[2])) * 0_ket + (v[0] + 1i * v[1]) / std::sqrt(2. * (1. + v[2])) * 1_ket, 1e-12);
}

//! @brief Exercise 2.62
TEST(chapter2_2, povm_and_projective)
{
    using namespace qpp::literals;
    auto constexpr sqrt2 = std::numbers::sqrt2;

    auto const E1 = (sqrt2 / (1. + sqrt2) * 1_ket * (1_ket).adjoint()).eval();
    auto const E2 = (0.5 * sqrt2 / (1. + sqrt2) * (0_ket - 1_ket) * (0_ket - 1_ket).adjoint()).eval();
    auto const E3 = (Eigen::Matrix2cd::Identity() - E1 - E2).eval();

    auto const Ks = std::vector{ qpp::sqrtm(E1), qpp::sqrtm(E2), qpp::sqrtm(E3) };

    /* POVM is different from measurement... */
    EXPECT_MATRIX_NOT_CLOSE(E1, Ks[0], 1e-2);
    EXPECT_MATRIX_NOT_CLOSE(E2, Ks[1], 1e-2);
    EXPECT_MATRIX_NOT_CLOSE(E3, Ks[2], 1e-2);

    /* ... therefore not projective */
    for (auto&& i : std::views::iota(0, 3))
        for (auto&& j : std::views::iota(0, 3))
        {
            if (i == j)
                EXPECT_MATRIX_NOT_CLOSE(Ks[i] * Ks[i], Ks[i], 1e-2);
            else
                EXPECT_MATRIX_NOT_CLOSE(Ks[i] * Ks[j], Eigen::Matrix2cd::Zero(), 1e-2);

        }
}

//! @brief Equations 2.117 through 2.120
TEST(chapter2_2, povm)
{
    using namespace qpp::literals;
    auto constexpr sqrt2 = std::numbers::sqrt2;
    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto const psi1 = 0_ket;
    auto const psi2 = (inv_sqrt2 * (0_ket + 1_ket)).eval();
    auto const psi = std::vector{ psi1, psi2 };

    auto const E1 = (sqrt2 / (1. + sqrt2) * 1_ket * (1_ket).adjoint()).eval();
    auto const E2 = (0.5 * sqrt2 / (1. + sqrt2) * (0_ket - 1_ket) * (0_ket - 1_ket).adjoint()).eval();
    auto const E3 = (Eigen::Matrix2cd::Identity() - E1 - E2).eval();

    auto const Ks = std::vector{ qpp::sqrtm(E1), qpp::sqrtm(E2), qpp::sqrtm(E3) };

    auto const [result1, probabilities1, resulting_state1] = qpp::measure(psi1, Ks);
    EXPECT_THAT((std::array{ 1, 2 }), testing::Contains(result1));
    EXPECT_EQ(probabilities1[0], 0.);
    EXPECT_LT(probabilities1[1], 1.);
    EXPECT_LT(probabilities1[2], 1.);

    auto const [result2, probabilities2, resulting_state2] = qpp::measure(psi2, Ks);
    EXPECT_THAT((std::array{ 0, 2 }), testing::Contains(result2));
    EXPECT_LT(probabilities2[0], 1.);
    EXPECT_EQ(probabilities2[1], 0.);
    EXPECT_LT(probabilities2[2], 1.);

    auto rd = std::random_device{};
    auto gen = std::mt19937{ rd() };
    auto distrib = std::uniform_int_distribution{ 0, 1 };

    for (auto&& n : std::views::iota(0, 50))
    {
        static_cast<void>(n);
        auto const i = distrib(gen);
        auto const [result, probabilities, resulting_state] = qpp::measure(psi[i], Ks);
        switch (result)
        {
        case 0:
            EXPECT_MATRIX_EQ(psi[i], psi2);
            break;
        case 1:
            EXPECT_MATRIX_EQ(psi[i], psi1);
            break;
        case 2:
            break;
        default:
            ASSERT_FALSE(false);
            break;
        }
    }
}

//! @brief Exercise 2.63
//! @details Same as previous, but using random unitary matrices to extract measurement from POVM
TEST(chapter2_2, povm_and_measurement)
{
    using namespace qpp::literals;
    auto constexpr sqrt2 = std::numbers::sqrt2;
    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto const psi1 = 0_ket;
    auto const psi2 = (inv_sqrt2 * (0_ket + 1_ket)).eval();
    auto const psi = std::vector{ psi1, psi2 };

    auto const E1 = (sqrt2 / (1. + sqrt2) * 1_ket * (1_ket).adjoint()).eval();
    auto const E2 = (0.5 * sqrt2 / (1. + sqrt2) * (0_ket - 1_ket) * (0_ket - 1_ket).adjoint()).eval();
    auto const E3 = (Eigen::Matrix2cd::Identity() - E1 - E2).eval();

    auto const Ks = std::vector<qpp::cmat>{ qpp::randU() * qpp::sqrtm(E1), qpp::randU() * qpp::sqrtm(E2), qpp::randU() * qpp::sqrtm(E3) };

    auto const [result1, probabilities1, resulting_state1] = qpp::measure(psi1, Ks);
    EXPECT_THAT((std::array{ 1, 2 }), testing::Contains(result1));
    EXPECT_EQ(probabilities1[0], 0.);
    EXPECT_LT(probabilities1[1], 1.);
    EXPECT_LT(probabilities1[2], 1.);

    auto const [result2, probabilities2, resulting_state2] = qpp::measure(psi2, Ks);
    EXPECT_THAT((std::array{ 0, 2 }), testing::Contains(result2));
    EXPECT_LT(probabilities2[0], 1.);
    EXPECT_EQ(probabilities2[1], 0.);
    EXPECT_LT(probabilities2[2], 1.);

    auto rd = std::random_device{};
    auto gen = std::mt19937{ rd() };
    auto distrib = std::uniform_int_distribution{ 0, 1 };

    for (auto&& n : std::views::iota(0, 50))
    {
        static_cast<void>(n);
        auto const i = distrib(gen);
        auto const [result, probabilities, resulting_state] = qpp::measure(psi[i], Ks);
        switch (result)
        {
        case 0:
            EXPECT_MATRIX_EQ(psi[i], psi2);
            break;
        case 1:
            EXPECT_MATRIX_EQ(psi[i], psi1);
            break;
        case 2:
            break;
        default:
            ASSERT_FALSE(false);
            break;
        }
    }
}

//! @brief Exercise 2.64
TEST(chapter2_2, povm_construction)
{
    qpp_e::maths::seed(0u);
    auto constexpr n = 5u;
    auto constexpr m = 3u;
    auto constexpr range = std::views::iota(0u, m) | std::views::common;

    auto psi = Eigen::MatrixXcd::Zero(n, m).eval();

    auto constexpr rank = [](auto const& M)
    {
        return M.jacobiSvd().setThreshold(1e-1).rank();
    };

    while (rank(psi) < m)
        psi.setRandom();

    for (auto&& p : psi.colwise())
        p.normalize();

    auto E = std::vector<Eigen::MatrixXcd>{ m + 1, Eigen::MatrixXcd::Zero(n, n) };

    auto const povm = [&](auto const& i)
    {
        auto A = psi.eval();
        A.col(i).swap(A.rightCols<1>());
        auto const x = qpp::grams(A).rightCols<1>().eval();
        return (x * x.adjoint()).eval();
    };

    for (auto&& i : range)
    {
        E[i] = povm(i);
        E[m] -= E[i];
    }

    auto const min_eigen_value = qpp::hevals(E[m]).minCoeff();

    if (min_eigen_value < 0.)
        for (auto& e : E)
            e /= -1.01 * min_eigen_value;
    E[m] += Eigen::MatrixXcd::Identity(n, n);

    auto Ks = std::vector<Eigen::MatrixXcd>{};
    Ks.reserve(m + 1);
    for (auto& e : E)
        Ks.emplace_back(qpp::sqrtm(e));

    for (auto&& i : range)
    {
        auto const [result, probabilities, resulting_state] = qpp::measure(psi.col(i), Ks);
        EXPECT_THAT((std::array{ i, m }), testing::Contains(result));
        EXPECT_LT(probabilities[i], 1.);
        debug() << "Probability for psi[" << i << "]: " << probabilities[i] << '\n';
        EXPECT_GT(probabilities[i], 0.);
        for (auto&& j : range)
            if (i != j)
            {
                EXPECT_NEAR(probabilities[j], 0., 1e-12);
            }
        EXPECT_LT(probabilities[m], 1.);
        EXPECT_GT(probabilities[m], 0.);
    }

    auto rd = std::random_device{};
    auto gen = std::mt19937{ rd() };
    auto distrib = std::uniform_int_distribution{ 0u, m - 1u };

    for (auto&& n : std::views::iota(0, 50))
    {
        static_cast<void>(n);
        auto const i = distrib(gen);
        auto const [result, probabilities, resulting_state] = qpp::measure(psi.col(i), Ks);
        if (result != m)
        {
            EXPECT_EQ(result, i);
        }
    }
}

//! @brief Equation 2.121 and exercise 2.65
TEST(chapter2_2, relative_phase)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    auto const psi1 = (inv_sqrt2 * (0_ket + 1_ket)).eval();
    auto const psi2 = (inv_sqrt2 * (0_ket - 1_ket)).eval();

    for (auto&& i : {0, 1})
        EXPECT_NEAR(std::norm(psi1[i]), std::norm(psi2[i]), 1e-12);

    auto const psi1b = (qpp::gt.H * psi1).eval();
    auto const psi2b = (qpp::gt.H * psi2).eval();

    for (auto&& i : {0, 1})
        EXPECT_THAT(std::norm(psi1b[i]), testing::Not(testing::DoubleNear(std::norm(psi2b[i]), 1e-1)));
}

//! @brief Exercise 2.66
TEST(chapter2_2, composite_system)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    auto const state = (inv_sqrt2 * (00_ket + 11_ket)).eval();
    auto const observable = qpp::kron(qpp::gt.X, qpp::gt.Z);

    auto const [mean, variance, STD] = statistics(observable, state);
    EXPECT_NEAR(mean, 0., 1e-12);
}

//! @brief Equations 2.122 through 2.131 and exercise 2.67
TEST(chapter2_2, ancillary_system)
{
    auto constexpr n = 1u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr MM = 3u;
    auto constexpr D = _2_pow_n * MM;

    auto const M = qpp::randkraus(MM, _2_pow_n);
    auto const completeness = std::transform_reduce(std::execution::seq, M.cbegin(), M.cend()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& Mm)
    {
        return (Mm.adjoint() * Mm).eval();
    });
    EXPECT_MATRIX_CLOSE(completeness, Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);

    auto U_partial = Eigen::MatrixXcd::Zero(D, _2_pow_n).eval();
    for (auto&& i : range)
    {
        auto const psi = Eigen::VectorXcd::Unit(_2_pow_n, i);
        for (auto&& m : std::views::iota(0u, MM))
            U_partial.col(i) += qpp::kron(M[m] * psi, Eigen::VectorXcd::Unit(MM, m));
    }
    auto const U_orthogonal = qpp::grams(U_partial.adjoint().fullPivLu().kernel());
    EXPECT_EQ(U_orthogonal.rows(), U_partial.rows());
    EXPECT_EQ(U_orthogonal.cols(), U_partial.rows() - U_partial.cols());

    auto U = Eigen::MatrixXcd::Identity(D, D).eval();
    auto it = U_orthogonal.colwise().cbegin();
    for (auto&& i : range)
        for (auto&& j : std::views::iota(0u, MM))
        {
            auto const psi_m = qpp::kron(Eigen::VectorXcd::Unit(_2_pow_n, i), Eigen::VectorXcd::Unit(MM, j));
            auto const col = qpp::zket2dits(psi_m, { D })->front();
            if(j == 0)
                U.col(col) = U_partial.col(i);
            else
                U.col(col) = *(it++);
        }
    debug() << "Matrix U:\n" << qpp::disp(U) << "\n\n";
    EXPECT_MATRIX_CLOSE(U.adjoint() * U, Eigen::MatrixXcd::Identity(D, D), 1e-12);

    auto P = std::vector<Eigen::MatrixXcd>(MM);
    for (auto&& m : std::views::iota(0u, MM))
        P[m] = qpp::kron(Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), qpp::prj(Eigen::VectorXcd::Unit(MM, m)));

    auto const psi = qpp::randket(_2_pow_n);
    auto const [result, probabilities, resulting_state] = qpp::measure(U * qpp::kron(psi, Eigen::VectorXcd::Unit(MM, 0)), P);

    for (auto&& m : std::views::iota(0u, MM))
    {
        auto const pm = psi.dot(M[m].adjoint() * M[m] * psi);
        EXPECT_COMPLEX_CLOSE(pm, probabilities[m], 1e-12);
        auto const post_measurement_state_m = (qpp::kron(M[m] * psi, Eigen::VectorXcd::Unit(MM, m)) / std::sqrt(probabilities[m])).eval();
        EXPECT_MATRIX_CLOSE(post_measurement_state_m, resulting_state[m], 1e-12);
    }
}
