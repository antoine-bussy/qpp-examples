#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <execution>
#include <ranges>

#include <qpp/qpp.hpp>
#include <qube/debug.hpp>
#include <qube/gates.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/gtest_macros.hpp>
#include <qube/maths/norm.hpp>
#include <qube/maths/random.hpp>
#include <qube/approximations.hpp>
#include <qube/decompositions.hpp>
#include <qube/introspection.hpp>

using namespace qube::stream;

//! @brief Equation 4.61
TEST(chapter4_5, operator_error)
{
    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr n = qube::maths::pow(2ul, nq);

    auto const U = qpp::randU(n);
    auto const V = qpp::randU(n);
    auto const W = (U - V).eval();

    auto const error = qube::maths::operator_norm_2(W);
    auto const psi_max = W.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeFullV).matrixV().col(0).eval();
    auto const e_max = qpp::norm(W * psi_max);
    auto constexpr epsilon = 1e-12;
    EXPECT_LT(e_max, error + epsilon);

    for ([[maybe_unused]] auto&& i: std::views::iota(0, 50))
    {
        auto const psi = qpp::randket(n);
        auto const e = (W * psi).norm();
        EXPECT_LT(e, error + epsilon);
    }
}

//! @brief Equation 4.62, and equations 4.64 through 4.68
TEST(chapter4_5_2, probability_error_bound)
{
    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr n = qube::maths::pow(2ul, nq);
    auto constexpr m = 5ul;

    auto const U = qpp::randU(n);
    auto const V = qpp::randU(n);
    auto const error = qube::maths::operator_norm_2(U - V);

    auto const Ks = qpp::randkraus(m, n);
    auto const M = (Ks[2].adjoint() * Ks[2]).eval();

    auto constexpr epsilon = 1e-12;

    for([[maybe_unused]] auto&& i: std::views::iota(0, 50))
    {
        auto const psi = qpp::randket(n);
        auto const [P_U, P_U_imag] = (U * psi).dot(M * (U * psi));
        auto const [P_V, P_V_imag] = (V * psi).dot(M * (V * psi));

        // Check that P_U and P_V are probabilities
        EXPECT_GT(P_U, -epsilon);
        EXPECT_LT(P_U, 1. + epsilon);
        EXPECT_LT(std::abs(P_U_imag), epsilon);
        EXPECT_GT(P_V, -epsilon);
        EXPECT_LT(P_V, 1. + epsilon);
        EXPECT_LT(std::abs(P_V_imag), epsilon);

        EXPECT_LT(std::abs(P_U - P_V), 2 * error + epsilon);
    }

}

//! @brief Equation 4.63, and equations 4.69 through 4.73
TEST(chapter4_5, operator_error_composition)
{
    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr n = qube::maths::pow(2ul, nq);
    auto constexpr m = 5;

    auto composed_error = 0.;
    auto U = Eigen::MatrixXcd::Identity(n, n).eval();
    auto V = Eigen::MatrixXcd::Identity(n, n).eval();

    for ([[maybe_unused]] auto&& i: std::views::iota(0, m))
    {
        auto const U_i = qpp::randU(n);
        auto const V_i = qpp::randU(n);
        composed_error += qube::maths::operator_norm_2(U_i - V_i);
        U = U_i * U;
        V = V_i * V;
    }

    auto const error = qube::maths::operator_norm_2(U - V);
    auto constexpr epsilon = 1e-12;
    EXPECT_LT(error, composed_error + epsilon);
}

//! @brief Equations 4.74 and 4.75
TEST(chapter4_5, H_T_phase_CNOT_universality)
{
    using namespace std::complex_literals;

    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto const T = (c * qpp::gt.Id2 - 1.i * s * qpp::gt.Z).eval();
    EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(T, qpp::gt.T, 1e-12);
    auto const HTH = (c * qpp::gt.Id2 - 1.i * s * qpp::gt.X).eval();
    EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(HTH, qpp::gt.H * qpp::gt.T * qpp::gt.H, 1e-12);

    auto const THTH = (c * c * qpp::gt.Id2 - 1.i * s * (c * (qpp::gt.X + qpp::gt.Z) + s * qpp::gt.Y)).eval();
    EXPECT_MATRIX_CLOSE(T * HTH, THTH, 1e-12);

    auto n = std::array{c, s, c};
    Eigen::Vector3d::Map(n.data()).normalize();
    auto constexpr theta = 2. * std::acos(c*c);
    auto const Rn_theta = qpp::gt.Rn(theta, n).eval();
    EXPECT_MATRIX_CLOSE(Rn_theta, THTH, 1e-12);
}

//! @brief Exercise 4.40
TEST(chapter4_5, H_T_phase_CNOT_universality_2)
{
    using namespace std::complex_literals;

    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto n = std::array{c, s, c};
    Eigen::Vector3d::Map(n.data()).normalize();

    auto const alpha = qpp::rand(0., 2. * pi);
    auto const beta = qpp::rand(0., 2. * pi);

    auto const error = qube::maths::operator_norm_2(qpp::gt.Rn(alpha, n) - qpp::gt.Rn(alpha + beta, n));
    auto const expected_error = std::abs(1. - std::exp(0.5i * beta));
    EXPECT_NEAR(error, expected_error, 1e-12);
}

//! @brief Equations 4.76 and 4.77
TEST(chapter4_5, H_T_phase_CNOT_universality_3)
{
    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto n = std::array{c, s, c};
    Eigen::Vector3d::Map(n.data()).normalize();
    auto constexpr theta = 2. * std::acos(c*c);

    auto alpha = qpp::rand(0., 2. * pi);
    // Don't ask for too high a precision, otherwise the test will reach floating-point precision limits
    auto constexpr epsilon = 1e-4;

    auto constexpr beta = 4. * std::asin(epsilon / 6.);
    auto constexpr delta = beta;
    auto constexpr N = static_cast<unsigned long int>(std::ceil(2. * pi / delta)) + 2ul;

    auto const R = std::views::iota(1ul, N) |
        std::views::transform([&](auto&& k)
        {
            return std::abs(std::fmod(k * theta, 2. * pi));
        });
    auto const k = 1ul + static_cast<unsigned long int>(std::ranges::distance(R.cbegin(), std::ranges::min_element(R)));
    auto const theta_k = std::fmod(k * theta, 2. * pi);

    EXPECT_LT(std::abs(theta_k), delta);
    EXPECT_NE(theta_k, 0.);

    if (theta_k < 0.)
        alpha -= 2. * pi;

    auto const m = static_cast<unsigned long int>(std::floor(alpha/theta_k));
    EXPECT_GE(m, 0ul);

    auto const Rn_alpha = qpp::gt.Rn(alpha, n).eval();
    auto const alpha_approx = std::fmod(m * k * theta, 2. * pi);
    auto const Rn_approx = qpp::gt.Rn(alpha_approx, n).eval();
    auto const error = qube::maths::operator_norm_2(Rn_alpha - Rn_approx);

    EXPECT_LT(error, epsilon / 3.);
}

//! @brief Equations 4.78 and 4.79
TEST(chapter4_5, H_T_phase_CNOT_universality_4)
{
    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto n = std::array{c, s, c};
    Eigen::Vector3d::Map(n.data()).normalize();
    auto m = std::array{c,-s, c};
    Eigen::Vector3d::Map(m.data()).normalize();

    auto const alpha = qpp::rand(0., 2. * std::numbers::pi);
    auto const& H = qpp::gt.H;

    auto const expected_Rm = qpp::gt.Rn(alpha, m).eval();
    auto const Rm = (H * qpp::gt.Rn(alpha, n) * H).eval();
    EXPECT_MATRIX_CLOSE(Rm, expected_Rm, 1e-12);

    debug() << ">> Rm:\n" << qpp::disp(Rm) << "\n\n";
    debug() << ">> expected_Rm:\n" << qpp::disp(expected_Rm) << "\n\n";
}

//! @brief Equations 4.80 and 4.81
//! @details Equation 4.80 is not verified in general.
//! @see Errata of N&C: https://michaelnielsen.org/qcqi/errata/errata/errata.html, for the correct formula
//! @details However, it holds on some necessary and sufficient condition: @see generalized_euler_decomposition_exists
TEST(chapter4_5, H_T_phase_CNOT_universality_5)
{
    using namespace std::complex_literals;

    qube::maths::seed(1754132672u);

    auto constexpr pi = std::numbers::pi;
    auto constexpr c = std::cos(pi / 8.);
    auto constexpr s = std::sin(pi / 8.);

    auto n = std::array{c, s, c};
    auto n_v = Eigen::Vector3d::Map(n.data());
    n_v.normalize();
    auto m = std::array{c,-s, c};
    auto m_v = Eigen::Vector3d::Map(m.data());
    m_v.normalize();

    auto const U = qpp::randU();
    auto const [phase_U, theta_U, n_U] = qube::unitary_to_rotation(U);
    auto alpha = qube::generalized_euler_decomposition(phase_U, theta_U, n_U, n_v, m_v, n_v);

    auto const computed_U = (std::exp(1.i * alpha[0])
        * qpp::gt.Rn(alpha[1], n)
        * qpp::gt.Rn(alpha[2], m)
        * qpp::gt.Rn(alpha[3], n)).eval();
    EXPECT_MATRIX_CLOSE(computed_U, U, 1e-12);

    auto constexpr theta = 2. * std::acos(c*c);
    // Don't ask for too high a precision, otherwise the test will reach floating-point precision limits
    auto constexpr epsilon = 1e-4;

    auto constexpr beta = 4. * std::asin(epsilon / 6.);
    auto constexpr delta = beta;

    auto alpha_approx = Eigen::Vector3d::Zero().eval();
    auto mk = Eigen::Vector3<unsigned long int>::Zero().eval();
    auto theta_k = Eigen::Vector3d::Zero().eval();

    for (auto&& i: std::views::iota(0, 3))
    {
        debug() << "\n";
        alpha[i+1] = std::fmod(alpha[i+1], 2. * pi);
        if (alpha[i+1] < 0.)
        {
            alpha[i+1] += 2. * pi;
            // Adding 2pi to alpha[i+1] multiplies by the corresponding rotation matrix by -1.
            // This equivalent to multiplying the global phase by -1. To compensate, we add pi to alpha[0].
            alpha[0] += pi;
        }
        std::tie(alpha_approx[i], mk[i], theta_k[i]) = qube::angle_approximation(alpha[i+1], theta, delta);
        EXPECT_LT(std::abs(theta_k[i]), delta);
        EXPECT_NE(theta_k[i], 0.);
    }

    auto const& H = qpp::gt.H;

    for (auto&& i: std::views::iota(0, 3))
    {
        auto R_approx = qpp::gt.Rn(alpha_approx[i], n).eval();
        if (i == 1)
            R_approx = H * R_approx * H;
        auto const R = qpp::gt.Rn(alpha[i+1], (i == 1 ? m : n)).eval();
        auto const error = qube::maths::operator_norm_2(R - R_approx);
        EXPECT_LT(error, epsilon / 3.);

        debug() << ">> R_approx:\n" << qpp::disp(R_approx) << "\n\n";
        debug() << ">> R:\n" << qpp::disp(R) << "\n\n";
        debug() << ">> error: " << error << "\n";
    }

    auto const U_approx = (std::exp(1.i * alpha[0])
        * qpp::gt.Rn(alpha_approx[0], n)
        * H * qpp::gt.Rn(alpha_approx[1], n) * H
        * qpp::gt.Rn(alpha_approx[2], n)).eval();

    auto const error = qube::maths::operator_norm_2(U - U_approx);
    EXPECT_LT(error, epsilon);

    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
    debug() << ">> U_approx:\n" << qpp::disp(U_approx) << "\n\n";
    debug() << ">> error: " << error << "\n";
}

//! @brief Exercise 4.41 and Figure 4.17
TEST(chapter4_5, universality_with_toffoli)
{
    using namespace qpp::literals;
    using namespace std::complex_literals;

    qube::maths::seed();

    auto constexpr nq = 3ul;

    auto circuit = qpp::QCircuit{ nq, 2ul }
        .gate(qpp::gt.H, 0)
        .gate(qpp::gt.H, 1);
    auto engine = qpp::QEngine{ circuit };

    auto const psi = qpp::randket();
    auto const alpha = psi[0];
    auto const beta = psi[1];
    auto const psi_0 = qpp::kron(00_ket, psi);
    debug() << ">> psi:\n" << qpp::disp(psi) << "\n\n";
    debug() << ">> psi_0:\n" << qpp::disp(psi_0) << "\n\n";

    engine.reset(psi_0).execute();
    auto const psi_1 = engine.get_state();
    debug() << ">> psi_1:\n" << qpp::disp(psi_1) << "\n\n";
    auto const psi_1_expected = (0.5 * (
        alpha * 000_ket + beta * 001_ket +
        alpha * 010_ket + beta * 011_ket +
        alpha * 100_ket + beta * 101_ket +
        alpha * 110_ket + beta * 111_ket)).eval();
    EXPECT_MATRIX_CLOSE(psi_1, psi_1_expected, 1e-12);

    circuit.CTRL(qpp::gt.X, { 0, 1 }, 2);
    engine.reset(psi_0).execute();
    auto const psi_2 = engine.get_state();
    debug() << ">> psi_2:\n" << qpp::disp(psi_2) << "\n\n";
    auto const psi_2_expected = (0.5 * (
        alpha * 000_ket + beta * 001_ket +
        alpha * 010_ket + beta * 011_ket +
        alpha * 100_ket + beta * 101_ket +
        beta * 110_ket + alpha * 111_ket)).eval();
    EXPECT_MATRIX_CLOSE(psi_2, psi_2_expected, 1e-12);

    circuit.gate(qpp::gt.S, 2);
    engine.reset(psi_0).execute();
    auto const psi_3 = engine.get_state();
    debug() << ">> psi_3:\n" << qpp::disp(psi_3) << "\n\n";
    auto const psi_3_expected = (0.5 * (
        alpha * 000_ket + beta * 1.i * 001_ket +
        alpha * 010_ket + beta * 1.i * 011_ket +
        alpha * 100_ket + beta * 1.i * 101_ket +
        beta * 110_ket + alpha * 1.i * 111_ket)).eval();
    EXPECT_MATRIX_CLOSE(psi_3, psi_3_expected, 1e-12);

    circuit.CTRL(qpp::gt.X, { 0, 1 }, 2);
    engine.reset(psi_0).execute();
    auto const psi_4 = engine.get_state();
    debug() << ">> psi_4:\n" << qpp::disp(psi_4) << "\n\n";
    auto const psi_4_expected = (0.5 * (
        alpha * 000_ket + beta * 1.i * 001_ket +
        alpha * 010_ket + beta * 1.i * 011_ket +
        alpha * 100_ket + beta * 1.i * 101_ket +
        alpha * 1.i * 110_ket + beta * 111_ket)).eval();
    EXPECT_MATRIX_CLOSE(psi_4, psi_4_expected, 1e-12);

    circuit
        .gate(qpp::gt.H, 0)
        .gate(qpp::gt.H, 1);
    engine.reset(psi_0).execute();
    auto const psi_5 = engine.get_state();
    debug() << ">> psi_5:\n" << qpp::disp(psi_5) << "\n\n";
    auto const psi_5_expected = (0.25 * (
        alpha * ( 3. + 1.i) * 000_ket + beta * ( 1. + 3.i) * 001_ket +
        alpha * ( 1. - 1.i) * 010_ket + beta * (-1. + 1.i) * 011_ket +
        alpha * ( 1. - 1.i) * 100_ket + beta * (-1. + 1.i) * 101_ket +
        alpha * (-1. + 1.i) * 110_ket + beta * ( 1. - 1.i) * 111_ket)).eval();
    EXPECT_MATRIX_CLOSE(psi_5, psi_5_expected, 1e-12);

    auto const U = (0.25 * Eigen::Vector2cd{3. + 1.i,  1. + 3.i}).asDiagonal().toDenseMatrix().eval();
    auto const V = (0.25 * Eigen::Vector2cd{1. - 1.i, -1. + 1.i}).asDiagonal().toDenseMatrix().eval();
    auto psi_5_factorized = (qpp::kron(00_ket, U * psi) + qpp::kron(01_ket + 10_ket - 11_ket, V * psi)).eval();
    EXPECT_MATRIX_CLOSE(psi_5, psi_5_factorized, 1e-12);

    auto constexpr pi = std::numbers::pi;
    auto constexpr sqrt2 = std::numbers::sqrt2;
    auto constexpr sqrt5 = std::sqrt(5.);
    auto constexpr sqrt10 = std::sqrt(10.);

    auto const U_factorized = (0.25 * sqrt2 * std::exp(1.i * pi / 4.)
        * Eigen::Vector2cd{2. - 1.i, 2. + 1.i}).asDiagonal().toDenseMatrix().eval();
    EXPECT_MATRIX_CLOSE(U, U_factorized, 1e-12);
    auto const Rz = (2./sqrt5 * qpp::gt.Id2 - 1.i / sqrt5 * qpp::gt.Z).eval();
    auto const U_factorized_2 = (0.25 * sqrt10 * std::exp(1.i * pi / 4.) * Rz).eval();
    auto const U_norm2 = qube::maths::pow(0.25 * sqrt10, 2);
    EXPECT_MATRIX_CLOSE(U, U_factorized_2, 1e-12);
    auto constexpr theta = std::acos(3./5.);
    auto const expected_Rz = qpp::gt.RZ(theta).eval();
    EXPECT_MATRIX_CLOSE(Rz, expected_Rz, 1e-12);

    auto const V_factorized = (0.25 * sqrt2 * std::exp(-1.i * pi / 4.) * qpp::gt.Z).eval();
    auto const V_norm2 = qube::maths::pow(0.25 * sqrt2, 2);
    EXPECT_MATRIX_CLOSE(V, V_factorized, 1e-12);

    // We want probabilities and all states after measurements, which is not available in QEngine
    // So we use qpp::measure() instead
    auto const [result, probabilities, resulting_state] = qpp::measure(psi_5, Eigen::Matrix4cd::Identity(), { 0, 1 });

    debug() << ">> Measurement result: " << result << '\n';
    debug() << ">> Probabilities: ";
    debug() << qpp::disp(probabilities, {", "}) << '\n';
    debug() << ">> Resulting states:\n";
    for (auto&& state : resulting_state)
        debug() << qpp::disp(state) << "\n\n";

    auto const U_psi = (U * psi).normalized().eval();
    EXPECT_MATRIX_CLOSE(resulting_state[0], U_psi, 1e-12);
    auto const Rz_psi = (Rz * psi).eval();
    EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(U_psi, Rz_psi, 1e-12);
    EXPECT_NEAR(U_norm2, 0.625, 1e-12);
    EXPECT_NEAR(probabilities[0], 0.625, 1e-12);

    auto const V_psi = (V * psi).normalized().eval();
    EXPECT_MATRIX_CLOSE(resulting_state[1], V_psi, 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state[2], V_psi, 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state[3],-V_psi, 1e-12);
    auto const Z_psi = (qpp::gt.Z * psi).eval();
    EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(V_psi, Z_psi, 1e-12);
    EXPECT_NEAR(V_norm2, 0.125, 1e-12);
    EXPECT_NEAR(probabilities[1], 0.125, 1e-12);
    EXPECT_NEAR(probabilities[2], 0.125, 1e-12);
    EXPECT_NEAR(probabilities[3], 0.125, 1e-12);
}

//! @brief Exercise 4.41 and Figure 4.17
//! @details Instead of applying the subcircuit iteratively (using a "classical" iteration), we build a big circuit
//! that applies the subcircuit multiple times, and then we measure the state using deferred measurement and qpp::measure().
//! This allows us to get the probabilities from qpp::measure() and to check the resulting states.
TEST(chapter4_5, universality_with_toffoli_2)
{
    using namespace std::complex_literals;
    using namespace qpp::literals;

    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;

    auto subcircuit = qpp::QCircuit{ 3ul }
        .gate(qpp::gt.H, 1)
        .gate(qpp::gt.H, 2)
        .CTRL(qpp::gt.X, { 1, 2 }, 0)
        .gate(qpp::gt.S, 0)
        .CTRL(qpp::gt.X, { 1, 2 }, 0)
        .gate(qpp::gt.H, 1)
        .gate(qpp::gt.H, 2);
    // Use matrix instead of circuit instead of compose_CTRL_circuit from qpp::QCircuit.
    // It seems simpler to me.
    // Also apply a global phase, so that the 3 last resulting states are equal to the initial state
    auto const U = (std::exp(1.i * pi / 4.) * qube::extract_matrix<8>(subcircuit)).eval();

    auto const psi = qpp::randket();
    auto const Rz = qpp::gt.RZ(std::acos(3./5.)).eval();
    auto const Rz_psi = (Rz * psi).eval();

    {
        auto const psi_out = (U * qpp::kron(psi, 00_ket)).eval();
        auto const [result, probabilities, resulting_state] = qpp::measure(psi_out, Eigen::Matrix4cd::Identity(), { 1, 2 });
        EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(resulting_state[0], Rz_psi, 1e-12);
        EXPECT_NEAR(probabilities[0], 0.625, 1e-12);
    }

    auto const or_CTRL_Z = qube::or_CTRL(qpp::gt.Z);
    auto const or_CTRL_U = qube::or_CTRL(U);

    for (auto&& i: std::views::iota(1ul, 7ul))
    {
        debug() << ">> Number of iterations: " << i << "\n";
        auto const nq = 1ul + 2ul * i;

        // We use deferred measurement, so we can measure the state after applying all gates.
        auto circuit = qpp::QCircuit{ nq }
            .gate(U, { 0, 1, 2 })
            .gate(or_CTRL_Z, { 1, 2, 0 });

        for (auto&& j: std::views::iota(1ul, i))
        {
            auto const current_qbit_1 = 1 + 2 * j;
            auto const current_qbit_2 = current_qbit_1 + 1;
            auto const previous_qbit_1 = current_qbit_1 - 2;
            auto const previous_qbit_2 = previous_qbit_1 + 1;
            circuit
                .gate(or_CTRL_U, { previous_qbit_1, previous_qbit_2, 0, current_qbit_1, current_qbit_2 })
                .gate(or_CTRL_Z, { current_qbit_1, current_qbit_2, 0 });
        }

        auto engine = qpp::QEngine{ circuit };
        auto const ket_0 = Eigen::VectorXcd::Unit(qube::maths::pow(2ul, nq - 1), 0);
        auto const psi_0 = qpp::kron(psi, ket_0).eval();
        engine.reset(psi_0).execute();
        auto const psi_out = engine.get_state();

        auto target = std::vector<qpp::idx>(2 * i);
        std::ranges::iota(target, 1ul);
        auto const D = qube::maths::pow(2ul, target.size());
        auto const [result, probabilities, resulting_state] = qpp::measure(psi_out, Eigen::MatrixXcd::Identity(D, D), target);

        ASSERT_EQ(probabilities.size(), resulting_state.size());

        auto p = 0.;
        auto p_bar = 0.;

        for (auto&& i: std::views::iota(0ul, probabilities.size()))
        {
            auto constexpr precision = 1e-9;

            auto const& state = resulting_state[i];
            auto const& prob = probabilities[i];

            if (state.isZero(precision))
            {
                EXPECT_NEAR(prob, 0., precision);
                continue;
            }

            if (qube::maths::matrix_close_up_to_phase_factor_l(state, Rz_psi, precision))
            {
                p += prob;
                continue;
            }

            if (qube::maths::matrix_close_up_to_phase_factor_l(state, psi, precision))
            {
                p_bar += prob;
                continue;
            }

            ADD_FAILURE() << "Unexpected state of probability " << prob << ":\n" << qpp::disp(state) << "\n";
        }

        EXPECT_NEAR(p, 1. - p_bar, 1e-12);
        EXPECT_NEAR(p_bar, qube::maths::pow(0.375, i), 1e-12);

        debug() << ">> p: " << p << "\n";
        debug() << ">> p_bar: " << p_bar << "\n";
    }
}

//! @brief This is not an exercise from Nielsen & Chuang, but it is a simple example of how to reset the state of a qubit to |0>.
TEST(chapter4_5, reset_circuit)
{
    using namespace qpp::literals;

    qube::maths::seed();

    auto circuit = qpp::QCircuit{ 1ul, 1ul }
        .measure(0, 0, false)
        .cCTRL(qpp::gt.X, { 0 }, 0);

    auto const U = qube::extract_matrix<2>(circuit);
    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
    EXPECT_MATRIX_CLOSE(U, (Eigen::Matrix2cd{ { 1., 1. }, { 0., 0. } }), 1e-12);

    auto const psi = qpp::randket();
    auto engine = qpp::QEngine{ circuit };
    engine.reset(psi).execute();
    auto const psi_out = engine.get_state();
    debug() << ">> psi_out:\n" << qpp::disp(psi_out) << "\n\n";
    EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(psi_out, 0_ket, 1e-12);
}

//! @brief Exercise 4.41 and Figure 4.17
//! @details Same as universality_with_toffoli_2, but with reusing qubits,
//! thus using less memory.
TEST(chapter4_5, universality_with_toffoli_3)
{
    using namespace std::complex_literals;
    using namespace qpp::literals;

    qube::maths::seed();

    auto constexpr pi = std::numbers::pi;

    auto subcircuit = qpp::QCircuit{ 3ul }
        .gate(qpp::gt.H, 1)
        .gate(qpp::gt.H, 2)
        .CTRL(qpp::gt.X, { 1, 2 }, 0)
        .gate(qpp::gt.S, 0)
        .CTRL(qpp::gt.X, { 1, 2 }, 0)
        .gate(qpp::gt.H, 1)
        .gate(qpp::gt.H, 2);
    // Use matrix instead of circuit instead of compose_CTRL_circuit from qpp::QCircuit.
    // It seems simpler to me.
    // Also apply a global phase, so that the 3 last resulting states are equal to the initial state
    auto const U = (std::exp(1.i * pi / 4.) * qube::extract_matrix<8>(subcircuit)).eval();

    auto const psi = qpp::randket();
    auto const Rz = qpp::gt.RZ(std::acos(3./5.)).eval();
    auto const Rz_psi = (Rz * psi).eval();

    auto const or_CNOT = qube::or_CNOT();

    auto constexpr max_iterations = 9ul;
    auto constexpr engine_repetitions = 100ul;
    auto rotation_probabilities = Eigen::VectorXd::Zero(max_iterations).eval();
    auto logs = std::vector<std::stringstream>(max_iterations);

    auto constexpr range = std::views::iota(1ul, max_iterations + 1ul);
    std::for_each(std::execution::par, range.begin(), range.end(),
        [&](auto&& i)
    {
        auto const debug = [&logs, i]() -> std::stringstream&
        {
            return logs[i - 1];
        };
        debug() << ">> Number of iterations: " << i << "\n";
        auto const nq = 4ul;

        // Trick to fill dit with 1.
        auto circuit = qpp::QCircuit{ nq, 1ul }
            .gate(qpp::gt.X, 3)
            .measure(3, 0, false)
            .gate(qpp::gt.X, 3);

        for ([[maybe_unused]] auto&& j: std::views::iota(0ul, i))
        {
            circuit
                .cCTRL(U, 0, { 0, 1, 2 })
                .gate(or_CNOT, { 1, 2, 3 })
                .measure(3, 0, false)
                .cCTRL(qpp::gt.Z, 0, 0)
                .reset({ 1, 2 })
                .cCTRL(qpp::gt.X, 0, 3) // Reset qubit 3 to |0> without new measurement
            ;
        }
        // Last cCTRL reset qubit 3 to |0>. We revert this last cCTRL by applying it again (involution).
        // Thus we get (after measurement) whether the rotation was applied or not.
        circuit.cCTRL(qpp::gt.X, 0, 3);

        auto engine = qpp::QEngine{ circuit };
        auto const psi_0 = qpp::kron(psi, 000_ket).eval();
        engine.reset(psi_0).execute(engine_repetitions);
        debug() << engine.get_stats() << "\n";
        auto const& stats = engine.get_stats();
        rotation_probabilities[i - 1] = stats.data().at({0}) / static_cast<double>(stats.get_num_reps());

        auto const [result, probabilities, resulting_state] = qpp::measure(engine.get_state(), Eigen::MatrixXcd::Identity(8, 8), {1, 2, 3});
        auto const& psi_out = resulting_state[result];
        auto const probabilities_eigen = Eigen::VectorXd::Map(probabilities.data(), probabilities.size());
        EXPECT_THAT(result, ::testing::AnyOf(0, 1));
        EXPECT_MATRIX_CLOSE(probabilities_eigen, Eigen::VectorXd::Unit(8, result), 1e-12);

        auto const measured_ket = qpp::n2multiidx(result, std::vector<qpp::idx>(3, 2));

        debug() << ">> Measurement result: " << qpp::disp(measured_ket, { "", "|", ">" }) << '\n';
        debug() << ">> Probabilities: " << qpp::disp(probabilities, {", "}) << '\n';
        debug() << ">> Resulting state:\n" << qpp::disp(resulting_state[result]) << "\n\n";

        debug() << ">> psi:\n" << qpp::disp(psi) << "\n\n";
        debug() << ">> Rz_psi:\n" << qpp::disp(Rz_psi) << "\n\n";
        debug() << ">> psi_out:\n" << qpp::disp(psi_out) << "\n\n";

        auto const& expected_psi_out = (result == 0 ? Rz_psi : psi).eval();
        EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(psi_out, expected_psi_out, 1e-12);
    });

    for (auto&& ss : logs)
        debug() << ss.rdbuf();

    debug() << ">> Rotation probabilities:\n" << qpp::disp(rotation_probabilities.transpose(), {", "}) << "\n\n";
}

//! @brief Exercise 4.43
//! @details Approximate the T gate using the RZ(theta) gate, with theta = acos(3/5),
//! using the same method as in H_T_phase_CNOT_universality_3.
//! T is an universal gate (plus phase, H, and CNOT), and RZ(theta) is constructed with Toffoli (plus phase, H, and CNOT),
//! proving the universality of Toffoli (plus phase, H, and CNOT).
TEST(chapter4_5, universality_with_toffoli_4)
{
    using namespace std::complex_literals;

    auto constexpr pi = std::numbers::pi;
    auto constexpr theta = 2. * std::acos(3./5.);

    // Don't ask for too high a precision, otherwise the test will reach floating-point precision limits
    auto constexpr epsilon = 1e-4;

    auto constexpr beta = 4. * std::asin(epsilon / 6.);
    auto constexpr delta = beta;
    auto constexpr N = static_cast<unsigned long int>(std::ceil(2. * pi / delta)) + 2ul;

    auto const R = std::views::iota(1ul, N) |
        std::views::transform([&](auto&& k)
        {
            return std::abs(std::fmod(k * theta, 2. * pi));
        });
    auto const k = 1ul + static_cast<unsigned long int>(std::ranges::distance(R.cbegin(), std::ranges::min_element(R)));
    auto const theta_k = std::fmod(k * theta, 2. * pi);

    EXPECT_LT(std::abs(theta_k), delta);
    EXPECT_NE(theta_k, 0.);

    auto const alpha = (theta_k >= 0. ? 1. : -7) * pi / 4.;

    auto const m = static_cast<unsigned long int>(std::floor(alpha/theta_k));
    EXPECT_GE(m, 0ul);

    auto const alpha_approx = std::fmod(m * k * theta, 2. * pi);
    // Add global phase to the approximation
    auto const T_approx = (std::exp(1.i * pi / 8.) * qpp::gt.RZ(alpha_approx)).eval();
    auto const error = qube::maths::operator_norm_2(T_approx - qpp::gt.T);

    debug() << ">> k: " << k << "\n";
    debug() << ">> theta_k: " << theta_k << "\n";
    debug() << ">> alpha: " << alpha << "\n";
    debug() << ">> m: " << m << "\n";
    debug() << ">> alpha_approx: " << alpha_approx << "\n";
    debug() << ">> T_approx:\n" << qpp::disp(T_approx) << "\n\n";
    debug() << ">> T:\n" << qpp::disp(qpp::gt.T) << "\n\n";
    debug() << ">> error: " << error << "\n";

    EXPECT_LT(error, epsilon / 3.);
}
