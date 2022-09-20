#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/qube/debug.hpp>

#include <numbers>

using namespace qpp_e::qube::stream;

//! @brief Equations 1.8 through 1.12
TEST(chapter1_3, not_gate)
{
    auto const state = qpp::randket(2).eval();
    auto const not_state = (qpp::gt.X * state).eval();

    EXPECT_MATRIX_EQ(not_state, state.reverse());
    EXPECT_MATRIX_EQ(qpp::gt.X, Eigen::Matrix2cd::Identity().rowwise().reverse());

    debug() << ">> State:\n" << qpp::disp(state) << '\n';
    debug() << ">> NOT Gate:\n" << qpp::disp(qpp::gt.X) << '\n';
    debug() << ">> NOT State:\n" << qpp::disp(not_state) << '\n';
}

//! @brief Equation 1.13
TEST(chapter1_3, z_gate)
{
    auto const state = qpp::randket(2).eval();
    auto const z_state = (qpp::gt.Z * state).eval();

    EXPECT_EQ(z_state[0], state[0]);
    EXPECT_EQ(z_state[1],-state[1]);
    EXPECT_MATRIX_EQ(qpp::gt.Z, Eigen::Vector2cd(1, -1).asDiagonal().toDenseMatrix());

    debug() << ">> State:\n" << qpp::disp(state) << '\n';
    debug() << ">> Z Gate:\n" << qpp::disp(qpp::gt.Z) << '\n';
    debug() << ">> Z State:\n" << qpp::disp(z_state) << '\n';
}

//! @brief Equation 1.14
TEST(chapter1_3, hadamard_gate)
{
    auto const state = qpp::randket(2).eval();
    auto const h_state = (qpp::gt.H * state).eval();

    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    EXPECT_COMPLEX_CLOSE(h_state[0], (state[0] + state[1]) * inv_sqrt2, 1e-12);
    EXPECT_COMPLEX_CLOSE(h_state[1], (state[0] - state[1]) * inv_sqrt2, 1e-12);

    debug() << ">> State:\n" << qpp::disp(state) << '\n';
    debug() << ">> H Gate:\n" << qpp::disp(qpp::gt.H) << '\n';
    debug() << ">> H State:\n" << qpp::disp(h_state) << '\n';
}

//! @brief Equations 1.15 and 1.16, and Block 1.1
TEST(chapter1_3, general_single_qubit_gate)
{
    auto constexpr alpha = 0.57;
    auto constexpr beta  = -1.3;
    auto constexpr gamma = -0.86;
    auto constexpr delta = 1.02;

    auto const phase_shift = std::polar(1., alpha);
    EXPECT_MATRIX_CLOSE(qpp::gt.RZ(beta), Eigen::Vector2cd(std::polar(1., -0.5 * beta), std::polar(1., 0.5 * beta)).asDiagonal().toDenseMatrix(), 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::gt.RY(gamma), Eigen::Rotation2Dd(0.5 * gamma).toRotationMatrix().cast<std::complex<double>>(), 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::gt.RZ(delta), Eigen::Vector2cd(std::polar(1., -0.5 * delta), std::polar(1., 0.5 * delta)).asDiagonal().toDenseMatrix(), 1e-12);

    // Note: it looks like Euler angles
    auto const U = (phase_shift * qpp::gt.RZ(beta) * qpp::gt.RY(gamma) * qpp::gt.RZ(delta)).eval();
    EXPECT_MATRIX_CLOSE(U * U.adjoint(), Eigen::Matrix2cd::Identity(), 1e-12);

    debug() << ">> Phase shift:\n" << phase_shift << '\n';
    debug() << ">> Rotation-Z Beta:\n" << qpp::disp(qpp::gt.RZ(beta)) << '\n';
    debug() << ">> Rotation Gamma:\n" << qpp::disp(qpp::gt.RY(gamma)) << '\n';
    debug() << ">> Rotation-Z Delta:\n" << qpp::disp(qpp::gt.RZ(delta)) << '\n';
    debug() << ">> General single qubit gate:\n" << qpp::disp(U) << '\n';
}

//! @brief Equation 1.18 and Figure 1.6
TEST(chapter1_3, cnot_gate)
{
    using namespace qpp::literals;

    auto cnot_matrix = Eigen::Matrix4cd::Zero().eval();
    cnot_matrix(0, 0) = 1.;
    cnot_matrix(1, 1) = 1.;
    cnot_matrix(2, 3) = 1.;
    cnot_matrix(3, 2) = 1.;

    EXPECT_MATRIX_EQ(qpp::gt.CNOT, cnot_matrix);

    EXPECT_MATRIX_EQ(qpp::gt.CNOT * 00_ket, 00_ket);
    EXPECT_MATRIX_EQ(qpp::gt.CNOT * 01_ket, 01_ket);
    EXPECT_MATRIX_EQ(qpp::gt.CNOT * 10_ket, 11_ket);
    EXPECT_MATRIX_EQ(qpp::gt.CNOT * 11_ket, 10_ket);

    EXPECT_MATRIX_EQ(qpp::gt.CNOT * qpp::gt.CNOT.adjoint(), Eigen::Matrix4cd::Identity());

    debug() << ">> CNOT Gate:\n" << qpp::disp(qpp::gt.CNOT) << '\n';
}

//! @brief Equation 1.19
TEST(chapter1_3, plus_minus_states)
{
    using namespace qpp::literals;

    auto const plus_ket = qpp::st.plus();
    auto const minus_ket = qpp::st.minus();

    EXPECT_MATRIX_CLOSE(plus_ket, (0_ket + 1_ket).normalized(), 1e-12);
    EXPECT_MATRIX_CLOSE(minus_ket, (0_ket - 1_ket).normalized(), 1e-12);

    auto const state = qpp::randket().eval();
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;
    EXPECT_MATRIX_CLOSE(state, ((state[0] + state[1]) * plus_ket + (state[0] - state[1]) * minus_ket) * inv_sqrt2, 1e-12);

    debug() << ">> |+> State:\n" << qpp::disp(plus_ket) << '\n';
    debug() << ">> |-> State:\n" << qpp::disp(minus_ket) << '\n';
}

//! @brief Equation 1.19
TEST(chapter1_3, plus_minus_basis_measure)
{
    using namespace qpp::literals;
    auto const plus_minus_basis = (Eigen::Matrix2cd() << qpp::st.plus(), qpp::st.minus()).finished();
    EXPECT_MATRIX_EQ(plus_minus_basis, qpp::gt.H);

    auto const state = qpp::randket().eval();
    auto const [result, probabilities, resulting_state] = qpp::measure(state, qpp::gt.H);

    EXPECT_THAT((std::array{ 0, 1 }), testing::Contains(result));
    EXPECT_NEAR(probabilities[0], 0.5 * std::norm(state[0] + state[1]), 1e-12);
    EXPECT_NEAR(probabilities[1], 0.5 * std::norm(state[0] - state[1]), 1e-12);
    EXPECT_COLLINEAR(resulting_state[0], qpp::st.plus(), 1e-12);
    EXPECT_COLLINEAR(resulting_state[1], qpp::st.minus(), 1e-12);

    debug() << ">> State:\n" << qpp::disp(state) << '\n';
    debug() << ">> Measurement result: " << result << '\n';
    debug() << ">> Probabilities: ";
    debug() << qpp::disp(probabilities, ", ") << '\n';
    debug() << ">> Resulting states:\n";
    for (auto&& it : resulting_state)
        debug() << qpp::disp(it) << "\n\n";
}

//! @brief Equation 1.19
TEST(chapter1_3, general_basis_measure)
{
    using namespace qpp::literals;

    auto const a = qpp::randket().eval();
    auto const b = Eigen::Vector2cd(a[1], -a[0]).conjugate().eval();

    EXPECT_NEAR(a.squaredNorm(), 1., 1e-12);
    EXPECT_NEAR(b.squaredNorm(), 1., 1e-12);
    EXPECT_NEAR(std::norm(a.dot(b)), 0., 1e-12);

    auto const basis = (Eigen::Matrix2cd() << a, b).finished();
    EXPECT_MATRIX_CLOSE(basis.adjoint() * basis, Eigen::Matrix2cd::Identity(), 1e-12);

    auto const state = qpp::randket().eval();
    auto const [result, probabilities, resulting_state] = qpp::measure(state, basis);

    auto const alpha = a.dot(state);
    auto const beta = b.dot(state);
    EXPECT_MATRIX_CLOSE(alpha * a + beta * b, state, 1e-12);

    EXPECT_THAT((std::array{ 0, 1 }), testing::Contains(result));
    EXPECT_NEAR(probabilities[0], std::norm(alpha), 1e-12);
    EXPECT_NEAR(probabilities[1], std::norm(beta), 1e-12);
    EXPECT_COLLINEAR(resulting_state[0], a, 1e-12);
    EXPECT_COLLINEAR(resulting_state[1], b, 1e-12);

    debug() << ">> State:\n" << qpp::disp(state) << '\n';
    debug() << ">> Measurement result: " << result << '\n';
    debug() << ">> Probabilities: ";
    debug() << qpp::disp(probabilities, ", ") << '\n';
    debug() << ">> Resulting states:\n";
    for (auto&& it : resulting_state)
        debug() << qpp::disp(it) << "\n\n";
}

//! @brief Equation 1.20 and Figure 1.7
TEST(chapter1_3, swap_gate)
{
    using namespace qpp::literals;

    auto const U = (qpp::gt.CNOT * qpp::gt.CNOTba * qpp::gt.CNOT).eval();
    EXPECT_MATRIX_CLOSE(U, qpp::gt.SWAP, 1e-12);

    EXPECT_MATRIX_EQ(U * 00_ket, 00_ket);
    EXPECT_MATRIX_EQ(U * 01_ket, 10_ket);
    EXPECT_MATRIX_EQ(U * 10_ket, 01_ket);
    EXPECT_MATRIX_EQ(U * 11_ket, 11_ket);

    debug() << ">> SWAP:\n" << qpp::disp(qpp::gt.SWAP) << '\n';
}

//! @brief Equation 1.20 and Figure 1.7
TEST(chapter1_3, swap_circuit)
{
    using namespace qpp::literals;

    auto constexpr circuit = [](auto const& s0)
    {
        auto const s1 = qpp::apply(s0, qpp::gt.CNOT, { 0, 1 });
        auto const s2 = qpp::apply(s1, qpp::gt.CNOT, { 1, 0 });
        auto const s3 = qpp::apply(s2, qpp::gt.CNOT, { 0, 1 });
        return s3;
    };

    EXPECT_MATRIX_EQ(circuit(00_ket), 00_ket);
    EXPECT_MATRIX_EQ(circuit(01_ket), 10_ket);
    EXPECT_MATRIX_EQ(circuit(10_ket), 01_ket);
    EXPECT_MATRIX_EQ(circuit(11_ket), 11_ket);
}

//! @brief Figure 1.9
TEST(chapter1_3, controlled_u)
{
    using namespace qpp::literals;

    auto const controlled_X = qpp::gt.CTRL(qpp::gt.X, { 0 }, { 1 }, 2);
    EXPECT_MATRIX_EQ(controlled_X, qpp::gt.CNOT);

    auto const controlled_Z = qpp::gt.CTRL(qpp::gt.Z, { 0 }, { 1 }, 2);
    EXPECT_MATRIX_EQ(controlled_Z, Eigen::Vector4cd(1, 1, 1, -1).asDiagonal().toDenseMatrix());

    debug() << ">> Controlled-X:\n" << qpp::disp(controlled_X) << '\n';
    debug() << ">> Controlled-Z:\n" << qpp::disp(controlled_Z) << '\n';
}

//! @brief Equations 1.21 and 1.22, and Figure 1.11
TEST(chapter1_3, qubit_copy)
{
    using namespace qpp::literals;

    EXPECT_MATRIX_EQ(qpp::kron(0_ket, 0_ket), 00_ket);
    EXPECT_MATRIX_EQ(qpp::kron(1_ket, 0_ket), 10_ket);

    auto constexpr copy = [](auto const& state)
    {
        return (qpp::gt.CNOT * qpp::kron(state, 0_ket)).eval();
    };

    EXPECT_MATRIX_EQ(copy(0_ket), 00_ket);
    EXPECT_MATRIX_EQ(copy(1_ket), 11_ket);

    auto const state = qpp::randket().eval();

    ASSERT_NE(state[0], 0.);
    ASSERT_NE(state[1], 0.);
    EXPECT_NE(copy(state), qpp::kron(state, state));

    debug() << ">> |0>|0>:\n" << qpp::disp(qpp::kron(0_ket, 0_ket)) << '\n';
    debug() << ">> |1>|0>:\n" << qpp::disp(qpp::kron(1_ket, 0_ket)) << '\n';
    debug() << ">> copy(|0>):\n" << qpp::disp(copy(0_ket)) << '\n';
    debug() << ">> copy(|1>):\n" << qpp::disp(copy(1_ket)) << '\n';
    debug() << ">> |psi>:\n" << qpp::disp(state) << '\n';
    debug() << ">> copy(|psi>):\n" << qpp::disp(copy(state)) << '\n';
    debug() << ">> |psi>|psi>:\n" << qpp::disp(qpp::kron(state, state)) << '\n';
}

//! @brief Equations 1.23 through 1.26, and Figure 1.12 left
TEST(chapter1_3, bell_states)
{
    using namespace qpp::literals;

    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    EXPECT_MATRIX_CLOSE(qpp::st.b00, (00_ket + 11_ket) * inv_sqrt2, 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::st.b01, (01_ket + 10_ket) * inv_sqrt2, 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::st.b10, (00_ket - 11_ket) * inv_sqrt2, 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::st.b11, (01_ket - 10_ket) * inv_sqrt2, 1e-12);

    debug() << ">> b00:\n" << qpp::disp(qpp::st.b00) << '\n';
    debug() << ">> b01:\n" << qpp::disp(qpp::st.b01) << '\n';
    debug() << ">> b10:\n" << qpp::disp(qpp::st.b10) << '\n';
    debug() << ">> b11:\n" << qpp::disp(qpp::st.b11) << '\n';
}

//! @brief Equation 1.27
TEST(chapter1_3, bell_state_mnemonic)
{
    using namespace qpp::literals;

    auto constexpr bell = [](auto&& x, auto&& y)
    {
        auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;
        assert(std::set({0, 1}).contains(x));
        assert(std::set({0, 1}).contains(y));
        return ((qpp::kron(0_ket, Eigen::Vector2cd::Unit(y)) + qpp_e::maths::pow(-1, x) * qpp::kron(1_ket, Eigen::Vector2cd::Unit(1 - y))) * inv_sqrt2).eval();
    };

    EXPECT_MATRIX_CLOSE(bell(0, 0), qpp::st.b00, 1e-12);
    EXPECT_MATRIX_CLOSE(bell(0, 1), qpp::st.b01, 1e-12);
    EXPECT_MATRIX_CLOSE(bell(1, 0), qpp::st.b10, 1e-12);
    EXPECT_MATRIX_CLOSE(bell(1, 1), qpp::st.b11, 1e-12);

    debug() << ">> b00:\n" << qpp::disp(bell(0, 0)) << '\n';
    debug() << ">> b01:\n" << qpp::disp(bell(0, 1)) << '\n';
    debug() << ">> b10:\n" << qpp::disp(bell(1, 0)) << '\n';
    debug() << ">> b11:\n" << qpp::disp(bell(1, 1)) << '\n';
}

//! @brief Figure 1.12 right
TEST(chapter1_3, bell_state_circuit)
{
    using namespace qpp::literals;

    auto constexpr circuit = [](auto&& x, auto&& y)
    {
        return (qpp::gt.CNOT * qpp::kron(qpp::gt.H * x, y)).eval();
    };
    EXPECT_MATRIX_CLOSE(circuit(0_ket, 0_ket), qpp::st.b00, 1e-12);
    EXPECT_MATRIX_CLOSE(circuit(0_ket, 1_ket), qpp::st.b01, 1e-12);
    EXPECT_MATRIX_CLOSE(circuit(1_ket, 0_ket), qpp::st.b10, 1e-12);
    EXPECT_MATRIX_CLOSE(circuit(1_ket, 1_ket), qpp::st.b11, 1e-12);

    debug() << ">> b00:\n" << qpp::disp(circuit(0_ket, 0_ket)) << '\n';
    debug() << ">> b01:\n" << qpp::disp(circuit(0_ket, 1_ket)) << '\n';
    debug() << ">> b10:\n" << qpp::disp(circuit(1_ket, 0_ket)) << '\n';
    debug() << ">> b11:\n" << qpp::disp(circuit(1_ket, 1_ket)) << '\n';
}

//! @brief Equations 1.28 through 1.36, and Figure 1.13
TEST(chapter1_3, quantum_teleportation)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    auto const psi = qpp::randket().eval();
    auto const alpha = psi[0];
    auto const beta = psi[1];

    auto const psi0 = qpp::kron(psi, qpp::st.b00);
    EXPECT_MATRIX_CLOSE(psi0, inv_sqrt2 * (alpha * (000_ket + 011_ket) + beta * (100_ket + 111_ket)), 1e-12);

    auto const psi1 = qpp::apply(psi0, qpp::gt.CNOT, { 0, 1 });
    EXPECT_MATRIX_CLOSE(psi1, inv_sqrt2 * (alpha * (000_ket + 011_ket) + beta * (110_ket + 101_ket)), 1e-12);

    auto const psi2 = qpp::apply(psi1, qpp::gt.H, { 0 });
    EXPECT_MATRIX_CLOSE(psi2, 0.5 * (alpha * qpp::kron(0_ket + 1_ket, 00_ket + 11_ket)  + beta * qpp::kron(0_ket - 1_ket, 10_ket + 01_ket)), 1e-12);
    EXPECT_MATRIX_CLOSE(psi2, 0.5 * (qpp::kron(00_ket, alpha * 0_ket + beta * 1_ket)
                                   + qpp::kron(01_ket, alpha * 1_ket + beta * 0_ket)
                                   + qpp::kron(10_ket, alpha * 0_ket - beta * 1_ket)
                                   + qpp::kron(11_ket, alpha * 1_ket - beta * 0_ket)), 1e-12);


    auto const [alice_measure_ids, probabilities, psi3] = qpp::measure_seq(psi2, { 0, 1 });
    auto const m1 = alice_measure_ids[0];
    auto const m2 = alice_measure_ids[1];
    auto const alice_measure = qpp::kron(Eigen::Vector2cd::Unit(m1), Eigen::Vector2cd::Unit(m2));

    if (alice_measure == 00_ket)
        EXPECT_MATRIX_CLOSE(psi3, alpha * 0_ket + beta * 1_ket, 1e-12);
    else if (alice_measure == 01_ket)
        EXPECT_MATRIX_CLOSE(psi3, alpha * 1_ket + beta * 0_ket, 1e-12);
    else if (alice_measure == 10_ket)
        EXPECT_MATRIX_CLOSE(psi3, alpha * 0_ket - beta * 1_ket, 1e-12);
    else if (alice_measure == 11_ket)
        EXPECT_MATRIX_CLOSE(psi3, alpha * 1_ket - beta * 0_ket, 1e-12);
    else
        EXPECT_FALSE("Impossible Alice's measure");

    auto const psi4 = (qpp::powm(qpp::gt.Z, m1) * qpp::powm(qpp::gt.X, m2) * psi3).eval();
    EXPECT_MATRIX_CLOSE(psi4, psi, 1e-12);

    debug() << ">> psi:\n" << qpp::disp(psi) << '\n';
    debug() << ">> psi0:\n" << qpp::disp(psi0) << '\n';
    debug() << ">> psi1:\n" << qpp::disp(psi1) << '\n';
    debug() << ">> psi2:\n" << qpp::disp(psi2) << '\n';
    debug() << ">> Alice\'s measure: |" << m1 << m2 << ">\n" << qpp::disp(alice_measure) << '\n';
    debug() << ">> psi3:\n" << qpp::disp(psi3) << '\n';
    debug() << ">> psi4:\n" << qpp::disp(psi4) << '\n';
}

//! @brief Equations 1.28 through 1.36, and Figure 1.13
TEST(chapter1_3, quantum_teleportation_circuit)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    auto const psi = qpp::randket().eval();
    auto const alpha = psi[0];
    auto const beta = psi[1];

    auto const psi_init = qpp::kron(psi, qpp::st.b00);

    auto circuit = qpp::QCircuit{ 3, 2 };
    auto engine = qpp::QEngine{ circuit };
    engine.reset().set_psi(psi_init).execute();
    auto const psi0 = engine.get_psi();
    EXPECT_MATRIX_CLOSE(psi0, inv_sqrt2 * (alpha * (000_ket + 011_ket) + beta * (100_ket + 111_ket)), 1e-12);

    circuit.gate_joint(qpp::gt.CNOT, { 0, 1 });
    engine.reset().set_psi(psi_init).execute();
    auto const psi1 = engine.get_psi();
    EXPECT_MATRIX_CLOSE(psi1, inv_sqrt2 * (alpha * (000_ket + 011_ket) + beta * (110_ket + 101_ket)), 1e-12);

    circuit.gate(qpp::gt.H, 0);
    engine.reset().set_psi(psi_init).execute();
    auto const psi2 = engine.get_psi();
    EXPECT_MATRIX_CLOSE(psi2, 0.5 * (alpha * qpp::kron(0_ket + 1_ket, 00_ket + 11_ket)  + beta * qpp::kron(0_ket - 1_ket, 10_ket + 01_ket)), 1e-12);
    EXPECT_MATRIX_CLOSE(psi2, 0.5 * (qpp::kron(00_ket, alpha * 0_ket + beta * 1_ket)
                                   + qpp::kron(01_ket, alpha * 1_ket + beta * 0_ket)
                                   + qpp::kron(10_ket, alpha * 0_ket - beta * 1_ket)
                                   + qpp::kron(11_ket, alpha * 1_ket - beta * 0_ket)), 1e-12);

    circuit.measureV(qpp::gt.Id2, 0, 0);
    circuit.measureV(qpp::gt.Id2, 1, 1);
    engine.reset().set_psi(psi_init).execute();
    auto const psi3 = engine.get_psi();
    auto const m1 = engine.get_dit(0);
    auto const m2 = engine.get_dit(1);
    auto const alice_measure = qpp::kron(Eigen::Vector2cd::Unit(m1), Eigen::Vector2cd::Unit(m2));

    if (alice_measure == 00_ket)
        EXPECT_MATRIX_CLOSE(psi3, alpha * 0_ket + beta * 1_ket, 1e-12);
    else if (alice_measure == 01_ket)
        EXPECT_MATRIX_CLOSE(psi3, alpha * 1_ket + beta * 0_ket, 1e-12);
    else if (alice_measure == 10_ket)
        EXPECT_MATRIX_CLOSE(psi3, alpha * 0_ket - beta * 1_ket, 1e-12);
    else if (alice_measure == 11_ket)
        EXPECT_MATRIX_CLOSE(psi3, alpha * 1_ket - beta * 0_ket, 1e-12);
    else
        EXPECT_FALSE("Impossible Alice's measure");

    circuit.cCTRL(qpp::gt.X, 1, 2);
    circuit.cCTRL(qpp::gt.Z, 0, 2);
    engine.reset().set_psi(psi_init).execute();
    auto const psi4 = engine.get_psi();
    EXPECT_MATRIX_CLOSE(psi4, psi, 1e-12);

    debug() << circuit << "\n\n" << circuit.get_resources() << "\n\n";
    debug() << engine << "\n\n";
    debug() << ">> psi:\n" << qpp::disp(psi) << '\n';
    debug() << ">> psi0:\n" << qpp::disp(psi0) << '\n';
    debug() << ">> psi1:\n" << qpp::disp(psi1) << '\n';
    debug() << ">> psi2:\n" << qpp::disp(psi2) << '\n';
    debug() << ">> Alice\'s measure: |" << m1 << m2 << ">\n" << qpp::disp(alice_measure) << '\n';
    debug() << ">> psi3:\n" << qpp::disp(psi3) << '\n';
    debug() << ">> psi4:\n" << qpp::disp(psi4) << '\n';
}

//! @brief Equations 1.28 through 1.36, and Figure 1.13
TEST(chapter1_3, quantum_teleportation_circuit_short)
{
    auto const psi = qpp::randket().eval();

    auto circuit = qpp::QCircuit{ 3, 2 };
    circuit.gate_joint(qpp::gt.CNOT, { 0, 1 });
    circuit.gate(qpp::gt.H, 0);
    circuit.measureV(qpp::gt.Id2, 0, 0);
    circuit.measureV(qpp::gt.Id2, 1, 1);
    circuit.cCTRL(qpp::gt.X, 1, 2);
    circuit.cCTRL(qpp::gt.Z, 0, 2);

    auto engine = qpp::QEngine{ circuit };
    engine.set_psi(qpp::kron(psi, qpp::st.b00));
    auto const psi_in = engine.get_psi();
    engine.execute();

    auto const psi_out = engine.get_psi();
    EXPECT_MATRIX_CLOSE(psi_out, psi, 1e-12);

    debug() << circuit << "\n\n" << circuit.get_resources() << "\n\n";
    debug() << engine << "\n\n";
    debug() << ">> psi:\n" << qpp::disp(psi) << '\n';
    debug() << ">> psi_in:\n" << qpp::disp(psi_in) << '\n';
    debug() << ">> psi_out:\n" << qpp::disp(psi_out) << '\n';
}
