#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>

#include <execution>
#include <numbers>
#include <ranges>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Equation 4.23
TEST(chapter4_3, cnot_circuit)
{
    using namespace qpp::literals;
    qpp_e::maths::seed();

    auto const cnot = Eigen::Matrix4cd
    {
        { 1., 0., 0., 0. },
        { 0., 1., 0., 0. },
        { 0., 0., 0., 1. },
        { 0., 0., 1., 0. },
    };
    EXPECT_MATRIX_EQ(cnot, qpp::gt.CNOT);

    auto circuit = qpp::QCircuit{ 2, 0 };
    circuit.gate_joint(qpp::gt.CNOT, { 0, 1 });
    auto engine = qpp::QEngine{ circuit };

    /* Random |c>|t> input */
    auto const c = qpp::randket();
    auto const t = qpp::randket();

    auto const in = qpp::kron(c,t);
    engine.reset().set_psi(in).execute();
    auto const out = engine.get_psi();

    auto const expected_out = (cnot * in).eval();
    EXPECT_MATRIX_CLOSE(out, expected_out, 1e-12);

    /* |0>|t> input */
    engine.reset().set_psi(qpp::kron(0_ket,t)).execute();
    EXPECT_MATRIX_CLOSE(engine.get_psi(), qpp::kron(0_ket,t), 1e-12);

    /* |1>|t> input */
    engine.reset().set_psi(qpp::kron(1_ket,t)).execute();
    EXPECT_MATRIX_CLOSE(engine.get_psi(), qpp::kron(1_ket,t.reverse()), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> Circuit:\n" << circuit << "\n\n" << circuit.get_resources() << "\n\n";
        std::cerr << ">> Engine:\n" << engine << "\n\n";
        std::cerr << ">> c: " << qpp::disp(c.transpose()) << "\n";
        std::cerr << ">> t: " << qpp::disp(t.transpose()) << "\n";
        std::cerr << ">> out: " << qpp::disp(out.transpose()) << "\n";
        std::cerr << ">> expected_out: " << qpp::disp(expected_out.transpose()) << "\n";
    }
}

//! @brief Figure 4.4
TEST(chapter4_3, controlled_u)
{
    using namespace qpp::literals;

    qpp_e::maths::seed();

    auto const U = qpp::randU();

    auto const cU = Eigen::Matrix4cd
    {
        { 1., 0.,     0.,     0. },
        { 0., 1.,     0.,     0. },
        { 0., 0., U(0,0), U(0,1) },
        { 0., 0., U(1,0), U(1,1) },
    };

    auto const controlled_U = qpp::gt.CTRL(U, { 0 }, { 1 }, 2);
    EXPECT_MATRIX_EQ(cU, controlled_U);

    auto circuit = qpp::QCircuit{ 2, 0 };
    circuit.gate_joint(controlled_U, { 0, 1 });
    auto engine = qpp::QEngine{ circuit };

    /* Random |c>|t> input */
    auto const c = qpp::randket();
    auto const t = qpp::randket();

    auto const in = qpp::kron(c,t);
    engine.reset().set_psi(in).execute();
    auto const out = engine.get_psi();

    auto const expected_out = (cU * in).eval();
    EXPECT_MATRIX_CLOSE(out, expected_out, 1e-12);

    /* |0>|t> input */
    engine.reset().set_psi(qpp::kron(0_ket,t)).execute();
    EXPECT_MATRIX_CLOSE(engine.get_psi(), qpp::kron(0_ket,t), 1e-12);

    /* |1>|t> input */
    engine.reset().set_psi(qpp::kron(1_ket,t)).execute();
    EXPECT_MATRIX_CLOSE(engine.get_psi(), qpp::kron(1_ket, (U*t).eval()), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> Circuit:\n" << circuit << "\n\n" << circuit.get_resources() << "\n\n";
        std::cerr << ">> Engine:\n" << engine << "\n\n";
        std::cerr << ">> c: " << qpp::disp(c.transpose()) << "\n";
        std::cerr << ">> t: " << qpp::disp(t.transpose()) << "\n";
        std::cerr << ">> out: " << qpp::disp(out.transpose()) << "\n";
        std::cerr << ">> expected_out: " << qpp::disp(expected_out.transpose()) << "\n";
    }
}

namespace
{
    //! @brief Extract circuit matrix from engine
    template < unsigned int Dim >
    auto extract_matrix(qpp::QEngine& engine, unsigned int dim = Dim)
    {
        auto matrix = Eigen::Matrix<Eigen::dcomplex, Dim, Dim>::Zero(dim, dim).eval();
        for(auto&& i : std::views::iota(0u, dim))
        {
            engine.reset().set_psi(Eigen::Vector<Eigen::dcomplex, Dim>::Unit(dim, i)).execute();
            matrix.col(i) = engine.get_psi();
        }
        return matrix;
    }
}

//! @brief Exercise 4.16
TEST(chapter4_3, matrix_representation_of_multiqubit_gates)
{
    auto const& H = qpp::gt.H;

    /* Remember rule: (AxB)(CxD) = (AC)x(BD) */

    /* Part 1 */
    auto const H_up = Eigen::Matrix4cd
    {
        { H(0,0), H(0,1),     0.,     0. },
        { H(1,0), H(1,1),     0.,     0. },
        {     0.,     0., H(0,0), H(0,1) },
        {     0.,     0., H(1,0), H(1,1) },
    };

    auto circuit_H_up = qpp::QCircuit{ 2, 0 };
    circuit_H_up.gate_joint(qpp::gt.H, { 1 });
    auto engine_H_up = qpp::QEngine{ circuit_H_up };

    auto const circuit_H_up_matrix = extract_matrix<4>(engine_H_up);
    EXPECT_MATRIX_CLOSE(H_up, circuit_H_up_matrix, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> H_up:\n" << qpp::disp(H_up) << "\n\n";
        std::cerr << ">> circuit_H_up_matrix:\n" << qpp::disp(circuit_H_up_matrix) << "\n\n";
    }

    /* Part 2 */
    auto const H_down = Eigen::Matrix4cd
    {
        { H(0,0),     0., H(0,1),     0. },
        {     0., H(0,0),     0., H(0,1) },
        { H(1,0),     0., H(1,1),     0. },
        {     0., H(1,0),     0., H(1,1) },
    };

    auto circuit_H_down = qpp::QCircuit{ 2, 0 };
    circuit_H_down.gate_joint(qpp::gt.H, { 0 });
    auto engine_H_down = qpp::QEngine{ circuit_H_down };

    auto const circuit_H_down_matrix = extract_matrix<4>(engine_H_down);
    EXPECT_MATRIX_CLOSE(H_down, circuit_H_down_matrix, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> H_down:\n" << qpp::disp(H_down) << "\n\n";
        std::cerr << ">> circuit_H_down_matrix:\n" << qpp::disp(circuit_H_down_matrix) << "\n\n";
    }
}

//! @brief Exercise 4.17
TEST(chapter4_3, cnot_as_controlled_z_and_h)
{
    auto circuit = qpp::QCircuit{ 2, 0 };
    circuit.gate(qpp::gt.H, 1);
    circuit.CTRL(qpp::gt.Z, { 0 }, { 1 });
    circuit.gate(qpp::gt.H, 1);
    auto engine = qpp::QEngine{ circuit };

    auto const cnot = extract_matrix<4>(engine);
    EXPECT_MATRIX_CLOSE(cnot, qpp::gt.CNOT, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> cnot:\n" << qpp::disp(cnot) << "\n\n";
        std::cerr << ">> qpp::gt.CNOT:\n" << qpp::disp(qpp::gt.CNOT) << "\n\n";
    }
}
