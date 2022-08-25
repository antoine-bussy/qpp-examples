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

    //! @brief Extract circuit matrix from circuit
    template < unsigned int Dim >
    auto extract_matrix(qpp::QCircuit const& circuit, unsigned int dim = Dim)
    {
        auto engine = qpp::QEngine{ circuit };
        return extract_matrix<Dim>(engine, dim);
    }
}

//! @brief Exercise 4.16
TEST(chapter4_3, matrix_representation_of_multiqubit_gates)
{
    auto const& H = qpp::gt.H;

    /* Remember rule: (AxB)(CxD) = (AC)x(BD) */

    /* Part 1 */
    auto const IxH = Eigen::Matrix4cd
    {
        { H(0,0), H(0,1),     0.,     0. },
        { H(1,0), H(1,1),     0.,     0. },
        {     0.,     0., H(0,0), H(0,1) },
        {     0.,     0., H(1,0), H(1,1) },
    };

    auto circuit_IxH = qpp::QCircuit{ 2, 0 };
    circuit_IxH.gate_joint(qpp::gt.H, { 1 });
    auto engine_IxH = qpp::QEngine{ circuit_IxH };

    auto const circuit_IxH_matrix = extract_matrix<4>(engine_IxH);
    EXPECT_MATRIX_CLOSE(IxH, circuit_IxH_matrix, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> IxH:\n" << qpp::disp(IxH) << "\n\n";
        std::cerr << ">> circuit_IxH_matrix:\n" << qpp::disp(circuit_IxH_matrix) << "\n\n";
    }

    /* Part 2 */
    auto const HxI = Eigen::Matrix4cd
    {
        { H(0,0),     0., H(0,1),     0. },
        {     0., H(0,0),     0., H(0,1) },
        { H(1,0),     0., H(1,1),     0. },
        {     0., H(1,0),     0., H(1,1) },
    };

    auto circuit_HxI = qpp::QCircuit{ 2, 0 };
    circuit_HxI.gate_joint(qpp::gt.H, { 0 });
    auto engine_HxI = qpp::QEngine{ circuit_HxI };

    auto const circuit_HxI_matrix = extract_matrix<4>(engine_HxI);
    EXPECT_MATRIX_CLOSE(HxI, circuit_HxI_matrix, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> HxI:\n" << qpp::disp(HxI) << "\n\n";
        std::cerr << ">> circuit_HxI_matrix:\n" << qpp::disp(circuit_HxI_matrix) << "\n\n";
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

//! @brief Exercise 4.18
TEST(chapter4_3, controlled_z_flip_invariance)
{
    auto circuit_down = qpp::QCircuit{ 2, 0 };
    circuit_down.CTRL(qpp::gt.Z, { 0 }, { 1 });
    auto engine_down = qpp::QEngine{ circuit_down };
    auto const cZ_down = extract_matrix<4>(engine_down);

    auto circuit_up = qpp::QCircuit{ 2, 0 };
    circuit_up.CTRL(qpp::gt.Z, { 1 }, { 0 });
    auto engine_up = qpp::QEngine{ circuit_up };
    auto const cZ_up = extract_matrix<4>(engine_up);

    EXPECT_MATRIX_CLOSE(cZ_down, cZ_up, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> cZ_down:\n" << qpp::disp(cZ_down) << "\n\n";
        std::cerr << ">> cZ_up:\n" << qpp::disp(cZ_up) << "\n\n";
    }
}

//! @brief Exercise 4.19
TEST(chapter4_3, cnot_action_on_density_matrices)
{
    qpp_e::maths::seed();

    auto const rho = qpp::randrho(4);
    auto const rho_out = qpp::apply(rho, qpp::gt.CNOT, { 0, 1 });

    auto expected_rho = rho;
    expected_rho({0, 1}, {2, 3}).colwise().reverseInPlace();
    expected_rho({2, 3}, {0, 1}).rowwise().reverseInPlace();
    expected_rho({2, 3}, {2, 3}).reverseInPlace();

    if constexpr (print_text)
    {
        std::cerr << ">> rho:\n" << qpp::disp(rho) << "\n\n";
        std::cerr << ">> rho_out:\n" << qpp::disp(rho_out) << "\n\n";
        std::cerr << ">> expected_rho:\n" << qpp::disp(expected_rho) << "\n\n";
    }
}

//! @brief Exercise 4.20 and Equations 4.24 through 4.27
TEST(chapter4_3, cnot_basis_transformations)
{
    using namespace qpp::literals;

    /* Part 1 */
    auto circuit_HxH_cnot_HxH = qpp::QCircuit{ 2, 0 };
    circuit_HxH_cnot_HxH.gate_fan(qpp::gt.H, { 0, 1 });
    circuit_HxH_cnot_HxH.gate(qpp::gt.CNOT, { 0 }, { 1 });
    circuit_HxH_cnot_HxH.gate_fan(qpp::gt.H, { 0, 1 });
    auto engine_HxH_cnot_HxH = qpp::QEngine{ circuit_HxH_cnot_HxH };
    auto const HxH_cnot_HxH = extract_matrix<4>(engine_HxH_cnot_HxH);

    auto circuit_cnot_flipped = qpp::QCircuit{ 2, 0 };
    circuit_cnot_flipped.gate(qpp::gt.CNOT, { 1 }, { 0 });
    auto engine_cnot_flipped = qpp::QEngine{ circuit_cnot_flipped };
    auto const cnot_flipped = extract_matrix<4>(engine_cnot_flipped);

    EXPECT_MATRIX_CLOSE(HxH_cnot_HxH, cnot_flipped, 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::gt.CNOTba, cnot_flipped, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> HxH_cnot_HxH:\n" << qpp::disp(HxH_cnot_HxH) << "\n\n";
        std::cerr << ">> cnot_flipped:\n" << qpp::disp(cnot_flipped) << "\n\n";
        std::cerr << ">> CNOTba (qpp):\n" << qpp::disp(qpp::gt.CNOTba) << "\n\n";
    }

    /* Part 2 */
    auto const HxH = qpp::kronpow(qpp::gt.H, 2);
    auto const& CNOT = qpp::gt.CNOT;
    auto const& CNOTba = qpp::gt.CNOTba;
    auto const pp_ket = qpp::st.plus(2u);
    auto const mp_ket = qpp::kron(qpp::st.minus(), qpp::st.plus());
    auto const pm_ket = qpp::kron(qpp::st.plus(), qpp::st.minus());
    auto const mm_ket = qpp::st.minus(2u);

    /* |00> ==> CNOTba ==> |00> */
    EXPECT_MATRIX_CLOSE((CNOTba * 00_ket).eval(), 00_ket, 1e-12);
    /* |00> ==> HxH ==> |++> ==> CNOT ==> |++> ==> HxH ==> |00> */
    EXPECT_MATRIX_CLOSE((HxH * 00_ket).eval(), pp_ket, 1e-12);
    EXPECT_MATRIX_CLOSE((CNOT * pp_ket).eval(), pp_ket, 1e-12);
    EXPECT_MATRIX_CLOSE((HxH * pp_ket).eval(), 00_ket, 1e-12);

    /* |01> ==> CNOTba ==> |11> */
    EXPECT_MATRIX_CLOSE((CNOTba * 01_ket).eval(), 11_ket, 1e-12);
    /* |01> ==> HxH ==> |+-> ==> CNOT ==> |--> ==> HxH ==> |11> */
    EXPECT_MATRIX_CLOSE((HxH * 01_ket).eval(), pm_ket, 1e-12);
    EXPECT_MATRIX_CLOSE((CNOT * pm_ket).eval(), mm_ket, 1e-12);
    EXPECT_MATRIX_CLOSE((HxH * mm_ket).eval(), 11_ket, 1e-12);

    /* |10> ==> CNOTba ==> |10> */
    EXPECT_MATRIX_CLOSE((CNOTba * 10_ket).eval(), 10_ket, 1e-12);
    /* |10> ==> HxH ==> |-+> ==> CNOT ==> |-+> ==> HxH ==> |10> */
    EXPECT_MATRIX_CLOSE((HxH * 10_ket).eval(), mp_ket, 1e-12);
    EXPECT_MATRIX_CLOSE((CNOT * mp_ket).eval(), mp_ket, 1e-12);
    EXPECT_MATRIX_CLOSE((HxH * mp_ket).eval(), 10_ket, 1e-12);

    /* |11> ==> CNOTba ==> |01> */
    EXPECT_MATRIX_CLOSE((CNOTba * 11_ket).eval(), 01_ket, 1e-12);
    /* |11> ==> HxH ==> |--> ==> CNOT ==> |+-> ==> HxH ==> |01> */
    EXPECT_MATRIX_CLOSE((HxH * 11_ket).eval(), mm_ket, 1e-12);
    EXPECT_MATRIX_CLOSE((CNOT * mm_ket).eval(), pm_ket, 1e-12);
    EXPECT_MATRIX_CLOSE((HxH * pm_ket).eval(), 01_ket, 1e-12);
}

//! @brief Reminder: CNOT = CTRL(X)
TEST(chapter4_3, cnot_is_controlled_x)
{
    auto const controlled_X = qpp::gt.CTRL(qpp::gt.X, { 0 }, { 1 }, 2);
    EXPECT_MATRIX_CLOSE(controlled_X, qpp::gt.CNOT, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> CNOT:\n" << qpp::disp(qpp::gt.CNOT) << "\n\n";
        std::cerr << ">> controlled-X:\n" << qpp::disp(controlled_X) << "\n\n";
    }
}

//! @brief Equation 4.28 and Figure 4.5
TEST(chapter4_3, controlled_phase_shift)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed();

    auto const alpha = qpp::rand(0., 2.*pi);
    auto const exp_ia = std::exp(1.i * alpha);

    auto const one_exp_ia = Eigen::Matrix2cd
    {
        { 1., 0. },
        { 0., exp_ia}
    };
    auto const exp_ia_id = Eigen::Matrix2cd
    {
        { exp_ia, 0. },
        { 0., exp_ia}
    };

    auto const circuit_controlled_exp_ia_id = qpp::QCircuit{ 2, 0 }
        .CTRL(exp_ia_id, { 1 }, { 0 });
    auto const controlled_exp_ia_id = extract_matrix<4>(circuit_controlled_exp_ia_id);

    auto const circuit_one_exp_ia_x_I = qpp::QCircuit{ 2, 0 }
        .gate(one_exp_ia, 1);
    auto const one_exp_ia_x_I = extract_matrix<4>(circuit_one_exp_ia_x_I);

    EXPECT_MATRIX_CLOSE(controlled_exp_ia_id, one_exp_ia_x_I, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> controlled-(exp(ia)I):\n" << qpp::disp(controlled_exp_ia_id) << "\n\n";
        std::cerr << ">> (1.,exp(ia)) x I :\n" << qpp::disp(one_exp_ia_x_I) << "\n\n";
    }
}
