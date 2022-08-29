#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>
#include <qpp-examples/qube/decompositions.hpp>

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

    //! @brief Extract indices of non-work qubits in a mket
    template < int NbOfOutputQubits = Eigen::Dynamic >
    auto extract_indices(unsigned int nb_of_qubits, Eigen::VectorX<unsigned int> const& work_qubits_zero = {}
                                                  , Eigen::VectorX<unsigned int> const& work_qubits_one = {})
    {
        auto const nb_of_work_qubits_zero = work_qubits_zero.size();
        auto const nb_of_work_qubits_one = work_qubits_one.size();
        EXPECT_GE(nb_of_qubits, nb_of_work_qubits_zero + nb_of_work_qubits_one);
        auto const nb_of_non_work_qubits = nb_of_qubits - nb_of_work_qubits_zero - nb_of_work_qubits_one;

        if constexpr (NbOfOutputQubits != Eigen::Dynamic)
        {
            EXPECT_EQ(nb_of_non_work_qubits, NbOfOutputQubits);
        }

        auto mask = std::vector<qpp::idx>(nb_of_qubits, 2u);
        for(auto&& i : work_qubits_zero)
            mask[i] = 0u;
        for(auto&& i : work_qubits_one)
            mask[i] = 1u;
        auto const dims = std::vector<qpp::idx>(nb_of_qubits, 2u);

        auto constexpr OuputDim = (NbOfOutputQubits == Eigen::Dynamic)
                                    ? Eigen::Dynamic
                                    : qpp_e::maths::pow(2u, static_cast<unsigned int>(NbOfOutputQubits));
        auto const output_dim = qpp_e::maths::pow(2u, nb_of_non_work_qubits);

        auto indices = Eigen::Vector<unsigned int, OuputDim>::Zero(output_dim).eval();

        auto constexpr fill_indices = [](auto&& callback, auto& mask, auto const& dims, auto& it_indices, auto const i) -> void
        {
            if(i == mask.size())
            {
                if constexpr (print_text)
                {
                    std::cerr << "mask: " << Eigen::VectorX<qpp::idx>::Map(mask.data(), mask.size())
                                .format(Eigen::IOFormat{ Eigen::StreamPrecision, Eigen::DontAlignCols, "", "", "", "", "", "" }) << "\n";
                }
                *it_indices = qpp::multiidx2n(mask, dims);
                ++it_indices;
                return;
            }

            auto& b = mask[i];
            if(b != 2u)
                return callback(callback, mask, dims, it_indices, i+1);

            b = 0u;
            callback(callback, mask, dims, it_indices, i+1);
            b = 1u;
            callback(callback, mask, dims, it_indices, i+1);
            b = 2u;
        };

        auto it_indices = indices.begin();
        fill_indices(fill_indices, mask, dims, it_indices, 0u);
        EXPECT_EQ(it_indices, indices.end());
        std::sort(indices.begin(), indices.end());

        if constexpr (print_text)
        {
            std::cerr << "indices: " << qpp::disp(indices.transpose()) << "\n";
        }

        return indices;
    }

    //! @brief Extract circuit matrix from engine
    template < int Dim = Eigen::Dynamic >
    auto extract_matrix(qpp::QEngine& engine, qpp_e::maths::Matrix auto const& indices)
    {
        auto const total_dim = qpp_e::maths::pow(2u, engine.get_circuit().get_nq());
        auto const dim = indices.size();
        EXPECT_EQ(total_dim % dim, 0);

        auto matrix = Eigen::Matrix<Eigen::dcomplex, Dim, Dim>::Zero(dim, dim).eval();
        auto j = 0u;
        for(auto&& i : indices)
        {
            auto const psi = Eigen::VectorXcd::Unit(total_dim, i);
            engine.reset().set_psi(psi).execute();
            matrix.col(j) = engine.get_psi()(indices, Eigen::all);
            ++j;
        }
        return matrix;
    }

    //! @brief Extract circuit matrix from engine
    template < int Dim = Eigen::Dynamic >
    auto extract_matrix(qpp::QEngine& engine, unsigned int dim = Dim)
    {
        EXPECT_GT(dim, 0);
        return extract_matrix<Dim>(engine, Eigen::Vector<unsigned int, Dim>::LinSpaced(dim, 0u, dim));
    }

    //! @brief Extract circuit matrix from circuit
    template < int Dim = Eigen::Dynamic >
    auto extract_matrix(qpp::QCircuit const& circuit, qpp_e::maths::Matrix auto const& indices)
    {
        auto engine = qpp::QEngine{ circuit };
        return extract_matrix<Dim>(engine, indices);
    }

    //! @brief Extract circuit matrix from circuit
    template < int Dim = Eigen::Dynamic >
    auto extract_matrix(qpp::QCircuit const& circuit, unsigned int dim = Dim)
    {
        EXPECT_NE(dim, static_cast<unsigned int>(Eigen::Dynamic));
        return extract_matrix<Dim>(circuit, Eigen::Vector<unsigned int, Dim>::LinSpaced(dim, 0u, dim));
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
        .CTRL(exp_ia_id, { 0 }, { 1 });
    auto const controlled_exp_ia_id = extract_matrix<4>(circuit_controlled_exp_ia_id);

    auto const circuit_one_exp_ia_x_I = qpp::QCircuit{ 2, 0 }
        .gate(one_exp_ia, 0);
    auto const one_exp_ia_x_I = extract_matrix<4>(circuit_one_exp_ia_x_I);

    EXPECT_MATRIX_CLOSE(controlled_exp_ia_id, one_exp_ia_x_I, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> controlled-(exp(ia)I):\n" << qpp::disp(controlled_exp_ia_id) << "\n\n";
        std::cerr << ">> (1.,exp(ia)) x I :\n" << qpp::disp(one_exp_ia_x_I) << "\n\n";
    }
}

//! @brief Figure 4.6
TEST(chapter4_3, controlled_u_built_circuit)
{
    using namespace std::literals::complex_literals;

    qpp_e::maths::seed();
    auto const U = qpp::randU();

    auto const circuit_controlled_u = qpp::QCircuit{ 2, 0 }
        .CTRL(U, { 0 }, { 1 });
    auto const controlled_U = extract_matrix<4>(circuit_controlled_u);

    auto const [alpha, A, B, C] = qpp_e::qube::abc_decomposition<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_Z, print_text>(U);

    auto const aAXBXC = (std::exp(1.i * alpha) * A * qpp::gt.X * B * qpp::gt.X * C).eval();
    EXPECT_MATRIX_CLOSE(aAXBXC, U, 1e-12);

    auto const one_exp_ia = Eigen::Matrix2cd
    {
        { 1., 0. },
        { 0., std::exp(1.i * alpha)}
    };

    auto const built_circuit = qpp::QCircuit{ 2, 0 }
        .gate(C, 1)
        .CTRL(qpp::gt.X, { 0 }, { 1 })
        .gate(B, 1)
        .CTRL(qpp::gt.X, { 0 }, { 1 })
        .gate(A, 1)
        .gate(one_exp_ia, 0);
    auto const built_controlled_U = extract_matrix<4>(built_circuit);

    EXPECT_MATRIX_CLOSE(built_controlled_U, controlled_U, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> U:\n" << qpp::disp(U) << "\n\n";
        std::cerr << ">> exp(ia) * A * X * B * X * C:\n" << qpp::disp(aAXBXC) << "\n\n";
        std::cerr << ">> built controlled-U:\n" << qpp::disp(built_controlled_U) << "\n\n";
        std::cerr << ">> controlled-U:\n" << qpp::disp(controlled_U) << "\n\n";
    }
}

//! @brief Equation 4.29
TEST(chapter4_3, n_controlled_k_operation)
{
    qpp_e::maths::seed(958u);

    auto constexpr n_controlled_k = []<unsigned int n, unsigned int k, bool ctrl>(auto&& x, auto&& U)
    {
        auto constexpr range_n = std::views::iota(0u, n) | std::views::common;
        auto constexpr _2_pow_k = qpp_e::maths::pow(2u, k);
        auto constexpr range_nk = std::views::iota(n, n+k) | std::views::common;

        EXPECT_EQ(U.rows(), _2_pow_k);
        EXPECT_EQ(U.cols(), _2_pow_k);

        EXPECT_EQ(x.rows(), n);
        EXPECT_EQ(x.cols(), 1);

        auto const x_idx = x.template cast<qpp::idx>().eval();
        auto const x_ket = qpp::mket({ x_idx.cbegin(), x_idx.cend() });
        auto const x_int = std::accumulate(x_idx.cbegin(), x_idx.cend(), 0, [](int a, int b) { return (a << 1) + b; });

        auto const psi = qpp::randket(_2_pow_k);
        auto const n_controlled_U = qpp::gt.CTRL(U
                                            , { range_n.begin(), range_n.end() }
                                            , { range_nk.begin(), range_nk.end() }
                                            , n + k);
        auto const n_controlled_U_x_phi = (n_controlled_U * qpp::kron(x_ket, psi)).eval();
        auto const x_controlled_U_phi = qpp::kron(x_ket, (qpp::powm(U, x.prod()) * psi).eval());

        EXPECT_MATRIX_CLOSE(n_controlled_U_x_phi, x_controlled_U_phi, 1e-12);

        if constexpr (ctrl)
        {
            EXPECT_EQ(x_idx.prod(), 1u);
            auto const x_U_phi = qpp::kron(x_ket, (U * psi).eval());
            EXPECT_MATRIX_CLOSE(n_controlled_U_x_phi, x_U_phi, 1e-12);
        }
        else
        {
            EXPECT_EQ(x_idx.prod(), 0u);
            auto const x_phi = qpp::kron(x_ket, psi);
            EXPECT_MATRIX_CLOSE(n_controlled_U_x_phi, x_phi, 1e-12);
        }

        if constexpr (print_text)
        {
            std::cerr << ">> ctrl: " << std::boolalpha << ctrl << "\n";
            std::cerr << ">> x: " << x_idx.format(Eigen::IOFormat{ Eigen::StreamPrecision, Eigen::DontAlignCols, "", "", "", "", "", "" }) << "\n";
            std::cerr << ">> x int: " << x_int << "\n";
            std::cerr << ">> x product: " << x_idx.prod() << "\n\n";

            std::cerr << ">> n_controlled_U_x_phi: " << qpp::disp(n_controlled_U_x_phi.transpose()) << "\n\n";
            std::cerr << ">> x_controlled_U_phi: " << qpp::disp(x_controlled_U_phi.transpose()) << "\n\n";

            using namespace Eigen::indexing;
            std::cerr << ">> n_controlled_U_x_phi:\n" << qpp::disp(n_controlled_U_x_phi(seqN(_2_pow_k * x_int, _2_pow_k), all).transpose()) << "\n";
            std::cerr << ">> x_controlled_U_phi:\n" << qpp::disp(x_controlled_U_phi(seqN(_2_pow_k * x_int, _2_pow_k), all).transpose()) << "\n\n";
        }
    };

    auto constexpr n = 4u;
    auto constexpr k = 3u;
    auto constexpr _2_pow_k = qpp_e::maths::pow(2u, k);
    auto const U = qpp::randU(_2_pow_k);
    n_controlled_k.template operator()<n, k, false>(Eigen::Vector<bool, n>::Random().eval(), U);
    n_controlled_k.template operator()<n, k, true>(Eigen::Vector<bool, n>::Ones(), U);
}

//! @brief Figure 4.7
TEST(chapter4_3, n_controlled_k_operation_circuit)
{
    qpp_e::maths::seed(95658u);

    auto constexpr n = 4u;
    auto constexpr range_n = std::views::iota(0u, n) | std::views::common;

    auto constexpr k = 3u;
    auto constexpr _2_pow_k = qpp_e::maths::pow(2u, k);
    auto constexpr range_nk = std::views::iota(n, n+k) | std::views::common;

    auto const U = qpp::randU(_2_pow_k);
    auto const psi = qpp::randket(_2_pow_k);

    auto const circuit = qpp::QCircuit{ n + k }
        .CTRL_joint(U, { range_n.begin(), range_n.end() }, { range_nk.begin(), range_nk.end() });

    auto engine = qpp::QEngine{ circuit };

    {
        auto const x = Eigen::Vector<bool, n>::Random().cast<qpp::idx>().eval();
        auto const x_ket = qpp::mket({ x.cbegin(), x.cend() });
        engine.reset().set_psi(qpp::kron(x_ket, psi)).execute();

        auto const expected_out = qpp::kron(x_ket, psi);
        EXPECT_MATRIX_CLOSE(engine.get_psi(), expected_out, 1e-12);
    }
    {
        auto const x = Eigen::Vector<bool, n>::Ones().cast<qpp::idx>().eval();
        auto const x_ket = qpp::mket({ x.cbegin(), x.cend() });
        engine.reset().set_psi(qpp::kron(x_ket, psi)).execute();

        auto const expected_out = qpp::kron(x_ket, (U * psi).eval());
        EXPECT_MATRIX_CLOSE(engine.get_psi(), expected_out, 1e-12);
    }
}

//! @brief Figure 4.8 and Exercise 4.21
TEST(chapter4_3, _2_controlled_1_U)
{
    using namespace std::complex_literals;

    qpp_e::maths::seed();

    auto constexpr _2_controlled_1_U_circuit = [](auto&& U, auto&& V)
    {
        EXPECT_MATRIX_CLOSE((V * V).eval(), U, 1e-12);
        EXPECT_MATRIX_CLOSE((V * V.adjoint()).eval(), Eigen::Matrix2cd::Identity(), 1e-12);
        EXPECT_MATRIX_CLOSE((V.adjoint() * V).eval(), Eigen::Matrix2cd::Identity(), 1e-12);

        auto const circuit_U = qpp::QCircuit{ 3u }
            .CTRL(U, { 0, 1 }, { 2 });
        auto const controlled_U = extract_matrix<8>(circuit_U);

        auto const& X = qpp::gt.X;

        auto const built_circuit_U = qpp::QCircuit{ 3u }
            .CTRL(V, { 1 }, { 2 })
            .CTRL(X, { 0 }, { 1 })
            .CTRL(V.adjoint(), { 1 }, { 2 })
            .CTRL(X, { 0 }, { 1 })
            .CTRL(V, { 0 }, { 2 });
        auto const built_controlled_U = extract_matrix<8>(built_circuit_U);

        EXPECT_MATRIX_CLOSE(built_controlled_U, controlled_U, 1e-12);

        if constexpr (print_text)
        {
            std::cerr << ">> controlled_U:\n" << qpp::disp(controlled_U) << "\n\n";
            std::cerr << ">> built_controlled_U:\n" << qpp::disp(built_controlled_U) << "\n\n";
        }

        return built_controlled_U;
    };
    {
        auto const U = qpp::randU();
        auto const V = qpp::sqrtm(U);
        _2_controlled_1_U_circuit(U, V);
    }
    {
        auto const& U = qpp::gt.X;
        auto const V = (0.5 * (1. - 1.i) * (qpp::gt.Id2 + 1.i * qpp::gt.X)).eval();
        auto const built_controlled_U = _2_controlled_1_U_circuit(U, V);
        EXPECT_MATRIX_CLOSE(built_controlled_U, qpp::gt.TOF, 1e-12);
    }
}

//! @brief Check that diagonal operators and CNOT can be switched
//! @details It applies in particular to phase gates
TEST(chapter4_3, diagonal_gate_and_cnot)
{
    qpp_e::maths::seed();

    auto const D = Eigen::Vector2cd::Random().asDiagonal().toDenseMatrix().eval();
    auto const& I = qpp::gt.Id2;
    auto const& CNOT = qpp::gt.CNOT;

    auto const DxI = qpp::kron(D, I);

    auto const expected_DxI = Eigen::Matrix4cd
    {
        { D(0,0), 0., 0., 0.},
        { 0., D(0,0), 0., 0.},
        { 0., 0., D(1,1), 0.},
        { 0., 0., 0., D(1,1)},
    };
    EXPECT_MATRIX_CLOSE(DxI, expected_DxI, 1e-12);

    auto const DxI_CNOT = (DxI * CNOT).eval();
    auto const CNOT_DxI = (CNOT * DxI).eval();
    EXPECT_MATRIX_CLOSE(DxI_CNOT, CNOT_DxI, 1e-12);

    auto const expected_DxI_CNOT = Eigen::Matrix4cd
    {
        { D(0,0), 0., 0., 0.},
        { 0., D(0,0), 0., 0.},
        { 0., 0., 0., D(1,1)},
        { 0., 0., D(1,1), 0.},
    };
    EXPECT_MATRIX_CLOSE(DxI_CNOT, expected_DxI_CNOT, 1e-12);

    auto const& X = qpp::gt.X;
    auto const circuit_DxI_CNOT = qpp::QCircuit{ 2u }
        .CTRL(X, { 0 }, { 1 })
        .gate(D, 0);
    auto const built_DxI_CNOT = extract_matrix<4>(circuit_DxI_CNOT);
    EXPECT_MATRIX_CLOSE(built_DxI_CNOT, DxI_CNOT, 1e-12);

    auto const circuit_CNOT_DxI = qpp::QCircuit{ 2u }
        .gate(D, 0)
        .CTRL(X, { 0 }, { 1 });
    auto const built_CNOT_DxI = extract_matrix<4>(circuit_CNOT_DxI);
    EXPECT_MATRIX_CLOSE(built_CNOT_DxI, CNOT_DxI, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> D:\n" << qpp::disp(D) << "\n\n";
        std::cerr << ">> I:\n" << qpp::disp(I) << "\n\n";
        std::cerr << ">> CNOT:\n" << qpp::disp(CNOT) << "\n\n";
        std::cerr << ">> DxI:\n" << qpp::disp(DxI) << "\n\n";
        std::cerr << ">> DxI_CNOT:\n" << qpp::disp(DxI_CNOT) << "\n\n";
        std::cerr << ">> CNOT_DxI:\n" << qpp::disp(CNOT_DxI) << "\n\n";
        std::cerr << ">> built_DxI_CNOT:\n" << qpp::disp(built_DxI_CNOT) << "\n\n";
        std::cerr << ">> built_CNOT_DxI:\n" << qpp::disp(built_CNOT_DxI) << "\n\n";
    }
}

//! @brief Exercise 4.22
TEST(chapter4_3, simplified_2_controlled_1_U)
{
    using namespace std::complex_literals;

    qpp_e::maths::seed();

    auto const U = qpp::randU();

    auto const circuit_U = qpp::QCircuit{ 3u }
        .CTRL(U, { 0, 1 }, { 2 });
    auto const controlled_U = extract_matrix<8>(circuit_U);

    auto const V = qpp::sqrtm(U);
    EXPECT_MATRIX_CLOSE((V * V).eval(), U, 1e-12);
    EXPECT_MATRIX_CLOSE((V * V.adjoint()).eval(), Eigen::Matrix2cd::Identity(), 1e-12);
    EXPECT_MATRIX_CLOSE((V.adjoint() * V).eval(), Eigen::Matrix2cd::Identity(), 1e-12);

    auto const [alpha, A, B, C] = qpp_e::qube::abc_decomposition<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_Z, print_text>(V);
    auto const exp_ia = Eigen::Matrix2cd
    {
        { 1., 0. },
        { 0., std::exp(1.i * alpha) },
    };
    auto const& X = qpp::gt.X;

    auto const built_circuit_U = qpp::QCircuit{ 3u }
        .gate(C, 2)
        .CTRL(X, { 1 }, { 2 })
        .gate(B, 2)
        .CTRL(X, { 0 }, { 2 })
        .gate(B.adjoint(), 2)
        .CTRL(X, { 1 }, { 2 })
        .gate(B, 2)
        .CTRL(X, { 0 }, { 2 })
        .gate(A, 2)

        .gate(exp_ia, 0)
        .gate(exp_ia, 1)
        .CTRL(X, { 0 }, { 1 })
        .gate(exp_ia.adjoint(), 1)
        .CTRL(X, { 0 }, { 1 })
    ;
    auto const built_controlled_U = extract_matrix<8>(built_circuit_U);

    EXPECT_MATRIX_CLOSE(built_controlled_U, controlled_U, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> controlled_U:\n" << qpp::disp(controlled_U) << "\n\n";
        std::cerr << ">> built_controlled_U:\n" << qpp::disp(built_controlled_U) << "\n\n";
    }
}

//! @brief Exercise 4.23, Part 1
TEST(chapter4_3, controlled_rx_circuit)
{
    using namespace std::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed();
    auto const theta = qpp::rand(0., 2.*pi);

    auto const U = qpp::gt.RX(theta);
    auto const circuit_U = qpp::QCircuit{ 2u }
        .CTRL(U, { 0 }, { 1 });
    auto const controlled_U = extract_matrix<4>(circuit_U);

    /* Force special values for A, B, C */
    auto const A = (qpp::gt.RZ(-0.5 * pi) * qpp::gt.RY(0.5 * theta)).eval();
    auto const B = qpp::gt.RY(-0.5 * theta);
    auto const C = qpp::gt.RZ(0.5 * pi);

    auto const& X = qpp::gt.X;

    auto const ABC = (A * B * C).eval();
    EXPECT_MATRIX_CLOSE(ABC, Eigen::Matrix2cd::Identity(), 1e-12);
    auto const AXBXC = (A * X * B * X * C).eval();
    EXPECT_MATRIX_CLOSE(AXBXC, U, 1e-12);

    auto const built_circuit_U = qpp::QCircuit{ 2u }
        .gate(C, 1)
        .CTRL(X, { 0 }, { 1 })
        .gate(B, 1)
        .CTRL(X, { 0 }, { 1 })
        .gate(A, 1)
    ;
    auto const built_controlled_U = extract_matrix<4>(built_circuit_U);
    EXPECT_MATRIX_CLOSE(built_controlled_U, controlled_U, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> controlled_U:\n" << qpp::disp(controlled_U) << "\n\n";
        std::cerr << ">> built_controlled_U:\n" << qpp::disp(built_controlled_U) << "\n\n";
    }
}

//! @brief Exercise 4.23, Part 2
TEST(chapter4_3, controlled_ry_circuit)
{
    using namespace std::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed(3141592u);
    auto const theta = qpp::rand(0., 2.*pi);

    auto const U = qpp::gt.RY(theta);
    auto const circuit_U = qpp::QCircuit{ 2u }
        .CTRL(U, { 0 }, { 1 });
    auto const controlled_U = extract_matrix<4>(circuit_U);

    auto [alpha, A, B, C] = qpp_e::qube::abc_decomposition<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_Z, print_text>(U);

    /* RY is exactly at the singularity of the euler system */
    ASSERT_LT(std::abs(alpha), 1e-12);

    EXPECT_MATRIX_CLOSE(C, Eigen::Matrix2cd::Identity(), 1e-12);

    auto const& X = qpp::gt.X;

    auto const built_circuit_U = qpp::QCircuit{ 2u }
        .CTRL(X, { 0 }, { 1 })
        .gate(B, 1)
        .CTRL(X, { 0 }, { 1 })
        .gate(A, 1)
    ;
    auto const built_controlled_U = extract_matrix<4>(built_circuit_U);
    EXPECT_MATRIX_CLOSE(built_controlled_U, controlled_U, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> controlled_U:\n" << qpp::disp(controlled_U) << "\n\n";
        std::cerr << ">> built_controlled_U:\n" << qpp::disp(built_controlled_U) << "\n\n";
    }
}

//! @brief Figure 4.8 and Exercise 4.24
TEST(chapter4_3, abc_decomposition_of_v_adjoint)
{
    using namespace std::complex_literals;
    using namespace std::numbers;

    auto const& I = qpp::gt.Id2;
    auto const& X = qpp::gt.X;
    auto const& H = qpp::gt.H;
    auto const& T = qpp::gt.T;

    auto const V = (0.5 * (1.-1.i) * (I + 1.i * X)).eval();

    auto const alpha = 0.25 * pi;
    auto const A = (H * T).eval();
    auto const B = T.adjoint();
    auto const& C = H;

    auto const eiaAXBXC = (std::exp(1.i * alpha) * A * X * B * X * C).eval();
    EXPECT_MATRIX_CLOSE(eiaAXBXC, V.adjoint(), 1e-12);
    auto const ABC = (A * B * C).eval();
    EXPECT_MATRIX_CLOSE(ABC, I, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> alpha: " << qpp::disp(alpha) << "\n";
        std::cerr << ">> A:\n" << qpp::disp(A) << "\n";
        std::cerr << ">> B:\n" << qpp::disp(B) << "\n";
        std::cerr << ">> C:\n" << qpp::disp(C) << "\n";

        std::cerr << ">> ABC:\n" << qpp::disp(ABC) << "\n\n";
        std::cerr << ">> eiaAXBXC:\n" << qpp::disp(eiaAXBXC) << "\n\n";
        std::cerr << ">> V.adjoint():\n" << qpp::disp(V.adjoint()) << "\n\n";
    }
}

//! @brief Figure 4.8 and Exercise 4.24
TEST(chapter4_3, toffoli_circuit)
{
    auto const& X = qpp::gt.X;
    auto const& H = qpp::gt.H;
    auto const& T = qpp::gt.T;
    auto const& S = qpp::gt.S;

    auto const circuit_toffoli = qpp::QCircuit{ 3, 0 }
        .gate(qpp::gt.TOF, 0, 1, 2);
    auto const toffoli = extract_matrix<8>(circuit_toffoli);
    EXPECT_MATRIX_CLOSE(toffoli, qpp::gt.TOF, 1e-12);

    auto const built_circuit_toffoli = qpp::QCircuit{ 3, 0 }
        .gate(H, 2)
        .CTRL(X, { 1 }, { 2 })
        .gate(T.adjoint(), 2)
        .CTRL(X, { 0 }, { 2 })
        .gate(T, 2)
        .CTRL(X, { 1 }, { 2 })
        .gate(T.adjoint(), 2)
        .CTRL(X, { 0 }, { 2 })
        .gate(T, 2)
        .gate(H, 2)

        .gate(T.adjoint(), 1)
        .CTRL(X, { 0 }, { 1 })
        .gate(T.adjoint(), 1)
        .CTRL(X, { 0 }, { 1 })
        .gate(S, 1)

        .gate(T, 0)
    ;
    auto const built_toffoli = extract_matrix<8>(built_circuit_toffoli);
    EXPECT_MATRIX_CLOSE(built_toffoli, toffoli, 1e-12);

    auto const built_circuit_toffoli_b = qpp::QCircuit{ 3, 0 }
        .gate(H, 2)
        .CTRL(X, { 1 }, { 2 })
        .gate(T.adjoint(), 2)
        .CTRL(X, { 0 }, { 2 })
        .gate(T, 2)
        .CTRL(X, { 1 }, { 2 })
        .gate(T.adjoint(), 2)
        .CTRL(X, { 0 }, { 2 })
        .gate(T, 2)
        .gate(H, 2)

        .gate(T, 1)
        .CTRL(X, { 0 }, { 1 })
        .gate(T.adjoint(), 1)
        .CTRL(X, { 0 }, { 1 })

        .gate(T, 0)
    ;
    auto const built_toffoli_b = extract_matrix<8>(built_circuit_toffoli_b);
    EXPECT_MATRIX_CLOSE(built_toffoli_b, toffoli, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> toffoli (qpp):\n" << qpp::disp(qpp::gt.TOF) << "\n\n";
        std::cerr << ">> toffoli:\n" << qpp::disp(toffoli) << "\n\n";
        std::cerr << ">> built_toffoli:\n" << qpp::disp(built_toffoli) << "\n\n";
        std::cerr << ">> built_toffoli_b:\n" << qpp::disp(built_toffoli_b) << "\n\n";
    }
}

//! @brief Exercise 4.25
TEST(chapter4_3, fredkin_circuit)
{
    using namespace Eigen::indexing;
    using namespace std::complex_literals;

    auto const& I = qpp::gt.Id2;
    auto const& X = qpp::gt.X;
    auto const& TOF = qpp::gt.TOF;

    auto fredkin_matrix = Eigen::Matrix<Eigen::dcomplex, 8, 8>::Identity().eval();
    fredkin_matrix({5,6}, {5,6}) = X;
    EXPECT_MATRIX_EQ(fredkin_matrix, qpp::gt.FRED);

    auto const circuit_fredkin = qpp::QCircuit{ 3, 0 }
        .gate(qpp::gt.FRED, 0, 1, 2);
    auto const fredkin = extract_matrix<8>(circuit_fredkin);
    EXPECT_MATRIX_CLOSE(fredkin, qpp::gt.FRED, 1e-12);

    /* Part 1 */
    auto const circuit_fredkin_as_tof = qpp::QCircuit{ 3, 0 }
        .gate(TOF, 0, 2, 1)
        .gate(TOF, 0, 1, 2)
        .gate(TOF, 0, 2, 1)
        ;
    auto const fredkin_as_tof = extract_matrix<8>(circuit_fredkin_as_tof);
    EXPECT_MATRIX_CLOSE(fredkin_as_tof, fredkin, 1e-12);

    /* Part 2 */
    auto const circuit_fredkin_as_tof_and_cnot = qpp::QCircuit{ 3, 0 }
        .CTRL(X, 2, 1)
        .gate(TOF, 0, 1, 2)
        .CTRL(X, 2, 1)
        ;
    auto const fredkin_as_tof_and_cnot = extract_matrix<8>(circuit_fredkin_as_tof_and_cnot);
    EXPECT_MATRIX_CLOSE(fredkin_as_tof_and_cnot, fredkin, 1e-12);

    /* Part 3 */
    auto const V = (0.5 * (1.-1.i) * (I + 1.i * X)).eval();
    auto const A = (qpp::gt.CTRL(V, {0}, {1}, 2) * qpp::gt.CNOTba).eval();

    auto const circuit_fredkin_6_gates = qpp::QCircuit{ 3, 0 }
        .gate(A, 1, 2)
        .CTRL(X, 0, 1)
        .CTRL(V.adjoint(), 1, 2)
        .CTRL(X, 0, 1)
        .CTRL(V, 0, 2)
        .CTRL(X, 2, 1)
        ;
    auto const fredkin_6_gates = extract_matrix<8>(circuit_fredkin_6_gates);
    EXPECT_MATRIX_CLOSE(fredkin_6_gates, fredkin, 1e-12);

    /* Part 4 */
    auto const B = (qpp::gt.CNOTba * qpp::gt.CTRL(V.adjoint(), {0}, {1}, 2)).eval();
    EXPECT_MATRIX_CLOSE(B, A.adjoint(), 1e-12);

    auto const circuit_fredkin_5_gates = qpp::QCircuit{ 3, 0 }
        .gate(A, 1, 2)
        .CTRL(V, 0, 2)
        .CTRL(X, 0, 1)
        .gate(B, 1, 2)
        .CTRL(X, 0, 1)
        ;
    auto const fredkin_5_gates = extract_matrix<8>(circuit_fredkin_5_gates);
    EXPECT_MATRIX_CLOSE(fredkin_5_gates, fredkin, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> fredkin:\n" << qpp::disp(fredkin) << "\n\n";
        std::cerr << ">> fredkin_as_tof:\n" << qpp::disp(fredkin_as_tof) << "\n\n";
        std::cerr << ">> fredkin_as_tof_and_cnot:\n" << qpp::disp(fredkin_as_tof_and_cnot) << "\n\n";
        std::cerr << ">> fredkin_6_gates:\n" << qpp::disp(fredkin_6_gates) << "\n\n";
        std::cerr << ">> fredkin_5_gates:\n" << qpp::disp(fredkin_5_gates) << "\n\n";
    }
}

//! @brief Check that U, V commute => ctrl-U, ctrl-V commute
TEST(chapter4_3, ctrl_commute)
{
    qpp_e::maths::seed();

    auto const P = qpp::randU();
    EXPECT_MATRIX_CLOSE((P * P.adjoint()).eval(), Eigen::Matrix2cd::Identity(), 1e-12);
    auto const U = (P * Eigen::Vector2cd::Random().asDiagonal().toDenseMatrix() * P.adjoint()).eval();
    auto const V = (P * Eigen::Vector2cd::Random().asDiagonal().toDenseMatrix() * P.adjoint()).eval();

    auto const UV = (U * V).eval();
    auto const VU = (V * U).eval();
    EXPECT_MATRIX_CLOSE(UV, VU, 1e-12);

    auto const circuit_UV = qpp::QCircuit{ 3, 0 }
        .CTRL(V, 0, 2)
        .CTRL(U, 1, 2)
        ;
    auto const ctrl_UV = extract_matrix<8>(circuit_UV);

    auto const circuit_VU = qpp::QCircuit{ 3, 0 }
        .CTRL(U, 1, 2)
        .CTRL(V, 0, 2)
        ;
    auto const ctrl_VU = extract_matrix<8>(circuit_VU);

    EXPECT_MATRIX_CLOSE(ctrl_UV, ctrl_VU, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> UV:\n" << qpp::disp(UV) << "\n\n";
        std::cerr << ">> VU:\n" << qpp::disp(VU) << "\n\n";
        std::cerr << ">> ctrl_UV:\n" << qpp::disp(ctrl_UV) << "\n\n";
        std::cerr << ">> ctrl_VU:\n" << qpp::disp(ctrl_VU) << "\n\n";
    }
}

//! @brief Exercise 4.26
TEST(chapter4_3, toffoli_up_to_phase)
{
    using namespace std::numbers;

    auto const theta = 0.25 * pi;
    auto const Ry = qpp::gt.RY(theta);
    auto const& X = qpp::gt.X;

    auto const toffoli_up_to_phase_circuit = qpp::QCircuit{ 3, 0 }
        .gate(Ry, 2)
        .CTRL(X, 1, 2)
        .gate(Ry, 2)
        .CTRL(X, 0, 2)
        .gate(Ry.adjoint(), 2)
        .CTRL(X, 1, 2)
        .gate(Ry.adjoint(), 2)
        ;

    auto constexpr relative_phase = [](auto&& c1, auto&& c2, auto&& t)
    {
        if (c1 == 1
            && c2 == 0
            && t == 1
        )
            return -1.;

        return 1.;
    };

    auto relative_phase_vector = Eigen::Vector<Eigen::dcomplex, 8>::Zero().eval();
    auto engine = qpp::QEngine{ toffoli_up_to_phase_circuit };

    for (auto&& c1 : { 0u, 1u })
        for (auto&& c2 : { 0u, 1u })
            for (auto&& t : { 0u, 1u })
            {
                auto const ket_in = qpp::mket({ c1, c2, t });
                engine.reset().set_psi(ket_in).execute();
                auto const expected_ket_out = engine.get_psi();

                auto const phase = relative_phase(c1, c2, t);
                auto const ket_out = (phase * qpp::gt.TOF * ket_in).eval();
                EXPECT_MATRIX_CLOSE(ket_out, expected_ket_out, 1e-12);

                auto const i = (c1 << 2) + (c2 << 1) + t;
                EXPECT_MATRIX_EQ(ket_in, (Eigen::Vector<Eigen::dcomplex, 8>::Unit(i)));
                relative_phase_vector[i] = phase;

                if constexpr (print_text)
                {
                    std::cerr << ">> | c1, c2, t >: " << c1 << c2 << t << "\n";
                    std::cerr << ">> ket_in:           " << qpp::disp(ket_in.transpose()) << "\n";
                    std::cerr << ">> ket_out:          " << qpp::disp(ket_out.transpose()) << "\n";
                    std::cerr << ">> expected_ket_out: " << qpp::disp(expected_ket_out.transpose()) << "\n\n";
                }
            }


    auto const toffoli_up_to_phase = extract_matrix<8>(engine);
    auto const expected_toffoli_up_to_phase = (relative_phase_vector.asDiagonal().toDenseMatrix() * qpp::gt.TOF).eval();
    EXPECT_MATRIX_CLOSE(toffoli_up_to_phase, expected_toffoli_up_to_phase, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> toffolli:\n" << qpp::disp(qpp::gt.TOF) << "\n\n";
        std::cerr << ">> toffoli_up_to_phase:\n" << qpp::disp(toffoli_up_to_phase) << "\n\n";
        std::cerr << ">> expected_toffoli_up_to_phase:\n" << qpp::disp(expected_toffoli_up_to_phase) << "\n\n";
    }
}

//! @brief Exercise 4.27 and Equation 4.31
TEST(chapter4_3, permutation_circuit)
{
    auto const& X = qpp::gt.X;
    auto const& TOF = qpp::gt.TOF;

    auto const circuit = qpp::QCircuit{ 3, 0 }
        .gate(TOF, 1, 2, 0)
        .CTRL(X, 2, 1)
        .CTRL(X, 0, 2)
        .CTRL(X, 1, 2)
        .gate(TOF, 0, 1, 2)
        ;
    auto const perm = extract_matrix<8>(circuit);

    auto const expected_perm = Eigen::Matrix<Eigen::dcomplex, 8, 8>
    {
        { 1., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 1. },
        { 0., 1., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 1., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 1., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 1., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 1., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 1., 0. },
    };
    EXPECT_MATRIX_CLOSE(perm, expected_perm, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> perm:\n" << qpp::disp(perm) << "\n\n";
        std::cerr << ">> expected_perm:\n" << qpp::disp(expected_perm) << "\n\n";
    }
}

//! @brief Figure 4.10
TEST(chapter4_3, n_controlled_U)
{
    qpp_e::maths::seed();

    auto constexpr n = 4u;
    ASSERT_GE(n, 2u);
    auto constexpr _2_pow_n_1 = qpp_e::maths::pow(2u, n + 1u);
    auto const U = qpp::randU();
    auto constexpr ctrl = std::views::iota(0u, n) | std::views::common;

    auto const expected_circuit = qpp::QCircuit{ n + 1 }
        .CTRL(U, { ctrl.begin(), ctrl.end() }, n)
        ;
    auto const expected_ctrl_U = extract_matrix<_2_pow_n_1>(expected_circuit);

    auto const& TOF = qpp::gt.TOF;
    auto circuit = qpp::QCircuit{ 2u * n };

    circuit.gate(TOF, 0u, 1u, n);
    for (auto&& i : std::views::iota(2u, n))
        circuit.gate(TOF, i, i + n - 2u, i +  n - 1u);

    circuit.CTRL(U, 2u * n - 2u, 2u * n - 1u);

    for (auto&& i : std::views::iota(2u, n) | std::views::reverse)
        circuit.gate(TOF, i, i + n - 2u, i +  n - 1u);
    circuit.gate(TOF, 0u, 1u, n);

    auto const work_qubits_zero = Eigen::VectorX<unsigned int>::LinSpaced(n - 1u, n, 2u * n - 2u).eval();
    auto const indices = extract_indices<n + 1u>(2u * n, work_qubits_zero);
    auto const ctrl_U = extract_matrix<_2_pow_n_1>(circuit, indices);

    EXPECT_MATRIX_CLOSE(ctrl_U, expected_ctrl_U, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> indices:\n" << qpp::disp(indices.transpose()) << "\n";
        std::cerr << ">> work_qubits_zero:\n" << qpp::disp(work_qubits_zero.transpose()) << "\n";
        std::cerr << ">> ctrl_U:\n" << qpp::disp(ctrl_U) << "\n\n";
        std::cerr << ">> expected_ctrl_U:\n" << qpp::disp(expected_ctrl_U) << "\n\n";
    }
}
