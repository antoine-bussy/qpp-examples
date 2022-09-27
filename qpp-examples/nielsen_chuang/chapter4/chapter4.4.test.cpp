#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>
#include <qpp-examples/qube/debug.hpp>

#include <chrono>
#include <execution>
#include <numbers>
#include <ranges>

using namespace qpp_e::qube::stream;

//! @brief Figure 4.14
TEST(chapter4_4, projective_measurment_circuit)
{
    using namespace qpp::literals;
    qpp_e::maths::seed();

    auto const psi = qpp::randket();

    auto const circuit = qpp::QCircuit{ 1u, 1u }
        .measureZ(0u, 0u)
        ;
    auto engine = qpp::QEngine{ circuit };
    engine.set_psi(psi).execute();

    auto const out_psi = engine.get_psi();
    auto const measured = engine.get_dit(0u);

    auto const p = engine.get_probs();
    auto const probs = Eigen::VectorXd::Map(p.data(), p.size());

    auto const expected_out = psi.row(measured).normalized().eval();
    auto const expected_probs = psi.cwiseAbs2().eval();

    EXPECT_MATRIX_CLOSE(out_psi, expected_out, 1e-12);
    EXPECT_COMPLEX_CLOSE(probs[0], expected_probs[measured], 1e-12);

    debug() << ">> psi: " << qpp::disp(psi.transpose()) << "\n";
    debug() << ">> measured: " << measured << "\n\n";

    debug() << ">> out_psi: " << qpp::disp(out_psi.transpose()) << "\n";
    debug() << ">> expected_out: " << qpp::disp(expected_out.transpose()) << "\n\n";

    debug() << ">> probs: " << qpp::disp(probs.transpose()) << "\n";
    debug() << ">> expected_probs: " << qpp::disp(expected_probs.transpose()) << "\n";
}

//! @brief Figure 4.15 (and Figure 1.13)
TEST(chapter4_4, quantum_teleportation_with_deferred_measurement)
{
    qpp_e::maths::seed();

    auto const& b00 = qpp::st.b00;

    auto const& X = qpp::gt.X;
    auto const& Z = qpp::gt.Z;
    auto const& H = qpp::gt.H;

    auto const circuit = qpp::QCircuit{ 3, 2 }
        .CTRL(X, 0, 1)
        .gate(H, 0)
        .measureZ(0, 0)
        .measureZ(1, 1)
        .cCTRL(X, 1, 2)
        .cCTRL(Z, 0, 2)
    ;

    auto const deferred_circuit = qpp::QCircuit{ 3, 2 }
        .CTRL(X, 0, 1)
        .gate(H, 0)
        .CTRL(X, 1, 2)
        .CTRL(Z, 0, 2)
        .measureZ(0, 0)
        .measureZ(1, 1)
    ;

    auto const psi = qpp::randket();

    auto engine = qpp::QEngine{ circuit };
    engine.reset().set_psi(qpp::kron(psi, b00)).execute();
    EXPECT_MATRIX_CLOSE(engine.get_psi(), psi, 1e-12);

    auto deferred_engine = qpp::QEngine{ deferred_circuit };
    deferred_engine.reset().set_psi(qpp::kron(psi, b00)).execute();
    EXPECT_MATRIX_CLOSE(deferred_engine.get_psi(), psi, 1e-12);

    auto const d = engine.get_dits();
    auto const dits = Eigen::VectorX<long unsigned int>::Map(d.data(), d.size());
    auto const dd = deferred_engine.get_dits();
    auto const deferred_dits = Eigen::VectorX<long unsigned int>::Map(dd.data(), dd.size());

    debug() << ">> psi:\n" << qpp::disp(psi) << '\n';
    debug() << ">> psi_out:\n" << qpp::disp(engine.get_psi()) << '\n';
    debug() << ">> deferred psi_out:\n" << qpp::disp(deferred_engine.get_psi()) << '\n';
    debug() << ">> dits:\n" << qpp::disp(dits) << '\n';
    debug() << ">> deferred dits:\n" << qpp::disp(deferred_dits) << '\n';
}

//! @brief Exercise 4.32 (with formulas)
TEST(chapter4_4, implicit_measurement_formulas)
{
    using namespace qpp::literals;

    qpp_e::maths::seed();

    auto const& I = qpp::gt.Id2;

    auto const P = std::vector<qpp::cmat>{ 0_prj, 1_prj };
    auto const P_tilde = std::vector<qpp::cmat>{ qpp::kron(I, P[0]), qpp::kron(I, P[1]) };

    auto const rho = qpp::randrho(4);

    auto const [result, p, rho_out] = qpp::measure(rho, P, { 1 }, { 2 }, false);

    auto const rho_prime = (p[0] * rho_out[0] + p[1] * rho_out[1]).eval();
    auto const expectel_rho_prime = (P_tilde[0] * rho * P_tilde[0] + P_tilde[1] * rho * P_tilde[1]).eval();

    EXPECT_MATRIX_CLOSE(rho_prime, expectel_rho_prime, 1e-12);

    debug() << ">> rho:\n" << qpp::disp(rho) << '\n';
    debug() << ">> rho':\n" << qpp::disp(rho_prime) << '\n';
    debug() << ">> expected rho':\n" << qpp::disp(expectel_rho_prime) << '\n';
    debug() << ">> Probabilities: ";
    debug() << qpp::disp(p, ", ") << '\n';
    debug() << ">> Resulting states:\n";
    for (auto&& r : rho_out)
        debug() << qpp::disp(r) << "\n\n";
}

//! @brief Exercise 4.32 (with circuit)
TEST(chapter4_4, implicit_measurement_circuit)
{
    qpp_e::maths::seed();

    auto const U = qpp::randU(4);
    auto circuit = qpp::QCircuit{ 2, 2 }
        .gate(U, 0, 1)
        .measureZ(0, 0)
        ;
    auto engine = qpp::QEngine{ circuit };

    auto const psi = qpp::randket(4);

    engine.reset().set_psi(psi).execute();
    auto const p1 = engine.get_probs();
    auto const probs1 = Eigen::VectorXd::Map(p1.data(), p1.size());
    auto const dits1 = engine.get_dits();

    debug() << ">> probs 1: " << qpp::disp(probs1.transpose()) << "\n";
    debug() << ">> dits 1: " << qpp::disp(dits1, ", ") << '\n';

    circuit.measureZ(1, 1);

    engine.reset().set_psi(psi).execute();
    auto const p2 = engine.get_probs();
    auto const probs2 = Eigen::VectorXd::Map(p2.data(), p2.size());
    auto const dits2 = engine.get_dits();

    debug() << ">> probs 2: " << qpp::disp(probs2.transpose()) << "\n";
    debug() << ">> dits 2: " << qpp::disp(dits2, ", ") << '\n';

    if(dits1[0] == dits2[0])
    {
        EXPECT_COMPLEX_CLOSE(probs2[0], probs1[0], 1e-12);
    }
    else
    {
        EXPECT_COMPLEX_CLOSE(probs2[0], 1. - probs1[0], 1e-12);
    }
}

//! @brief Exercise 4.33
TEST(chapter4_4, measurement_in_the_bell_basis)
{
    using namespace qpp::literals;

    qpp_e::maths::seed(459);

    auto const psi_bell = Eigen::Vector4cd::Random().normalized().eval();
    auto bell_basis = Eigen::Matrix4cd{};
    bell_basis.col(0) = qpp::st.b00;
    bell_basis.col(1) = qpp::st.b01;
    bell_basis.col(2) = qpp::st.b10;
    bell_basis.col(3) = qpp::st.b11;
    auto const psi = (bell_basis * psi_bell).eval();

    auto const circuit = qpp::QCircuit{ 2 }
        .CTRL(qpp::gt.X, 0, 1)
        .gate(qpp::gt.H, 0)
    ;
    auto engine = qpp::QEngine{ circuit };
    engine.set_psi(psi).execute();

    auto const psi_out = engine.get_psi();

    EXPECT_MATRIX_CLOSE(psi_out, psi_bell, 1e-12);

    auto const [result, p, pm_states] = qpp::measure(psi_out, { 00_prj, 01_prj, 10_prj, 11_prj }, { 0, 1 }, 2, false);
    auto const probs = Eigen::Vector4d::Map(p.data());
    auto const [result_bell, p_bell, pm_states_bell] = qpp::measure(psi, { qpp::st.pb00, qpp::st.pb01, qpp::st.pb10, qpp::st.pb11 }, { 0, 1 }, 2, false);
    auto const probs_bell = Eigen::Vector4d::Map(p_bell.data());

    /* Probabilities are equal, but post-measurement states aren't -> POVM
    * This also means that measurement operators are different. */
    EXPECT_MATRIX_CLOSE(probs_bell, probs, 1e-12);
    for (auto&& i : { 0, 1, 2, 3 })
    {
        EXPECT_MATRIX_NOT_CLOSE(pm_states_bell[i], pm_states[i], 1e-1);
    }

    debug() << ">> psi_bell:\n" << qpp::disp(psi_bell) << "\n";
    debug() << ">> psi:\n" << qpp::disp(psi) << "\n";
    debug() << ">> psi_out:\n" << qpp::disp(psi_out) << "\n\n";
    debug() << ">> probs: " << qpp::disp(probs.transpose()) << "\n";
    debug() << ">> probs_bell: " << qpp::disp(probs_bell.transpose()) << "\n\n";
    debug() << ">> Resulting states :\n";
    for (auto&& s : pm_states)
        debug() << qpp::disp(s.transpose()) << "\n";
    debug() << ">> Resulting states bell :\n";
    for (auto&& s : pm_states_bell)
        debug() << qpp::disp(s.transpose()) << "\n";
}
