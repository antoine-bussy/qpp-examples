#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>

#include <numbers>
#include <ranges>

namespace
{
    auto constexpr print_text = false;

    auto complicated_circuit()
    {
        auto circuit = qpp::QCircuit{ 3 };

        circuit.gate(qpp::gt.H, 0);
        circuit.gate_joint(qpp::gt.CNOT, { 1, 2 });

        circuit.gate(qpp::gt.H, 1);

        circuit.gate_joint(qpp::gt.CNOT, { 0, 1 });

        circuit.gate(qpp::gt.H, 1);
        circuit.gate(qpp::gt.H, 2);

        circuit.gate_joint(qpp::gt.CNOT, { 0, 2 });

        circuit.gate_joint(qpp::gt.CNOT, { 1, 0 });
        circuit.gate(qpp::gt.H, 2);

        circuit.gate(qpp::gt.H, 1);

        circuit.gate_joint(qpp::gt.CNOT, { 2, 1 });

        circuit.gate_joint(qpp::gt.CNOT, { 0, 2 });

        circuit.gate(qpp::gt.H, 0);
        circuit.gate(qpp::gt.H, 2);

        circuit.gate_joint(qpp::gt.CNOT, { 1, 0 });

        circuit.gate_joint(qpp::gt.CNOT, { 2, 1 });

        circuit.gate(qpp::gt.H, 1);

        circuit.gate_joint(qpp::gt.CNOT, { 1, 0 });

        circuit.gate(qpp::gt.H, 0);
        circuit.gate(qpp::gt.H, 1);
        circuit.gate(qpp::gt.H, 2);

        circuit.gate_joint(qpp::gt.CNOT, { 0, 1 });

        circuit.gate_joint(qpp::gt.CNOT, { 2, 0 });

        circuit.gate(qpp::gt.H, 0);
        circuit.gate(qpp::gt.H, 1);

        circuit.gate_joint(qpp::gt.CNOT, { 2, 1 });

        return circuit;
    }

    auto simplified_circuit()
    {
        auto circuit = qpp::QCircuit{ 3 };

        circuit.gate(qpp::gt.H, 1);

        circuit.gate_joint(qpp::gt.CNOT, { 1, 0 });

        circuit.gate(qpp::gt.H, 0);
        circuit.gate(qpp::gt.RZ(1.5 * std::numbers::pi), 1);
        circuit.gate(qpp::gt.RZ(1.5 * std::numbers::pi), 2);

        circuit.gate_joint(qpp::gt.CNOT, { 2, 1 });

        circuit.gate(qpp::gt.RZ(0.5 * std::numbers::pi), 1);
        circuit.gate(qpp::gt.RZ(std::numbers::pi), 2);

        circuit.gate(qpp::gt.RX(std::numbers::pi), 1);

        circuit.gate_joint(qpp::gt.CNOT, { 2, 0 });

        circuit.gate(qpp::gt.RZ(std::numbers::pi), 0);

        circuit.gate_joint(qpp::gt.CNOT, { 0, 1 });
        circuit.gate_joint(qpp::gt.CNOT, { 1, 0 });
        circuit.gate_joint(qpp::gt.CNOT, { 0, 1 });

        return circuit;
    }

}

//! @brief Adapted from Slide 21, with the help of https://zxcalculus.com/
TEST(perdrix, circuit_simplification)
{
    auto const psi = qpp::randket(8).eval();

    auto const circuit = complicated_circuit();
    auto engine = qpp::QEngine{ circuit };
    engine.set_psi(psi).execute();

    auto const circuit_simple = simplified_circuit();
    auto engine_simple = qpp::QEngine{ circuit_simple };
    engine_simple.set_psi(psi).execute();

    EXPECT_COLLINEAR(engine.get_psi(), engine_simple.get_psi(), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << "Complicated circuit:\n" << circuit << "\n\n" << circuit.get_resources() << "\n\n";
        std::cerr << "Engine for complicated circuit:\n" << engine << "\n\n";
        std::cerr << "Simplified circuit:\n" << circuit_simple << "\n\n" << circuit_simple.get_resources() << "\n\n";
        std::cerr << "Engine for simplified circuit:\n" << engine_simple << "\n\n";
        std::cerr << ">> psi:\n" << qpp::disp(psi) << '\n';
        std::cerr << ">> psi_out complicated:\n" << qpp::disp(engine.get_psi()) << '\n';
        std::cerr << ">> psi_out simplified:\n" << qpp::disp(engine_simple.get_psi()) << '\n';
    }
}

//! @brief Adapted from Slide 21, with the help of https://zxcalculus.com/
TEST(perdrix, circuit_simplification_adjoint)
{
    using namespace std::complex_literals;

    auto const psi = qpp::randket(8).eval();

    auto const circuit = complicated_circuit().add_circuit(simplified_circuit().adjoint(), 0);
    auto engine = qpp::QEngine{ circuit };
    engine.set_psi(psi).execute();

    EXPECT_COLLINEAR(engine.get_psi(), psi, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << "Circuit:\n" << circuit << "\n\n" << circuit.get_resources() << "\n\n";
        std::cerr << "Engine for circuit:\n" << engine << "\n\n";
        std::cerr << ">> psi:\n" << qpp::disp(psi) << '\n';
        std::cerr << ">> psi_out:\n" << qpp::disp(engine.get_psi()) << '\n';
    }

    auto M = Eigen::MatrixXcd::Zero(8, 8).eval();
    for(auto&& i : std::views::iota(0, 8))
    {
        auto const state = Eigen::Vector<std::complex<double>,8>::Unit(i).eval();
        engine.set_psi(state).execute();
        M.col(i) = engine.get_psi();
    }
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;
    EXPECT_TRUE((inv_sqrt2 * (-1. + 1i) * M).isIdentity(1e-12));
    if constexpr (print_text)
        std::cerr << "Circuit Matrix:\n" << qpp::disp(M) << '\n';
}
