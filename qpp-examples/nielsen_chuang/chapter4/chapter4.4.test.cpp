#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>

#include <chrono>
#include <execution>
#include <numbers>
#include <ranges>

namespace
{
    auto constexpr print_text = false;
}

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

    if constexpr (print_text)
    {
        std::cerr << ">> psi: " << qpp::disp(psi.transpose()) << "\n";
        std::cerr << ">> measured: " << measured << "\n\n";

        std::cerr << ">> out_psi: " << qpp::disp(out_psi.transpose()) << "\n";
        std::cerr << ">> expected_out: " << qpp::disp(expected_out.transpose()) << "\n\n";

        std::cerr << ">> probs: " << qpp::disp(probs.transpose()) << "\n";
        std::cerr << ">> expected_probs: " << qpp::disp(expected_probs.transpose()) << "\n";
    }
}
