#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>
#include <qpp-examples/qube/debug.hpp>
#include <qpp-examples/qube/decompositions.hpp>

#include <chrono>
#include <execution>
#include <numbers>
#include <ranges>

using namespace qpp_e::qube::stream;

//! @brief Exercise 4.36
TEST(chapter4_5, addition_mod_4)
{
    using namespace qpp::literals;

    auto const& X = qpp::gt.X;

    auto const circuit = qpp::QCircuit{ 4u }
        .CTRL(X, 0, 2)
        .CTRL(X, {1,3}, 2)
        .CTRL(X, 1, 3)
        ;
    auto engine = qpp::QEngine{ circuit };

    auto constexpr range = std::views::iota(0, 4) | std::views::common;

    for(auto&& x : range)
    {
        for(auto&& y : range)
        {
            auto const xy = 4*x + y;
            auto const xy_bin = qpp::n2multiidx(xy, { 2, 2, 2, 2 });
            auto const psi = qpp::mket(xy_bin);

            engine.reset().set_psi(psi).execute();

            auto const psi_out = engine.get_psi();
            auto const xy_bin_out = qpp::zket2dits(psi_out);
            auto const x_out = 2*xy_bin_out[0] + xy_bin_out[1];
            auto const y_out = 2*xy_bin_out[2] + xy_bin_out[3];

            EXPECT_EQ(x_out, x);
            EXPECT_EQ(y_out, (x + y) % 4);

            debug() << ">> x: " << x << "\n";
            debug() << ">> y: " << y << "\n";
            debug() << ">> xy_bin: " << qpp::disp(xy_bin, "") << "\n";
            debug() << ">> x_out: " << x_out << "\n";
            debug() << ">> y_out: " << y_out << "\n";
            debug() << ">> xy_bin_out: " << qpp::disp(xy_bin_out, "") << "\n\n";
        }
    }
}

//! @brief Equations 4.41 through 4.51
TEST(chapter4_5, two_level_unitary_decomposition)
{
    qpp_e::maths::seed();

    auto constexpr n = 8;
    auto const U = qpp::randU(n);

    auto const u = qpp_e::qube::two_level_unitary_decomposition(U);
    auto const computed_U = std::accumulate(u.cbegin(), u.cend(), Eigen::MatrixXcd::Identity(n, n).eval(), std::multiplies<>());

    EXPECT_MATRIX_CLOSE(computed_U, U, 1.e-12);

    debug() << ">> computed_U:\n" << qpp::disp(computed_U) << "\n\n";
    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
}
