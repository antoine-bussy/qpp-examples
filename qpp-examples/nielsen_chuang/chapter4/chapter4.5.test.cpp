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

namespace
{
    struct two_level_matrix_t
    {
        Eigen::Index index = 0;
        Eigen::Matrix2cd U = Eigen::Matrix2cd::Identity();
    };

    auto two_level_unitary_decomposition(qpp_e::maths::Matrix auto const& U, std::vector<two_level_matrix_t>& out, Eigen::Index const offset = 0) -> void
    {
        using namespace Eigen::indexing;

        debug() << ">> U:\n" << qpp::disp(U) << "\n\n";

        auto const n = U.cols();
        assert(n >= 2);

        if(n == 2)
        {
            out.emplace_back(offset, U);
            return;
        }

        auto const range = std::views::iota(0, n-1) | std::views::common;

        auto const I2 = Eigen::Matrix2cd::Identity();
        auto const I = Eigen::MatrixXcd::Identity(n, n);
        EXPECT_MATRIX_CLOSE((U.adjoint()*U).eval(), I, 1.e-12);
        EXPECT_MATRIX_CLOSE((U*U.adjoint()).eval(), I, 1.e-12);

        auto new_U = U.eval();
        for(auto&& i : range | std::views::reverse)
        {
            auto const& b = new_U(i+1, 0);
            if (b == 0.)
                continue;
            auto const& a = new_U(i, 0);

            auto s = two_level_matrix_t{};
            s.index = i + offset;
            s.U.col(0) = Eigen::Vector2cd{ a, b };
            s.U.col(1) = Eigen::Vector2cd{ -std::conj(b), std::conj(a) };
            s.U /= std::sqrt(std::norm(a) + std::norm(b));
            EXPECT_MATRIX_CLOSE((s.U.adjoint()*s.U).eval(), I2, 1.e-12);
            EXPECT_MATRIX_CLOSE((s.U*s.U.adjoint()).eval(), I2, 1.e-12);

            out.emplace_back(s);
            new_U({i, i+1}, all) = s.U.adjoint() * new_U({i, i+1}, all);
        }
        debug() << ">> new_U:\n" << qpp::disp(new_U) << "\n\n";

        return two_level_unitary_decomposition(new_U(lastN(n-1), lastN(n-1)), out, offset + 1);
    }

    auto two_level_unitary_decomposition(qpp_e::maths::Matrix auto const& U)
    {
        auto result = std::vector<two_level_matrix_t>{};
        two_level_unitary_decomposition(U, result);
        return result;
    }
}

//! @brief Equations 4.41 through 4.51
TEST(chapter4_5, two_level_unitary_decomposition)
{
    qpp_e::maths::seed();

    auto constexpr n = 8;
    auto const U = qpp::randU(n);

    auto const u = two_level_unitary_decomposition(U);

    auto constexpr multiply = [](qpp_e::maths::Matrix auto const& M, two_level_matrix_t const& s)
    {
        auto P = Eigen::MatrixXcd::Identity(n, n).eval();
        auto const& i = s.index;
        P({i, i+1}, {i, i+1}) = s.U;

        debug() << ">> P:\n" << qpp::disp(P) << "\n\n";

        return (M * P).eval();
    };

    auto const computed_U = std::accumulate(u.cbegin(), u.cend(), Eigen::MatrixXcd::Identity(n, n).eval(), multiply);

    EXPECT_MATRIX_CLOSE(computed_U, U, 1.e-12);

    debug() << ">> computed_U:\n" << qpp::disp(computed_U) << "\n\n";
    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
}
