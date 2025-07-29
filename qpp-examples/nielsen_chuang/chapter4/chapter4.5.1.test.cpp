#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/gtest_macros.hpp>
#include <qube/maths/random.hpp>
#include <qube/debug.hpp>
#include <qube/decompositions.hpp>

#include <chrono>
#include <execution>
#include <numbers>
#include <unordered_set>
#include <ranges>

using namespace qube::stream;

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

    auto constexpr range = std::views::iota(0ul, 4ul) | std::views::common;

    for(auto&& x : range)
    {
        for(auto&& y : range)
        {
            auto const xy = 4*x + y;
            auto const xy_bin = qpp::n2multiidx(xy, { 2, 2, 2, 2 });
            auto const psi = qpp::mket(xy_bin);

            engine.reset().set_state(psi).execute();

            auto const psi_out = engine.get_state();
            auto const xy_bin_out = *qpp::zket2dits(psi_out);
            auto const x_out = 2*xy_bin_out[0] + xy_bin_out[1];
            auto const y_out = 2*xy_bin_out[2] + xy_bin_out[3];

            EXPECT_EQ(x_out, x);
            EXPECT_EQ(y_out, (x + y) % 4);

            debug() << ">> x: " << x << "\n";
            debug() << ">> y: " << y << "\n";
            debug() << ">> xy_bin: " << qpp::disp(xy_bin, { "" }) << "\n";
            debug() << ">> x_out: " << x_out << "\n";
            debug() << ">> y_out: " << y_out << "\n";
            debug() << ">> xy_bin_out: " << qpp::disp(xy_bin_out, { "" }) << "\n\n";
        }
    }
}

//! @brief Equations 4.41 through 4.51
TEST(chapter4_5, two_level_unitary_decomposition)
{
    qube::maths::seed();

    auto constexpr n = 8;
    auto const U = qpp::randU(n);

    auto const u = qube::two_level_unitary_decomposition(U);
    auto const computed_U = std::accumulate(u.cbegin(), u.cend(), Eigen::MatrixXcd::Identity(n, n).eval(), std::multiplies<>());

    EXPECT_MATRIX_CLOSE(computed_U, U, 1.e-12);

    debug() << ">> computed_U:\n" << qpp::disp(computed_U) << "\n\n";
    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
}

//! @brief Exercise 4.37 and Equation 4.52
TEST(chapter4_5, two_level_unitary_decomposition_example)
{
    using namespace std::complex_literals;

    auto const U = (0.5 * Eigen::Matrix4cd
    {
        { 1., 1. , 1., 1. },
        { 1., 1.i,-1.,-1.i},
        { 1.,-1. , 1.,-1. },
        { 1.,-1.i,-1., 1.i},

    }).eval();

    auto const u = qube::two_level_unitary_decomposition(U);
    auto const computed_U = std::accumulate(u.cbegin(), u.cend(), Eigen::Matrix4cd::Identity().eval(), std::multiplies<>());

    EXPECT_MATRIX_CLOSE(computed_U, U, 1.e-12);

    debug() << ">> computed_U:\n" << qpp::disp(computed_U) << "\n\n";
    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
}

//! @brief Exercise 4.38
TEST(chapter4_5, minimal_two_level_unitary_decomposition)
{
    qube::maths::seed();

    auto constexpr d = 7u;

    /* Build (d-2) pairs of indices (i,j), where 0 <= i,j < d and i != j */
    auto index_pairs = Eigen::ArrayX2<unsigned int>::Zero(d-2, 2).eval();
    index_pairs.col(0) = (0.5 * d * (1. + Eigen::ArrayXd::Random(d-2))).cast<unsigned int>();
    index_pairs.col(1) = (0.5 * (d-1) * (1. + Eigen::ArrayXd::Random(d-2))).cast<unsigned int>();
    index_pairs.col(1) = (index_pairs.col(0) + index_pairs.col(1) + 1).unaryExpr([](auto&& x) { return x % d; });

    /* For each i, register j when (i,j) is a pair */
    auto index_pair_map = std::vector<std::unordered_set<unsigned int>>(d);
    for (auto&& p : index_pairs.rowwise())
    {
        index_pair_map[p[0]].emplace(p[1]);
        index_pair_map[p[1]].emplace(p[0]);
    }

    /* Build a partition r, t of [0, d[. t is the set of indices "connected" to index_pairs(0,0) by a pair.
    Then r = [0, d[ \ t */
    auto t = std::unordered_set<unsigned int>{};
    auto constexpr range = std::views::iota(0u, d) | std::views::common;
    auto r = std::unordered_set(range.begin(), range.end());

    auto queue = std::unordered_set{ index_pairs(0,0) };
    while (!queue.empty())
    {
        auto const n = *queue.begin();
        queue.erase(queue.begin());
        queue.merge(index_pair_map[n]);
        index_pair_map[n].clear();
        t.emplace(n);
        r.erase(n);
    }

    auto const T = std::vector(t.cbegin(), t.cend());
    auto const R = std::vector(r.cbegin(), r.cend());

    debug() << ">> index pairs:\n" << qpp::disp(index_pairs.matrix().transpose()) << "\n\n";
    debug() << ">> r:\n" << qpp::disp(R, {", "}) << "\n\n";
    debug() << ">> t:\n" << qpp::disp(T, {", "}) << "\n\n";

    /* Build U as the product of (d-2) two-level matrices, the two-level being determined by an index pair */
    auto U = Eigen::MatrixXcd::Identity(d,d).eval();
    auto u = Eigen::MatrixXcd::Identity(d,d).eval();
    for (auto&& p : index_pairs.rowwise())
    {
        u(p,p) = qpp::randU();
        U = U * u;
        u(p,p).setIdentity();
    }

    /* r is never empty, i.e. (d-2) pairs always fail to "connect" all of [0,d[ */
    EXPECT_FALSE(r.empty());
    /* t and r defines a partition of the vectors of the computational basis,
    and the two vectorial spaces E_t and E_r spanned by them: E = E_t + E_r.
    The result, which is examplified here, is that E_t and E_r are stable under the
    action of U. In other words, the "blocks" U(t,r) and U(r,t) are zero. */
    EXPECT_TRUE(U(R,T).isZero(1e-12));
    EXPECT_TRUE(U(T,R).isZero(1e-12));

    /* t and r not being empty, any matrix built as a product of (d-2) two-level matrices
    has two non-empty zero blocks. Conversely, any matrix without any zero coefficient
    cannot be decomposed as a product of less than (d-1) two-level matrices */

    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
    debug() << ">> U(r,t):\n" << qpp::disp(U(R,T)) << "\n\n";
    debug() << ">> U(t,r):\n" << qpp::disp(U(T,R)) << "\n\n";
}
