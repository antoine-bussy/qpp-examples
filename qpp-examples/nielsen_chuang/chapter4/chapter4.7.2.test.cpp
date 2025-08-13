#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.hpp>
#include <qube/debug.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/gtest_macros.hpp>
#include <qube/maths/random.hpp>

#include <ranges>

using namespace qube::stream;

//! @brief Exercise 4.47
TEST(chapter4_7, exponential_commutation)
{
    using namespace std::complex_literals;
    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr N = qube::maths::pow(2ul, nq);

    auto const H1 = qpp::randH(N);

    // To generate random commuting Hermitian matrices, we can use the fact that
    // they can be diagonalized simultaneously in a common basis, and that they
    // have real eigenvalues.
    auto const solver = Eigen::SelfAdjointEigenSolver<qpp::cmat>{ H1 };
    EXPECT_EQ(solver.info(), Eigen::Success);

    auto const H1_recomputed = (solver.eigenvectors() * solver.eigenvalues().asDiagonal() * solver.eigenvectors().adjoint()).eval();
    EXPECT_MATRIX_CLOSE(H1, H1_recomputed, 1e-12);

    auto const H2 = (solver.eigenvectors() * qpp::rand<qpp::rmat>(N, 1, -10., 10.).asDiagonal() * solver.eigenvectors().adjoint()).eval();
    EXPECT_MATRIX_CLOSE(H2, H2.adjoint(), 1e-12);
    EXPECT_MATRIX_CLOSE(H1 * H2, H2 * H1, 1e-12);

    auto const H3 = (solver.eigenvectors() * qpp::rand<qpp::rmat>(N, 1, -10., 10.).asDiagonal() * solver.eigenvectors().adjoint()).eval();
    EXPECT_MATRIX_CLOSE(H3, H3.adjoint(), 1e-12);
    EXPECT_MATRIX_CLOSE(H1 * H3, H3 * H1, 1e-12);
    EXPECT_MATRIX_CLOSE(H2 * H3, H3 * H2, 1e-12);

    auto const H = (H1 + H2 + H3).eval();
    EXPECT_MATRIX_CLOSE(H, H.adjoint(), 1e-12);

    auto const U_expected = qpp::expm(-1.i * H);
    auto const U = (qpp::expm(-1.i * H1) * qpp::expm(-1.i * H2) * qpp::expm(-1.i * H3)).eval();
    EXPECT_MATRIX_CLOSE(U, U_expected, 1e-12);
}

//! @brief Theorem 4.3 and Equations 4.98 through 4.102
TEST(chapter4_7, trotter_formula)
{
    using namespace std::complex_literals;
    qube::maths::seed(1755073394);

    auto constexpr nq = 3ul;
    auto constexpr N = qube::maths::pow(2ul, nq);
    auto constexpr n = 100ul;
    auto constexpr n_ = static_cast<double>(n);
    auto const t = qpp::rand(0.1, 1.);

    auto const A = qpp::randH(N);
    auto const B = qpp::randH(N);

    auto const expA_n = qpp::expm(1.i * t * A / n_);
    auto const expA_n_approx = (Eigen::MatrixXcd::Identity(N, N) + 1.i * t / n_ * A).eval();
    EXPECT_MATRIX_CLOSE(expA_n, expA_n_approx, 1. / qube::maths::pow(n_, 2));

    auto const expB_n = qpp::expm(1.i * t * B / n_);
    auto const expB_n_approx = (Eigen::MatrixXcd::Identity(N, N) + 1.i * t / n_ * B).eval();
    EXPECT_MATRIX_CLOSE(expB_n, expB_n_approx, 1. / qube::maths::pow(n_, 2));

    auto const eq_4_100 = (expA_n * expB_n).eval();
    auto const eq_4_100_approx = (Eigen::MatrixXcd::Identity(N, N) + 1.i * t / n_ * (A + B)).eval();
    EXPECT_MATRIX_CLOSE(eq_4_100, eq_4_100_approx, 1. / qube::maths::pow(n_, 2));

    auto const eq_4_101 = qpp::powm(eq_4_100, n);
    auto eq_4_101_approx = Eigen::MatrixXcd::Identity(N, N).eval();

    auto n_choose_k = 1.;
    auto nk_inv = 1.;
    auto term = Eigen::MatrixXcd::Identity(N, N).eval();
    auto const i_A_plus_Bt = (1.i * t * (A + B)).eval();
    for(auto&& k: std::views::iota(1ul, n + 1ul))
    {
        n_choose_k *= (n_ - k + 1.) / k;
        nk_inv /= n_;
        term = term * i_A_plus_Bt;
        eq_4_101_approx.noalias() = eq_4_101_approx + n_choose_k * nk_inv * term;
    }
    EXPECT_MATRIX_CLOSE(eq_4_101, eq_4_101_approx, 1. / n_);

    term.setIdentity(N, N);
    auto eq_4_102_approx = Eigen::MatrixXcd::Identity(N, N).eval();
    auto inv_k_factorial = 1.;
    for(auto&& k: std::views::iota(1ul, n + 1ul))
    {
        inv_k_factorial /= k;
        term = term * i_A_plus_Bt;
        eq_4_102_approx.noalias() = eq_4_102_approx + inv_k_factorial * term;
    }
    EXPECT_MATRIX_CLOSE(eq_4_101, eq_4_102_approx, 1. / n_);
    auto const exp_A_plus_Bt = qpp::expm(1.i * t * (A + B));
    EXPECT_MATRIX_CLOSE(eq_4_102_approx, exp_A_plus_Bt, 1. / n_);
}

//! @brief Exercise 4.49 and Equations 4.103 through 4.105
TEST(chapter4_7, bch_formula)
{
    qube::maths::seed(1755076482);

    auto constexpr nq = 3ul;
    auto constexpr N = qube::maths::pow(2ul, nq);
    auto const Dt = 0.001;

    auto const A = qpp::randH(N);
    auto const B = qpp::randH(N);

    auto const expA_Dt = qpp::expm(A * Dt);
    auto const expB_Dt = qpp::expm(B * Dt);

    auto const expA_plus_B_Dt = qpp::expm((A + B) * Dt);
    auto const eq_4_103_approx = (expA_Dt * expB_Dt).eval();
    EXPECT_MATRIX_CLOSE(expA_plus_B_Dt, eq_4_103_approx, 10. * qube::maths::pow(Dt, 2));

    debug() << ">> expA_plus_B_Dt:\n" << qpp::disp(expA_plus_B_Dt) << "\n\n";
    debug() << ">> eq_4_103_approx:\n" << qpp::disp(eq_4_103_approx) << "\n\n";

    auto const exp_halfA_Dt = qpp::expm(0.5 * A * Dt);
    auto const eq_4_104_approx = (exp_halfA_Dt * expB_Dt * exp_halfA_Dt).eval();
    EXPECT_MATRIX_CLOSE(expA_plus_B_Dt, eq_4_104_approx, 10. * qube::maths::pow(Dt, 3));

    debug() << ">> expA_plus_B_Dt:\n" << qpp::disp(expA_plus_B_Dt) << "\n\n";
    debug() << ">> eq_4_104_approx:\n" << qpp::disp(eq_4_104_approx) << "\n\n";

    auto const exp_commutator_Dt = qpp::expm(-0.5 * (A * B - B * A) * Dt * Dt);
    auto const eq_4_105_approx = (expA_Dt * expB_Dt * exp_commutator_Dt).eval();
    EXPECT_MATRIX_CLOSE(expA_plus_B_Dt, eq_4_105_approx, 50. * qube::maths::pow(Dt, 3));

    debug() << ">> expA_plus_B_Dt:\n" << qpp::disp(expA_plus_B_Dt) << "\n\n";
    debug() << ">> eq_4_105_approx:\n" << qpp::disp(eq_4_105_approx) << "\n\n";

}
