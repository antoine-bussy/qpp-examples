#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.hpp>
#include <qube/debug.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/gtest_macros.hpp>
#include <qube/maths/random.hpp>

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
