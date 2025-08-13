#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.hpp>
#include <qube/debug.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/gtest_macros.hpp>
#include <qube/maths/random.hpp>

using namespace qube::stream;

//! @brief Exercise 4.46
TEST(chapter4_7, exponential_complexity_growth)
{
    qube::maths::seed();

    auto constexpr nq = 4ul;
    auto constexpr N = qube::maths::pow(2ul, nq);
    auto const rho = qpp::randrho(N);

    auto const solver = Eigen::SelfAdjointEigenSolver<qpp::cmat>{ rho };
    EXPECT_EQ(solver.info(), Eigen::Success);

    // Check that the eigenvalues are real, positive, and sum to 1
    auto const& eigenvalues = solver.eigenvalues();
    using scalar_t = std::decay_t<decltype(eigenvalues)>::Scalar;
    static_assert(std::is_same<scalar_t, double>());
    EXPECT_TRUE((eigenvalues.array() >= 0.).all());
    EXPECT_NEAR(eigenvalues.sum(), 1., 1e-12);
    // Therefore, the diagonal matrix count for N reals, and 1 constraint, hence N-1 degrees of freedom

    // Check that the eigenvectors are orthonormal
    auto const& eigenvectors = solver.eigenvectors();
    EXPECT_MATRIX_CLOSE(eigenvectors * eigenvectors.adjoint(), Eigen::MatrixXcd::Identity(N, N), 1e-12);
    // The eigenvectors count for N^2 complex numbers
    // The orthogonality condition gives us N(N-1)/2 constraints (N choose 2)
    // The normalization condition gives us N constraints
    // Therefore, the eigenvectors count for N^2 - N(N-1)/2 - N = N(N-1)/2 complex numbers, or N(N-1) real numbers
    // The reasoning above is not a correct proof, but it is a good intuition. Note that it gives the correct dimension
    // of the orthogonal group O(N).

    // TOTAL: N(N-1) + (N-1) = N^2 - 1 real numbers, or N^2 - 1 degrees of freedom, with N = 2^nq, hence 4^nq - 1.

    // Another way to see this is to consider that an Hermitian matrix is self-adjoint,
    // The lower triangular part being equal to the upper triangular part yields N(N-1)/2 constraints.
    // The diagonal elements are real, hence N constraints.
    // In the case of a density matrix, the trace is 1, hence 1 constraint.
}
