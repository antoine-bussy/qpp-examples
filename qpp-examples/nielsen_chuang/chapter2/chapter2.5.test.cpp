#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/random.hpp>

#include <execution>
#include <numbers>

namespace
{
    auto constexpr print_text = false;
}

namespace
{
    //! @brief Characterization of density operator from theorem 2.5
    auto expect_density_operator(qpp_e::maths::Matrix auto const& rho, qpp_e::maths::RealNumber auto const& precision)
    {
        EXPECT_COMPLEX_CLOSE(rho.trace(), 1., precision);
        EXPECT_MATRIX_CLOSE(rho.adjoint(), rho, precision);
        EXPECT_GE(qpp::hevals(rho).minCoeff(), -precision);
    }
}

//! @brief Theorem 2.7 and equation 2.202
TEST(chapter2_5, schmidt_reduced_density_operator)
{
    qpp_e::maths::seed(899u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr m = 3u;
    auto constexpr _2_pow_m = qpp_e::maths::pow(2u, m);

    auto const psi = qpp::randket(_2_pow_n * _2_pow_m);
    auto const rho = qpp::prj(psi);
    expect_density_operator(rho, 1e-12);

    auto const rhoA = qpp::ptrace2(rho, { _2_pow_n, _2_pow_m });
    auto const eigsA = qpp::hevals(rhoA);
    auto const rhoB = qpp::ptrace1(rho, { _2_pow_n, _2_pow_m });
    auto const eigsB = qpp::hevals(rhoB);

    EXPECT_TRUE(eigsA.head(_2_pow_n - _2_pow_m).isZero(1e-12));
    EXPECT_MATRIX_CLOSE(eigsA(Eigen::lastN(_2_pow_m)), eigsB, 1e-12);

    EXPECT_GE(eigsA.minCoeff(), -1e-12);
    EXPECT_GE(eigsB.minCoeff(), -1e-12);

    EXPECT_COMPLEX_CLOSE(qpp::trace(rhoA * rhoA), qpp::trace(rhoB * rhoB), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << "EigsA:\n" << qpp::disp(eigsA) << "\n\n";
        std::cerr << "EigsB:\n" << qpp::disp(eigsB) << "\n\n";
    }
}

//! @brief Theorem 2.7 and equation 2.202
TEST(chapter2_5, schmidt_reduced_density_operator_example)
{
    using namespace qpp::literals;

    auto const psi = (std::numbers::inv_sqrt3_v<double> * (00_ket + 01_ket + 11_ket)).eval();
    auto const rho = (psi * psi.adjoint()).eval();
    expect_density_operator(rho, 1e-12);

    auto const rhoA = qpp::ptrace2(rho);
    auto const eigsA = qpp::hevals(rhoA);
    auto const rhoB = qpp::ptrace1(rho);
    auto const eigsB = qpp::hevals(rhoB);
    EXPECT_MATRIX_CLOSE(eigsA, eigsB, 1e-12);
    EXPECT_GE(eigsA.minCoeff(), -1e-12);
    EXPECT_GE(eigsB.minCoeff(), -1e-12);
    EXPECT_COMPLEX_CLOSE(qpp::trace(rhoA * rhoA), 7./9., 1e-12);
    EXPECT_COMPLEX_CLOSE(qpp::trace(rhoB * rhoB), 7./9., 1e-12);

    if constexpr (print_text)
    {
        std::cerr << "EigsA:\n" << qpp::disp(eigsA) << "\n\n";
        std::cerr << "EigsB:\n" << qpp::disp(eigsB) << "\n\n";
    }
}

//! @brief Theorem 2.7 and equations 2.203 through 2.205
TEST(chapter2_5, schmidt_decomposition_proof)
{
    qpp_e::maths::seed(3112u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr _4_pow_n = _2_pow_n * _2_pow_n;
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const a = qpp::randket(_2_pow_n);
    auto const b = qpp::randket(_2_pow_n);
    auto const A = (a * b.transpose()).eval();
    auto const psi = qpp::kron(a, b);
    EXPECT_COMPLEX_CLOSE(psi.squaredNorm(), 1., 1e-12);
    EXPECT_EQ(psi.size(), _4_pow_n);

    auto const psi2 = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::VectorXcd::Zero(_4_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& j)
    {
        return std::transform_reduce(std::execution::seq, range.begin(), range.end()
            , Eigen::VectorXcd::Zero(_4_pow_n).eval()
            , std::plus<>{}
            , [&](auto&& k)
        {
            return (A(j, k) * qpp::kron(Eigen::VectorXcd::Unit(_2_pow_n, j), Eigen::VectorXcd::Unit(_2_pow_n, k))).eval();
        });
    });
    EXPECT_MATRIX_CLOSE(psi2, psi, 1e-12);
    EXPECT_COMPLEX_CLOSE(psi2.squaredNorm(), 1., 1e-12);

    auto const svd = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto const& U = svd.matrixU();
    auto const& D = svd.singularValues();
    auto const  V = svd.matrixV().adjoint();

    EXPECT_MATRIX_CLOSE(A, U * D.asDiagonal() * V, 1e-12);
    EXPECT_MATRIX_CLOSE(U * U.adjoint(), Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);
    EXPECT_MATRIX_CLOSE(V * V.adjoint(), Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);

    for (auto&& j : range)
        for (auto&& k : range)
        {
            auto const A_jk = std::transform_reduce(std::execution::seq, range.begin(),range.end()
                , std::complex{ 0. }
                , std::plus<>{}
                , [&](auto&& i)
            {
                return U(j, i) * D[i] * V(i, k);
            });
            EXPECT_COMPLEX_CLOSE(A_jk, A(j, k), 1e-12);
        }

    auto const psi3 = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::VectorXcd::Zero(_4_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (D[i] * qpp::kron(U.col(i), V.row(i).transpose())).eval();
    });
    EXPECT_MATRIX_CLOSE(psi3, psi, 1e-12);
    EXPECT_COMPLEX_CLOSE(psi3.squaredNorm(), 1., 1e-12);

    EXPECT_GT(D.minCoeff(), -1e-12);
    EXPECT_COMPLEX_CLOSE(D.squaredNorm(), 1., 1e-12);
}

//! @brief Exercise 2.76
TEST(chapter2_5, schmidt_decomposition_proof_different_dimensions)
{
    qpp_e::maths::seed(1982u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range_n = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr m = 3u;
    auto constexpr _2_pow_m = qpp_e::maths::pow(2u, m);
    auto constexpr range_m = std::views::iota(0u, _2_pow_m) | std::views::common;
    auto constexpr _2_pow_npm = _2_pow_n * _2_pow_m;
    auto constexpr policy = std::execution::par;

    auto const a = qpp::randket(_2_pow_n);
    auto const b = qpp::randket(_2_pow_m);
    auto const A = (a * b.transpose()).eval();
    auto const psi = qpp::kron(a, b);
    EXPECT_COMPLEX_CLOSE(psi.squaredNorm(), 1., 1e-12);
    EXPECT_EQ(psi.size(), _2_pow_npm);

    auto const psi2 = std::transform_reduce(policy, range_n.begin(), range_n.end()
        , Eigen::VectorXcd::Zero(_2_pow_npm).eval()
        , std::plus<>{}
        , [&](auto&& j)
    {
        return std::transform_reduce(std::execution::seq, range_m.begin(), range_m.end()
            , Eigen::VectorXcd::Zero(_2_pow_npm).eval()
            , std::plus<>{}
            , [&](auto&& k)
        {
            return (A(j, k) * qpp::kron(Eigen::VectorXcd::Unit(_2_pow_n, j), Eigen::VectorXcd::Unit(_2_pow_m, k))).eval();
        });
    });
    EXPECT_MATRIX_CLOSE(psi2, psi, 1e-12);
    EXPECT_COMPLEX_CLOSE(psi2.squaredNorm(), 1., 1e-12);

    auto const svd = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto const& U = svd.matrixU();
    auto const& D = svd.singularValues();
    auto const  V = svd.matrixV().adjoint();

    EXPECT_MATRIX_CLOSE(A, U.leftCols(_2_pow_m) * D.asDiagonal() * V, 1e-12);
    EXPECT_MATRIX_CLOSE(U * U.adjoint(), Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);
    EXPECT_MATRIX_CLOSE(V * V.adjoint(), Eigen::MatrixXcd::Identity(_2_pow_m, _2_pow_m), 1e-12);

    for (auto&& j : range_n)
        for (auto&& k : range_m)
        {
            auto const A_jk = std::transform_reduce(std::execution::seq, range_m.begin(),range_m.end()
                , std::complex{ 0. }
                , std::plus<>{}
                , [&](auto&& i)
            {
                return U(j, i) * D[i] * V(i, k);
            });
            EXPECT_COMPLEX_CLOSE(A_jk, A(j, k), 1e-12);
        }

    auto const psi3 = std::transform_reduce(policy, range_m.begin(), range_m.end()
        , Eigen::VectorXcd::Zero(_2_pow_npm).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (D[i] * qpp::kron(U.col(i), V.row(i).transpose())).eval();
    });
    EXPECT_MATRIX_CLOSE(psi3, psi, 1e-12);
    EXPECT_COMPLEX_CLOSE(psi3.squaredNorm(), 1., 1e-12);

    EXPECT_GT(D.minCoeff(), -1e-12);
    EXPECT_COMPLEX_CLOSE(D.squaredNorm(), 1., 1e-12);
}

//! @brief Exercise 2.77
TEST(chapter2_5, three_vector_schmidt_decomposition)
{
    using namespace qpp::literals;

    qpp_e::maths::seed(12u);

    auto const psi = (000_ket + 011_ket).normalized().eval();

    auto const x = qpp::randket();
    auto const y = qpp::randket();
    auto const X = qpp::kron(1_ket, x, y);
    EXPECT_EQ(X.cols(), 1);
    EXPECT_NEAR(std::norm(psi.dot(X.col(0))), 0., 1e-12);

    auto const iA = qpp::randket();
    auto const phase = qpp::randU(1)(0, 0);
    EXPECT_COMPLEX_CLOSE(std::norm(phase), 1., 1e-12);

    auto const iA_orth = (phase * Eigen::Vector2cd{ std::conj(iA[1]), -std::conj(iA[0]) }).eval();
    EXPECT_NEAR(std::norm(iA_orth.dot(iA)), 0., 1e-12);

    EXPECT_COMPLEX_CLOSE((1_ket).dot(iA), iA[1], 1e-12);
    EXPECT_COMPLEX_CLOSE((1_ket).dot(iA_orth), -phase * std::conj(iA[0]), 1e-12);
}
