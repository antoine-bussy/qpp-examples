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
    using namespace Eigen::indexing;
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
    EXPECT_MATRIX_CLOSE(eigsA(lastN(_2_pow_m)), eigsB, 1e-12);

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

    auto const psi = qpp::randket(_4_pow_n);
    auto const A = psi.reshaped(_2_pow_n, _2_pow_n).transpose().eval();

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

    auto const psi = qpp::randket(_2_pow_npm);
    auto const A = psi.reshaped(_2_pow_m, _2_pow_n).transpose();

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

    auto const t0 = std::chrono::high_resolution_clock::now();
    auto const svd = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto const& U = svd.matrixU();
    auto const& D = svd.singularValues();
    auto const  V = svd.matrixV().adjoint();
    auto const tf = std::chrono::high_resolution_clock::now();

    if constexpr (print_text)
        std::cerr << "All: " << std::chrono::duration_cast<std::chrono::microseconds>(tf - t0).count() << " us" << std::endl;

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

//! @brief Exercise 2.76
TEST(chapter2_5, schmidt_decomposition_proof_different_dimensions_qpp)
{
    qpp_e::maths::seed(1982u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr m = 3u;
    auto constexpr _2_pow_m = qpp_e::maths::pow(2u, m);
    auto constexpr range_m = std::views::iota(0u, _2_pow_m) | std::views::common;
    auto constexpr _2_pow_npm = _2_pow_n * _2_pow_m;
    auto constexpr policy = std::execution::par;

    auto const psi = qpp::randket(_2_pow_npm);

    auto const t0 = std::chrono::high_resolution_clock::now();
    auto const U = qpp::schmidtA(psi, { _2_pow_n, _2_pow_m });
    auto const t1 = std::chrono::high_resolution_clock::now();
    auto const D = qpp::schmidtcoeffs(psi, { _2_pow_n, _2_pow_m });
    auto const t2 = std::chrono::high_resolution_clock::now();
    auto const V = qpp::schmidtB(psi, { _2_pow_n, _2_pow_m });
    auto const tf = std::chrono::high_resolution_clock::now();

    if constexpr (print_text)
    {
        std::cerr << "U:   " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us" << std::endl;
        std::cerr << "D:   " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
        std::cerr << "V:   " << std::chrono::duration_cast<std::chrono::microseconds>(tf - t2).count() << " us" << std::endl;
        std::cerr << "All: " << std::chrono::duration_cast<std::chrono::microseconds>(tf - t0).count() << " us" << std::endl;
    }

    EXPECT_MATRIX_CLOSE(U * U.adjoint(), Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);
    EXPECT_MATRIX_CLOSE(V * V.adjoint(), Eigen::MatrixXcd::Identity(_2_pow_m, _2_pow_m), 1e-12);

    auto const psi2 = std::transform_reduce(policy, range_m.begin(), range_m.end()
        , Eigen::VectorXcd::Zero(_2_pow_npm).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (D[i] * qpp::kron(U.col(i), V.col(i))).eval();
    });
    EXPECT_MATRIX_CLOSE(psi2, psi, 1e-12);
    EXPECT_COMPLEX_CLOSE(psi2.squaredNorm(), 1., 1e-12);

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

//! @brief Exercise 2.78 part 1
TEST(chapter2_5, schmidt_number)
{
    qpp_e::maths::seed(974u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr m = 3u;
    auto constexpr _2_pow_m = qpp_e::maths::pow(2u, m);

    auto const a = qpp::randket(_2_pow_n);
    auto const b = qpp::randket(_2_pow_m);

    auto const psi = qpp::kron(a, b);
    auto const A = psi.reshaped(_2_pow_m, _2_pow_n).transpose();

    auto const schmidt = A.bdcSvd().setThreshold(1e-2).rank();
    EXPECT_EQ(schmidt, 1);
}

//! @brief Exercise 2.78 part 2
TEST(chapter2_5, reduced_density_operator_of_product_state)
{
    qpp_e::maths::seed(33u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr m = 3u;
    auto constexpr _2_pow_m = qpp_e::maths::pow(2u, m);

    {
        auto const a = qpp::randket(_2_pow_n);
        auto const b = qpp::randket(_2_pow_m);
        auto const psi_product = qpp::kron(a, b);

        auto const rho = (psi_product * psi_product.adjoint());
        expect_density_operator(rho, 1e-12);

        auto const rhoA = qpp::ptrace2(rho, { _2_pow_n, _2_pow_m });
        auto const rhoB = qpp::ptrace1(rho, { _2_pow_n, _2_pow_m });

        auto const trace_rhoA_2 = (rhoA * rhoA).trace();
        auto const trace_rhoB_2 = (rhoB * rhoB).trace();

        EXPECT_COMPLEX_CLOSE(trace_rhoA_2, 1., 1e-12);
        EXPECT_COMPLEX_CLOSE(trace_rhoB_2, 1., 1e-12);

        auto const sum_lambda_i_pow_4 = qpp::schmidtcoeffs(psi_product, { _2_pow_n, _2_pow_m }).cwiseAbs2().squaredNorm();
        EXPECT_COMPLEX_CLOSE(trace_rhoA_2, sum_lambda_i_pow_4, 1e-12);
        EXPECT_COMPLEX_CLOSE(trace_rhoB_2, sum_lambda_i_pow_4, 1e-12);
    }

    {
        auto const psi_not_product = qpp::randket(_2_pow_n * _2_pow_m);

        auto const rho = (psi_not_product * psi_not_product.adjoint());
        expect_density_operator(rho, 1e-12);

        auto const rhoA = qpp::ptrace2(rho, { _2_pow_n, _2_pow_m });
        auto const rhoB = qpp::ptrace1(rho, { _2_pow_n, _2_pow_m });

        auto const trace_rhoA_2 = (rhoA * rhoA).trace();
        auto const trace_rhoB_2 = (rhoB * rhoB).trace();

        EXPECT_COMPLEX_NOT_CLOSE(trace_rhoA_2, 1., 1e-2);
        EXPECT_COMPLEX_NOT_CLOSE(trace_rhoB_2, 1., 1e-2);

        auto const sum_lambda_i_pow_4 = qpp::schmidtcoeffs(psi_not_product, { _2_pow_n, _2_pow_m }).cwiseAbs2().squaredNorm();
        EXPECT_COMPLEX_CLOSE(trace_rhoA_2, sum_lambda_i_pow_4, 1e-12);
        EXPECT_COMPLEX_CLOSE(trace_rhoB_2, sum_lambda_i_pow_4, 1e-12);
    }
}

//! @brief Equations 2.207 through 2.211
TEST(chapter2_5, purification)
{
    qpp_e::maths::seed(3u);

    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const rhoA = qpp::randrho(_2_pow_n);

    EXPECT_COMPLEX_NOT_CLOSE((rhoA * rhoA).trace(), 1., 1e-2);

    auto const [p, iA] = qpp::heig(rhoA);

    auto const AR = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::VectorXcd::Zero(_2_pow_n * _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (std::sqrt(p[i]) * qpp::kron(iA.col(i), Eigen::VectorXcd::Unit(_2_pow_n, i))).eval();
    });

    auto const rhoAR = qpp::prj(AR);
    expect_density_operator(rhoAR, 1e-12);

    auto const rho = qpp::ptrace2(rhoAR, { _2_pow_n, _2_pow_n });
    EXPECT_MATRIX_CLOSE(rho, rhoA, 1e-12);
}

//! @brief Exercice 2.79
TEST(chapter2_5, schmidt_decomposition_two_qubits)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    auto const states = std::array
    {
        (inv_sqrt2 * (00_ket + 11_ket)).eval(),
        (0.5 * (00_ket + 01_ket + 10_ket + 11_ket)).eval(),
        (std::numbers::inv_sqrt3 * (00_ket + 01_ket + 10_ket)).eval(),
    };

    for(auto&& state : states)
    {
        auto const [schmidt_basisA, schmidt_basisB, schmidt_coeffs, schmidt_probs] = qpp::schmidt(state);

        EXPECT_EQ(schmidt_coeffs.size(), 2);
        EXPECT_COMPLEX_CLOSE(schmidt_coeffs.squaredNorm(), 1., 1e-12);
        EXPECT_GE(schmidt_coeffs.minCoeff(), -1e-12);

        auto const verification_state = (schmidt_coeffs[0] * qpp::kron(schmidt_basisA.col(0), schmidt_basisB.col(0))
                                    +schmidt_coeffs[1] * qpp::kron(schmidt_basisA.col(1), schmidt_basisB.col(1))).eval();
        EXPECT_MATRIX_CLOSE(state, verification_state, 1e-12);

        if constexpr (print_text)
        {
            std::cerr << "State:\n" << qpp::disp(state) << "\n";
            std::cerr << "Schmidt Basis A:\n" << qpp::disp(schmidt_basisA) << "\n";
            std::cerr << "Schmidt Basis B:\n" << qpp::disp(schmidt_basisB) << "\n";
            std::cerr << "Schmidt Coeffs:\n" << qpp::disp(schmidt_coeffs) << "\n";
            std::cerr << "Schmidt Probs:\n" << qpp::disp(schmidt_probs) << "\n";
            std::cerr << "Verification State:\n" << qpp::disp(verification_state) << "\n\n";
        }
    }
}

//! @brief Exercice 2.80
TEST(chapter2_5, same_schmidt_coefficients)
{
    auto constexpr n = 5u;
    auto constexpr m = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr _2_pow_m = qpp_e::maths::pow(2u, m);

    auto const psi = qpp::randket(_2_pow_n * _2_pow_m);
    auto const [schmidt_basisA, schmidt_basisB, schmidt_coeffs, schmidt_probs] = qpp::schmidt(psi, { _2_pow_n, _2_pow_m });
    EXPECT_EQ(schmidt_coeffs.size(), std::min(_2_pow_n, _2_pow_m));

    /* Vector X only depends on Schmidt coefficients */
    auto const X = qpp::kron(schmidt_coeffs.cast<Eigen::dcomplex>(), Eigen::VectorXcd::Unit(1u + std::min(_2_pow_n, _2_pow_m), 0u)).eval();
    /* With U = schmidt_basisA.adjoint() and V = schmidt_basisB.adjoint(), we have Y = (U x V) * psi */
    auto const Y = (qpp::kron(schmidt_basisA.adjoint(), schmidt_basisB.adjoint()) * psi).eval();

    EXPECT_MATRIX_CLOSE(schmidt_basisA * schmidt_basisA.adjoint(), Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n), 1e-12);
    EXPECT_MATRIX_CLOSE(schmidt_basisB * schmidt_basisB.adjoint(), Eigen::MatrixXcd::Identity(_2_pow_m, _2_pow_m), 1e-12);

    /* With some padding on X, we have X = Y */
    EXPECT_MATRIX_CLOSE(X, Y.head(X.size()), 1e-12);
    EXPECT_TRUE(Y.tail(Y.size() - X.size()).isZero(1e-12));
    /* Therefore, for any psi1 and psi2 with the same Schmidt coefficients, we have: X = (U1 x V1) * psi1 = (U2 x V2) * psi2 */

    if constexpr (print_text)
    {
        std::cerr << "State:\n" << qpp::disp(psi) << "\n";
        std::cerr << "Schmidt Basis A:\n" << qpp::disp(schmidt_basisA) << "\n";
        std::cerr << "Schmidt Basis B:\n" << qpp::disp(schmidt_basisB) << "\n";
        std::cerr << "Schmidt Coeffs:\n" << qpp::disp(schmidt_coeffs) << "\n";
        std::cerr << "Schmidt Probs:\n" << qpp::disp(schmidt_probs) << "\n";
        std::cerr << "X:\n" << qpp::disp(X.transpose()) << "\n";
        std::cerr << "Y:\n" << qpp::disp(Y.transpose()) << "\n\n";
    }
}

//! @brief Exercice 2.81
TEST(chapter2_5, freedom_in_purifications)
{
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const rhoA = qpp::randrho(_2_pow_n);
    EXPECT_COMPLEX_NOT_CLOSE((rhoA * rhoA).trace(), 1., 1e-2);

    auto const iR = qpp::randU(_2_pow_n);
    auto const [p, iA] = qpp::heig(rhoA);

    auto const AR = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::VectorXcd::Zero(_2_pow_n * _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (std::sqrt(p[i]) * qpp::kron(iA.col(i), iR.col(i))).eval();
    });

    auto const rhoAR = qpp::prj(AR);
    expect_density_operator(rhoAR, 1e-12);

    auto const rho = qpp::ptrace2(rhoAR, { _2_pow_n, _2_pow_n });
    EXPECT_MATRIX_CLOSE(rho, rhoA, 1e-12);

    /* Matrix X only depends on rhoA */
    auto const X = (iA * p.cwiseSqrt().asDiagonal()).eval();

    /* With Ur as the matrix of basis iR, we have Y = (Ia x Ur) AR */
    auto const Y = (qpp::kron(Eigen::MatrixXcd::Identity(_2_pow_n,_2_pow_n), iR.adjoint()) * AR).eval();
    auto const Y_reshaped = Y.reshaped(_2_pow_n,_2_pow_n);

    /* After reshaping and transposition on Y, we have X = Y */
    EXPECT_MATRIX_CLOSE(X, Y_reshaped.transpose(), 1e-12);
    /* Therefore, for any purification AR1 and AR2, we have: X = (Ia x Ur1) AR1 = (Ia x Ur2) AR2 */

    if constexpr (print_text)
    {
        std::cerr << "X:\n" << qpp::disp(X) << "\n\n";
        std::cerr << "Y_reshaped:\n" << qpp::disp(Y_reshaped.transpose()) << "\n";
    }
}

//! @brief Exercice 2.82, part 1 & 2
TEST(chapter2_5, purification_and_measurement)
{
    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const rhoA = qpp::randrho(_2_pow_n);

    EXPECT_COMPLEX_NOT_CLOSE((rhoA * rhoA).trace(), 1., 1e-2);

    auto const [p, iA] = qpp::heig(rhoA);
    auto const iR = Eigen::MatrixXcd::Identity(_2_pow_n, _2_pow_n);

    auto const AR = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::VectorXcd::Zero(_2_pow_n * _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (std::sqrt(p[i]) * qpp::kron(iA.col(i), iR.col(i))).eval();
    });

    auto const rhoAR = qpp::prj(AR);
    expect_density_operator(rhoAR, 1e-12);

    /* Part 1 */
    auto const rho = qpp::ptrace2(rhoAR, { _2_pow_n, _2_pow_n });
    EXPECT_MATRIX_CLOSE(rho, rhoA, 1e-12);

    /* Part 2 */
    auto const [result, probabilities, resulting_state] = qpp::measure(rhoAR, iR, { 1u }, { _2_pow_n, _2_pow_n }, true);
    EXPECT_MATRIX_CLOSE(Eigen::VectorXd::Map(probabilities.data(), probabilities.size()), p, 1e-12);

    for(auto&& i : range)
    {
        auto const [resultA, probabilitiesA, resulting_stateA] = qpp::measure(resulting_state[i], iA);
        EXPECT_MATRIX_CLOSE(Eigen::VectorXd::Map(probabilitiesA.data(), probabilitiesA.size()), Eigen::VectorXd::Unit(probabilitiesA.size(), i), 1e-12);
        EXPECT_MATRIX_CLOSE(resulting_stateA[i], qpp::prj(iA.col(i)), 1e-12);
    }
}

//! @brief Exercice 2.82, part 3
TEST(chapter2_5, purification_and_measurement_3)
{
    using namespace Eigen::indexing;

    qpp_e::maths::seed(798u);

    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const rhoA = qpp::randrho(_2_pow_n);
    EXPECT_COMPLEX_NOT_CLOSE((rhoA * rhoA).trace(), 1., 1e-2);

    auto const iR = qpp::randU(_2_pow_n);
    auto const [p, iA] = qpp::heig(rhoA);

    auto const AR = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::VectorXcd::Zero(_2_pow_n * _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (std::sqrt(p[i]) * qpp::kron(iA.col(i), iR.col(i))).eval();
    });

    auto const rhoAR = qpp::prj(AR);
    expect_density_operator(rhoAR, 1e-12);

    /* Part 3 */
    auto const [result, probabilities, resulting_state] = qpp::measure(rhoAR, iR, { 1u }, { _2_pow_n, _2_pow_n }, true);

    EXPECT_MATRIX_CLOSE(Eigen::VectorXd::Map(probabilities.data(), probabilities.size()), p, 1e-12);


    if constexpr (print_text)
    {
        std::cerr << ">> p: " << qpp::disp(p.transpose()) << "\n";
        std::cerr << ">> Probabilities: " << qpp::disp(probabilities, ", ") << "\n\n";
    }

    for (auto&& i : range)
    {
        auto const& st = resulting_state[i];
        auto const [pi, iAi] = qpp::heig(st);
        EXPECT_MATRIX_CLOSE(pi, Eigen::VectorXd::Unit(_2_pow_n, _2_pow_n-1), 1e-12);
        EXPECT_COLLINEAR(iAi(all,last), iA.col(i), 1e-12);

        if constexpr (print_text)
        {
            std::cerr << ">> pi: " << qpp::disp(pi.transpose()) << "\n";
            std::cerr << ">> iA, last col: " << qpp::disp(iA.col(i).transpose()) << "\n";
            std::cerr << ">> iAi, ith col: " << qpp::disp(iAi(all,last).transpose()) << "\n\n";
        }
    }
}
