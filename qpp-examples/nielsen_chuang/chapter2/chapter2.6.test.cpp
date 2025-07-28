#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.hpp>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>
#include <qpp-examples/qube/debug.hpp>

#include <unsupported/Eigen/KroneckerProduct>

#include <execution>
#include <numbers>
#include <ranges>

using namespace qube::stream;

namespace
{
    //! @brief Compute mean and variance
    auto statistics(qube::maths::Matrix auto const& M, qube::maths::Matrix auto const& state)
    {
        auto const mean = state.dot(M * state);
        EXPECT_NEAR(mean.imag(), 0., 1e-12);
        auto const variance = state.dot(M * M * state) - mean * mean;
        EXPECT_GE(variance.real(), -1e-12);
        EXPECT_NEAR(variance.imag(), 0., 1e-12);
        return std::tuple{ mean.real(), variance.real(), std::sqrt(std::max(variance.real(), 0.)) };
    }

    //! @brief Characterization of density operator from theorem 2.5
    auto expect_density_operator(qube::maths::Matrix auto const& rho, qube::maths::RealNumber auto const& precision)
    {
        EXPECT_COMPLEX_CLOSE(rho.trace(), 1., precision);
        EXPECT_MATRIX_CLOSE(rho.adjoint(), rho, precision);
        EXPECT_GE(qpp::hevals(rho).minCoeff(), -precision);
    }
}

//! @brief Box 2.7 and equations 2.213 through 2.217
TEST(chapter2_6, anti_correlations)
{
    qube::maths::seed();

    auto constexpr sqrt2 = std::numbers::sqrt2;
    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto const state = qpp::st.b11;

    auto const v = Eigen::Vector3d::Random().normalized().eval();
    auto const v_dot_sigma = (v[0] * qpp::gt.X + v[1] * qpp::gt.Y + v[2] * qpp::gt.Z).eval();
    auto const v_dot_sigma_hvects = qpp::hevects(v_dot_sigma);

    auto const [result, probabilities, resulting_state] = qpp::measure(state, v_dot_sigma_hvects, { 0 });
    EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities.data()), Eigen::Vector2d::Constant(0.5), 1e-12);

    debug() << ">> state:\n" << qpp::disp(state) << '\n';
    debug() << ">> v . sigma:\n" << qpp::disp(v_dot_sigma) << "\n\n";

    debug() << ">> Measurement result: " << result << '\n';
    debug() << ">> Probabilities: ";
    debug() << qpp::disp(probabilities, {", "}) << '\n';
    debug() << ">> Resulting states:\n";
    for (auto&& st : resulting_state)
        debug() << qpp::disp(st) << "\n\n";

    for (auto&& i : { 0u, 1u })
    {
        auto const j = 1u - i;
        auto const [result2, probabilities2, resulting_state2] = qpp::measure(resulting_state[i], v_dot_sigma_hvects);

        EXPECT_EQ(result2, j);
        EXPECT_MATRIX_CLOSE(Eigen::Vector2d::Map(probabilities2.data()), Eigen::Vector2d::Unit(j), 1e-12);
        EXPECT_MATRIX_CLOSE(resulting_state2[j], resulting_state[i], 1e-12);

        debug() << ">> Measurement result: " << result2 << '\n';
        debug() << ">> Probabilities: ";
        debug() << qpp::disp(probabilities2, {", "}) << '\n';
        debug() << ">> Resulting states:\n";
        for (auto&& stt : resulting_state2)
            debug() << qpp::disp(stt) << "\n\n";
    }

    EXPECT_MATRIX_CLOSE(v_dot_sigma_hvects * v_dot_sigma_hvects.adjoint(), Eigen::Matrix2cd::Identity(), 1e-12);
    auto const v_dot_sigma_hvects_inv = v_dot_sigma_hvects.adjoint();

    auto const a = v_dot_sigma_hvects.col(0);
    auto const b = v_dot_sigma_hvects.col(1);
    auto const det = v_dot_sigma_hvects_inv.determinant();

    auto const ab_state = (inv_sqrt2 * (qpp::kron(a, b) - qpp::kron(b, a))).eval();
    EXPECT_MATRIX_CLOSE(det * ab_state, state, 1e-12);

    EXPECT_NEAR(std::norm(det), 1., 1e-12);

    debug() << ">> state    : " << qpp::disp(state.transpose()) << '\n';
    debug() << ">> ab_state : " << qpp::disp(ab_state.transpose()) << '\n';
    debug() << ">> alt_state: " << qpp::disp(det * ab_state.transpose()) << '\n';
    debug() << ">> det: " << det << ", norm: " << std::norm(det) << "\n";
}

//! @brief Equations 2.226 through 2.230
TEST(chapter2_6, bell_inequality)
{
    auto constexpr sqrt2 = std::numbers::sqrt2;
    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto const state = qpp::st.b11;

    auto const Q = qpp::gt.Z;
    auto const R = qpp::gt.X;
    auto const S = (-inv_sqrt2 * (qpp::gt.Z + qpp::gt.X)).eval();
    auto const T = ( inv_sqrt2 * (qpp::gt.Z - qpp::gt.X)).eval();

    auto const [mean_QS, variance_QS, std_QS] = statistics(qpp::kron(Q,S), state);
    auto const [mean_RT, variance_RT, std_RT] = statistics(qpp::kron(R,T), state);
    auto const [mean_RS, variance_RS, std_RS] = statistics(qpp::kron(R,S), state);
    auto const [mean_QT, variance_QT, std_QT] = statistics(qpp::kron(Q,T), state);

    EXPECT_NEAR(mean_QS, inv_sqrt2, 1e-12);
    EXPECT_NEAR(mean_RS, inv_sqrt2, 1e-12);
    EXPECT_NEAR(mean_RT, inv_sqrt2, 1e-12);
    EXPECT_NEAR(mean_QT,-inv_sqrt2, 1e-12);

    EXPECT_NEAR(mean_QS + mean_RS + mean_RT - mean_QT, 2 * sqrt2, 1e-12);
}

//! @brief Problem 2.1
//! @details The problem is poorly worded, missing hypothesis and definitions
//! @see https://math.stackexchange.com/questions/1049553/function-of-pauli-matrices
TEST(chapter2_6, functions_pauli_matrices)
{
    qube::maths::seed();

    auto const n = Eigen::Vector3d::Random().normalized().eval();
    auto const n_dot_sigma = (n[0] * qpp::gt.X + n[1] * qpp::gt.Y + n[2] * qpp::gt.Z).eval();

    auto const [evals, evects] = qpp::heig(n_dot_sigma);
    EXPECT_MATRIX_CLOSE(evals, Eigen::Vector2d(-1, 1), 1e-12);

    auto const e1 = evects.col(1);
    auto const e2 = evects.col(0);

    auto const e1_e1 = (e1 * e1.adjoint()).eval();
    auto const e2_e2 = (e2 * e2.adjoint()).eval();

    EXPECT_MATRIX_CLOSE(n_dot_sigma, e1_e1 - e2_e2, 1e-12);
    EXPECT_MATRIX_CLOSE(Eigen::Matrix2cd::Identity(), e1_e1 + e2_e2, 1e-12);

    debug() << ">> n_dot_sigma:\n" << qpp::disp(n_dot_sigma) << "\n\n";
    debug() << ">> e1_e1:\n" << qpp::disp(e1_e1) << "\n\n";
    debug() << ">> e2_e2:\n" << qpp::disp(e2_e2) << "\n\n";
}

//! @brief Problem 2.2, part (1)
TEST(chapter2_6, properties_of_the_schmidt_number_1)
{
    qube::maths::seed(974u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qube::maths::pow(2u, n);
    auto constexpr m = 3u;
    auto constexpr _2_pow_m = qube::maths::pow(2u, m);
    ASSERT_GE(n, m);
    auto constexpr range_m = std::views::iota(0u, _2_pow_m) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const iA = qpp::randU(_2_pow_n);
    auto const iB = qpp::randU(_2_pow_m);

    auto const mask = Eigen::VectorX<bool>::Random(_2_pow_n).eval();
    auto const lambda = Eigen::VectorXd::Random(_2_pow_n).cwiseProduct(mask.cast<double>()).normalized().eval();

    auto const psi = std::transform_reduce(policy, range_m.begin(), range_m.end()
        , Eigen::VectorXcd::Zero(_2_pow_n * _2_pow_m).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (lambda[i] * qpp::kron(iA.col(i), iB.col(i))).eval();
    });

    auto const [schmidt_basisA, schmidt_basisB, schmidt_coeffs, schmidt_probs] = qpp::schmidt(psi, { _2_pow_n, _2_pow_m });
    auto const schmidt_number = (schmidt_coeffs.array() > 1e-12).count();

    /* psi is a pure state */
    auto const rho = qpp::prj(psi);
    auto const rhoA = qpp::ptrace2(rho, { _2_pow_n, _2_pow_m });
    expect_density_operator(rhoA, 1e-12);

    auto const rank = rhoA.bdcSvd().setThreshold(1e-12).rank();
    EXPECT_EQ(rank, schmidt_number);

    debug() << "lambda:\n" << qpp::disp(lambda) << "\n";
    debug() << "psi:\n" << qpp::disp(psi) << "\n";
    debug() << "Schmidt Basis A:\n" << qpp::disp(schmidt_basisA) << "\n";
    debug() << "Schmidt Basis B:\n" << qpp::disp(schmidt_basisB) << "\n";
    debug() << "Schmidt Coeffs:\n" << qpp::disp(schmidt_coeffs) << "\n";
    debug() << "Schmidt Probs:\n" << qpp::disp(schmidt_probs) << "\n";
    debug() << "Schmidt Number:\n" << schmidt_number << "\n";
}

//! @brief Problem 2.2, part (2)
TEST(chapter2_6, properties_of_the_schmidt_number_2)
{
    for (auto&& s : {1660140020u, 1660140083u})
    {
        qube::maths::seed(s);

        auto constexpr N = 40u;
        auto constexpr M = 27u;
        ASSERT_GE(N, M);
        auto constexpr range_M = std::views::iota(0u, M) | std::views::common;

        auto constexpr n = 4u;
        auto constexpr _2_pow_n = qube::maths::pow(2u, n);
        auto constexpr m = 3u;
        auto constexpr _2_pow_m = qube::maths::pow(2u, m);
        auto constexpr policy = std::execution::par;

        auto const alpha = Eigen::MatrixXcd::Random(_2_pow_n, N).eval();
        auto const beta = Eigen::MatrixXcd::Random(_2_pow_m, M).eval();

        auto const mask_zero = Eigen::ArrayX<bool>::Random(M).eval();
        auto non_zero_alpha_beta = std::atomic{ 0u };

        auto const psi = std::transform_reduce(policy, range_M.begin(), range_M.end()
            , Eigen::VectorXcd::Zero(_2_pow_n * _2_pow_m).eval()
            , std::plus<>{}
            , [&](auto&& i)
        {
            if (mask_zero[i])
                return Eigen::VectorXcd::Zero(_2_pow_n * _2_pow_m).eval();

            auto const alpha_beta = (qpp::kron(alpha.col(i), beta.col(i))).col(0).eval();
            if (alpha_beta.isZero(1e-12))
                return Eigen::VectorXcd::Zero(_2_pow_n * _2_pow_m).eval();

            ++non_zero_alpha_beta;
            return alpha_beta;
        });

        auto const [schmidt_basisA, schmidt_basisB, schmidt_coeffs, schmidt_probs] = qpp::schmidt(psi, { _2_pow_n, _2_pow_m });
        auto const schmidt_number = (schmidt_coeffs.array() > 1e-12).count();

        EXPECT_GE(non_zero_alpha_beta, schmidt_number);

        debug() << "Schmidt Number:\n" << schmidt_number << "\n";
        debug() << "Non-zero alpha*beta:\n" << non_zero_alpha_beta << "\n";
        debug() << "N: " << N << ", M: " << M << "\n";
    }
}

//! @brief Problem 2.2, part (3)
TEST(chapter2_6, properties_of_the_schmidt_number_3)
{
    qube::maths::seed();

    auto constexpr n = 5u;
    auto constexpr _2_pow_n = qube::maths::pow(2u, n);
    auto constexpr m = 4u;
    auto constexpr _2_pow_m = qube::maths::pow(2u, m);
    ASSERT_GE(n, m);
    auto constexpr policy = std::execution::par;

    auto const iA = qpp::randU(_2_pow_n);
    auto const iB = qpp::randU(_2_pow_m);

    auto constexpr j_phi = 9u;
    auto constexpr j_gamma = 6u;
    auto constexpr j_common = 5u;

    ASSERT_GE(_2_pow_m, j_phi);
    ASSERT_GE(j_phi, j_gamma);

    ASSERT_GE(j_phi, j_common);
    ASSERT_GE(j_gamma, j_common);
    auto constexpr d_phi = j_phi - j_common;
    auto constexpr d_gamma = j_gamma - j_common;
    ASSERT_GE(_2_pow_m, j_common + d_phi + d_gamma);

    auto const lambda = Eigen::VectorXd::Random(_2_pow_m).eval();
    auto const schmidt_compose = [&](auto&& start, auto&& end)
    {
        auto const range = std::views::iota(start, end) | std::views::common;
        return std::transform_reduce(policy, range.begin(), range.end()
            , Eigen::VectorXcd::Zero(_2_pow_n * _2_pow_m).eval()
            , std::plus<>{}
            , [&](auto&& i)
        {
            return (lambda[i] * qpp::kron(iA.col(i), iB.col(i))).eval();
        });
    };

    /* It's difficult to get a meaningfull example, so we artificially build one */
    /* Compute a common part of a schmidt-like decomposition */
    auto const state_common = schmidt_compose(0u, j_common);
    /* Use it for alpha*phi and add another part of schmidt-like decomposition */
    auto const alpha_phi = (state_common + schmidt_compose(j_common, j_common + d_phi)).eval();
    /* Use it (negatively) for beta*gamma and add another part of schmidt-like decomposition */
    auto const beta_gamma = (-state_common + schmidt_compose(_2_pow_m - d_gamma, _2_pow_m)).eval();

    /* The common part cancels out, so that the difference of schmidt number between phi and gamma is not trivial (0),
    and neither is psi, i.e. not _2_pow_m. */
    auto const psi = (alpha_phi + beta_gamma).normalized().eval();

    auto const schmidt_number = [&](auto&& state)
    {
        auto const schmidt_coeffs = qpp::schmidtcoeffs(state, { _2_pow_n, _2_pow_m });
        return (schmidt_coeffs.array() > 1e-12).count();
    };

    auto const phi_schmidt_number = schmidt_number(alpha_phi.normalized().eval());
    auto const gamma_schmidt_number = schmidt_number(beta_gamma.normalized().eval());
    auto const psi_schmidt_number = schmidt_number(psi);

    EXPECT_GE(psi_schmidt_number, std::abs(phi_schmidt_number - gamma_schmidt_number));

    debug() << "Schmidt Number phi  : " << phi_schmidt_number << "\n";
    debug() << "Schmidt Number gamma: " << gamma_schmidt_number << "\n";
    debug() << "Schmidt Number psi  : " << psi_schmidt_number << "\n";
}

//! @brief Check that (A x B)(C x D) = (AC) x (BD)
TEST(chapter2_6, kron_product_mixed_product)
{
    qube::maths::seed();

    auto constexpr n = 2u;

    auto const A = Eigen::Matrix<Eigen::dcomplex, n, n>::Random().eval();
    auto const B = Eigen::Matrix<Eigen::dcomplex, n, n>::Random().eval();
    auto const C = Eigen::Matrix<Eigen::dcomplex, n, n>::Random().eval();
    auto const D = Eigen::Matrix<Eigen::dcomplex, n, n>::Random().eval();

    auto const AxB_CxD = (qpp::kron(A,B) * qpp::kron(C,D)).eval();
    auto const ACxBD = qpp::kron((A*C).eval(),(B*D).eval());

    auto const AxB_CxD_eigen = (Eigen::kroneckerProduct(A,B) * Eigen::kroneckerProduct(C,D)).eval();
    auto const ACxBD_eigen = Eigen::kroneckerProduct((A*C).eval(),(B*D).eval()).eval();

    EXPECT_MATRIX_CLOSE(AxB_CxD, ACxBD, 1e-12);
    EXPECT_MATRIX_CLOSE(AxB_CxD_eigen, ACxBD_eigen, 1e-12);

    EXPECT_MATRIX_CLOSE(AxB_CxD, AxB_CxD_eigen, 1e-12);
    EXPECT_MATRIX_CLOSE(ACxBD, ACxBD_eigen, 1e-12);

    debug() << "(A x B)(C x D):\n" << qpp::disp(AxB_CxD) << "\n\n";
    debug() << "(AC) x (BD):\n" << qpp::disp(ACxBD) << "\n\n";
    debug() << "(A x B)(C x D) Eigen:\n" << qpp::disp(AxB_CxD_eigen) << "\n\n";
    debug() << "(AC) x (BD) Eigen:\n" << qpp::disp(ACxBD_eigen) << "\n\n";
}

//! @brief Problem 2.3
TEST(chapter2_6, tsierelson_inequality)
{
    qube::maths::seed();

    auto constexpr n = 2u;
    auto constexpr _2_pow_n = qube::maths::pow(2u, n);

    auto const observable = []
    {
        auto const v = Eigen::Vector3d::Random().normalized().eval();
        return (v[0] * qpp::gt.X + v[1] * qpp::gt.Y + v[2] * qpp::gt.Z).eval();
    };

    auto const Q = observable();
    auto const R = observable();
    auto const S = observable();
    auto const T = observable();

    auto const lie_bracket = [](auto&& A, auto&& B)
    {
        return (A * B - B * A).eval();
    };

    auto const M = (qpp::kron(Q, S) + qpp::kron(R, S) + qpp::kron(R, T) - qpp::kron(Q, T)).eval();
    auto const M2 = (M * M).eval();
    auto const N = (4. * Eigen::Matrix<Eigen::dcomplex, _2_pow_n, _2_pow_n>::Identity() + qpp::kron(lie_bracket(Q,R), lie_bracket(S,T))).eval();

    EXPECT_MATRIX_CLOSE(M2, N, 1e-12);

    auto const [mean, variance, STD] = statistics(M, qpp::randket(_2_pow_n));

    EXPECT_LE(mean, 2. * std::numbers::sqrt2 + 1e-12);

    debug() << "M * M:\n" << qpp::disp(M2) << "\n\n";
    debug() << "N:\n" << qpp::disp(N) << "\n\n";
    debug() << "mean: " << mean << "\n";
    debug() << "2*sqrt(2): " << 2. * std::numbers::sqrt2 << "\n";
}
