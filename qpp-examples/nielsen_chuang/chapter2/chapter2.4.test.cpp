#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/random.hpp>

#include <execution>

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

//! @brief Equations 2.138 and 2.139
TEST(chapter2_4, density_operator_transformation)
{
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr I = 7u;
    auto constexpr range = std::views::iota(0u, I) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const p = Eigen::VectorXd::Map(qpp::randprob(I).data(), I).eval();
    EXPECT_NEAR(p.sum(), 1., 1e-12);

    auto const rho = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        auto const psi_i = qpp::randket(_2_pow_n);
        return (p[i] * qpp::prj(psi_i)).eval();
    });

    expect_density_operator(rho, 1e-12);

    auto target = qpp::qram(n);
    std::iota(target.begin(), target.end(), 0u);

    auto const U = qpp::randU(_2_pow_n);
    auto const rho_out = qpp::apply(rho, U, target);
    EXPECT_MATRIX_CLOSE(rho_out, U * rho * U.adjoint(), 1e-12);
}

//! @brief Equation 2.139
TEST(chapter2_4, density_operator_transformation_2)
{
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);

    auto const rho = qpp::randrho(_2_pow_n);
    expect_density_operator(rho, 1e-12);

    auto target = qpp::qram(n);
    std::iota(target.begin(), target.end(), 0u);

    auto const U = qpp::randU(_2_pow_n);
    auto const rho_out = qpp::apply(rho, U, target);
    EXPECT_MATRIX_CLOSE(rho_out, U * rho * U.adjoint(), 1e-12);
}

//! @brief Equations 2.140 through 2.147
TEST(chapter2_4, density_operator_measure)
{
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr MM = 7u;

    auto const rho = qpp::randrho(_2_pow_n);
    auto const M = qpp::randkraus(MM, _2_pow_n);

    auto const [result, probabilities, resulting_state] = qpp::measure(rho, M);

    for (auto&& m : std::views::iota(0u, MM))
    {
        auto const pm = (M[m].adjoint() * M[m] * rho).trace();
        EXPECT_COMPLEX_CLOSE(pm, probabilities[m], 1e-12);

        auto const rho_m = ((M[m] * rho * M[m].adjoint()) / pm).eval();
        EXPECT_MATRIX_CLOSE(rho_m, resulting_state[m], 1e-12);
    }
}

//! @brief Equations 2.148 through 2.152
TEST(chapter2_4, density_operator_measure_output)
{
    qpp_e::maths::seed(0u);
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr MM = 7u;
    auto constexpr range = std::views::iota(0u, MM) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const rho = qpp::randrho(_2_pow_n);
    expect_density_operator(rho, 1e-12);
    auto const M = qpp::randkraus(MM, _2_pow_n);

    auto const [result, probabilities, resulting_state] = qpp::measure(rho, M);

    auto const rho_out = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& m)
    {
        return (M[m] * rho * M[m].adjoint()).eval();
    });
    EXPECT_MATRIX_NOT_CLOSE(rho_out, rho, 1e-1);
    expect_density_operator(rho_out, 1e-12);
}

//! @brief Theorem 2.5 and equations 2.153 through 2.157
TEST(chapter2_4, density_operator_characterization)
{
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr I = 7u;
    auto constexpr range = std::views::iota(0u, I) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto const p = Eigen::VectorXd::Map(qpp::randprob(I).data(), I).eval();
    EXPECT_NEAR(p.sum(), 1., 1e-12);

    auto const rho = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        auto const psi_i = qpp::randket(_2_pow_n);
        return (p[i] * qpp::prj(psi_i)).eval();
    });

    expect_density_operator(rho, 1e-12);

    auto const phi = qpp::randket(_2_pow_n);
    auto const positivity = phi.dot(rho * phi);
    EXPECT_COMPLEX_CLOSE(positivity, positivity.real(), 1e-12);
    EXPECT_GE(positivity.real(), -1e-12);
}

//! @brief Exercise 2.71
TEST(chapter2_4, mixed_state_criterion)
{
    qpp_e::maths::seed(0u);
    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);

    auto const rho_mixed = qpp::randrho(_2_pow_n);
    expect_density_operator(rho_mixed, 1e-12);
    auto const trace2_mixed = (rho_mixed * rho_mixed).trace();
    EXPECT_COMPLEX_CLOSE(trace2_mixed, trace2_mixed.real(), 1e-12);
    EXPECT_LT(trace2_mixed.real(), 1.);

    auto const rho_pure = qpp::prj(qpp::randket(_2_pow_n));
    expect_density_operator(rho_pure, 1e-12);
    auto const trace2_pure = (rho_pure * rho_pure).trace();
    EXPECT_COMPLEX_CLOSE(trace2_pure, 1., 1e-12);
}

//! @brief Equations 2.162 through 2.165
TEST(chapter2_4, quantum_states_from_density)
{
    using namespace qpp::literals;

    EXPECT_MATRIX_EQ(0_prj, qpp::prj(0_ket));
    EXPECT_MATRIX_EQ(1_prj, qpp::prj(1_ket));

    auto const rho = (0.75 * 0_prj + 0.25 * 1_prj).eval();
    expect_density_operator(rho, 1e-12);

    auto const [eigen_values, eigen_vectors] = qpp::heig(rho);
    EXPECT_NEAR(eigen_values[0], 0.25, 1e-12);
    EXPECT_MATRIX_CLOSE(eigen_vectors.col(0), 1_ket, 1e-12);
    EXPECT_NEAR(eigen_values[1], 0.75, 1e-12);
    EXPECT_MATRIX_CLOSE(eigen_vectors.col(1), 0_ket, 1e-12);

    auto const a = (std::sqrt(0.75) * 0_ket + std::sqrt(0.25) * 1_ket).eval();
    auto const b = (std::sqrt(0.75) * 0_ket - std::sqrt(0.25) * 1_ket).eval();
    EXPECT_MATRIX_CLOSE(rho, 0.5 * qpp::prj(a) + 0.5 * qpp::prj(b), 1e-12);
}

//! @brief Theorem 2.6 and equations 2.166 through 2.174
TEST(chapter2_4, unitary_freedom_density_matrices_1)
{
    qpp_e::maths::seed(10u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);

    auto constexpr m = 7u;
    auto constexpr range_m = std::views::iota(0u, m) | std::views::common;
    auto constexpr l = 5u;
    auto constexpr range_l = std::views::iota(0u, l) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto psi = Eigen::MatrixXcd::Zero(_2_pow_n, m).eval();
    auto const p = Eigen::VectorXd::Map(qpp::randprob(m).data(), m).eval();
    EXPECT_NEAR(p.sum(), 1., 1e-12);
    for(auto&& i : std::views::iota(0u, m))
        psi.col(i) = std::sqrt(p[i]) * qpp::randket(_2_pow_n);

    auto const rho_psi = std::transform_reduce(policy, range_m.begin(), range_m.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (psi.col(i) * psi.col(i).adjoint()).eval();
    });
    expect_density_operator(rho_psi, 1e-12);

    auto phi = Eigen::MatrixXcd::Zero(_2_pow_n, m).eval();
    auto const q = Eigen::VectorXd::Map(qpp::randprob(l).data(), l).eval();
    EXPECT_NEAR(q.sum(), 1., 1e-12);
    for(auto&& i : std::views::iota(0u, l))
        phi.col(i) = std::sqrt(q[i]) * qpp::randket(_2_pow_n);

    auto const rho_phi = std::transform_reduce(policy, range_l.begin(), range_l.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (phi.col(i) * phi.col(i).adjoint()).eval();
    });
    expect_density_operator(rho_phi, 1e-12);
    EXPECT_MATRIX_NOT_CLOSE(rho_psi, rho_phi, 1e-1);

    auto const U = phi.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(psi).transpose().eval();
    EXPECT_MATRIX_NOT_CLOSE(U * U.adjoint(), Eigen::MatrixXcd::Identity(m, m), 1e-1);
}

//! @brief Theorem 2.6 and equations 2.166 through 2.174
TEST(chapter2_4, unitary_freedom_density_matrices_2)
{
    qpp_e::maths::seed(15u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);

    auto constexpr m = 5u;
    auto constexpr range_m = std::views::iota(0u, m) | std::views::common;
    auto constexpr l = 7u;
    auto constexpr range_l = std::views::iota(0u, l) | std::views::common;
    auto constexpr policy = std::execution::par;

    auto psi = Eigen::MatrixXcd::Zero(_2_pow_n, m).eval();
    auto const p = Eigen::VectorXd::Map(qpp::randprob(m).data(), m).eval();
    EXPECT_NEAR(p.sum(), 1., 1e-12);
    for(auto&& i : std::views::iota(0u, m))
        psi.col(i) = std::sqrt(p[i]) * qpp::randket(_2_pow_n);

    auto const rho_psi = std::transform_reduce(policy, range_m.begin(), range_m.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (psi.col(i) * psi.col(i).adjoint()).eval();
    });
    expect_density_operator(rho_psi, 1e-12);

    auto const U = qpp::randU(l).eval();
    auto const U_partial = U(Eigen::seqN(0, m), Eigen::seqN(0, l));
    EXPECT_MATRIX_CLOSE(U_partial * U_partial.adjoint(), Eigen::MatrixXcd::Identity(m, m), 1e-12);

    auto const phi = (psi * U_partial).eval();

    auto const rho_phi = std::transform_reduce(policy, range_l.begin(), range_l.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (phi.col(i) * phi.col(i).adjoint()).eval();
    });
    expect_density_operator(rho_phi, 1e-12);

    EXPECT_MATRIX_CLOSE(rho_psi, rho_phi, 1e-12);
}

//! @brief Theorem 2.6 and equations 2.166 through 2.174
TEST(chapter2_4, unitary_freedom_density_matrices_3)
{
    using namespace qpp::literals;

    auto psi = Eigen::Matrix2cd::Zero().eval();
    psi.col(0) = std::sqrt(0.75) * 0_ket;
    psi.col(1) = std::sqrt(0.25) * 1_ket;

    auto const rho_psi = (psi.col(0) * psi.col(0).adjoint() + psi.col(1) * psi.col(1).adjoint()).eval();
    expect_density_operator(rho_psi, 1e-12);

    auto phi = Eigen::Matrix2cd::Zero().eval();
    phi.col(0) = std::sqrt(0.5) * (std::sqrt(0.75) * 0_ket + std::sqrt(0.25) * 1_ket);
    phi.col(1) = std::sqrt(0.5) * (std::sqrt(0.75) * 0_ket - std::sqrt(0.25) * 1_ket);

    auto const rho_phi = (phi.col(0) * phi.col(0).adjoint() + phi.col(1) * phi.col(1).adjoint()).eval();
    expect_density_operator(rho_phi, 1e-12);

    EXPECT_MATRIX_CLOSE(rho_psi, rho_phi, 1e-12);

    auto const U = (phi.inverse() * psi).eval();
    EXPECT_MATRIX_CLOSE(U * U.adjoint(), Eigen::Matrix2cd::Identity(), 1e-12);
}

namespace
{
    auto density(qpp_e::maths::Matrix auto const& r) -> Eigen::Matrix2cd
    {
        assert(r.cols() == 1 && r.rows() == 3);
        return 0.5 * (Eigen::Matrix2cd::Identity() + r[0] * qpp::gt.X + r[1] * qpp::gt.Y + r[2] * qpp::gt.Z);
    }

    auto bloch_vector(qpp_e::maths::Matrix auto const& rho) -> Eigen::Vector3d
    {
        assert(rho.cols() == 2 && rho.rows() == 2);
        return { 2 * rho(1,0).real(), 2 * rho(1,0).imag(), 2 * rho(0,0).real() - 1 };
    }
}

//! @brief Exercise 2.72 (1)
TEST(chapter2_4, generalized_bloch_sphere_1)
{
    auto const r = Eigen::Vector3d::Random().normalized().eval();
    expect_density_operator(density(r), 1e-12);
    expect_density_operator(density(0.9 * r), 1e-12);
    EXPECT_LT(qpp::hevals(density(1.1 * r)).minCoeff(), 1e-12);
}

//! @brief Exercise 2.72 (2)
TEST(chapter2_4, generalized_bloch_sphere_2)
{
    auto const rho = density(Eigen::Vector3d::Zero());
    expect_density_operator(rho, 1e-12);
    EXPECT_MATRIX_CLOSE(rho, 0.5 * Eigen::Matrix2cd::Identity(), 1e-12);

    auto const r = bloch_vector(0.5 * Eigen::Matrix2cd::Identity());
    EXPECT_TRUE(r.isZero(1e-12));
}

//! @brief Exercise 2.72 (3)
TEST(chapter2_4, generalized_bloch_sphere_3)
{
    qpp_e::maths::seed(123u);

    auto const rho_mixed = qpp::randrho();
    EXPECT_LE(bloch_vector(rho_mixed).squaredNorm(), 1. - 1e-1);

    auto const rho_pure = qpp::prj(qpp::randket());
    EXPECT_NEAR(bloch_vector(rho_pure).squaredNorm(), 1., 1e-12);

    auto const r = Eigen::Vector3d::Random().normalized().eval();
    auto const rho_pure2 = density(r);
    EXPECT_COMPLEX_CLOSE((rho_pure2 * rho_pure2).trace(), 1., 1e-12);

    auto const rho_mixed2 = density(0.9 * r);
    EXPECT_COMPLEX_NOT_CLOSE((rho_mixed2 * rho_mixed2).trace(), 1., 1e-1);
}

//! @brief Exercise 2.72 (4)
TEST(chapter2_4, generalized_bloch_sphere_4)
{
    using namespace qpp::literals;
    using namespace std::complex_literals;
    qpp_e::maths::seed(123u);

    auto const angles = Eigen::Vector2d::Random().eval();
    auto const& theta = angles[0];
    auto const& phi = angles[1];

    auto const psi = (std::cos(0.5 * theta) * 0_ket + std::exp(1i * phi) * std::sin(0.5 * theta) * 1_ket).eval();
    EXPECT_NEAR(psi.squaredNorm(), 1., 1e-12);
    auto const rho = (psi * psi.adjoint()).eval();
    expect_density_operator(rho, 1e-12);
    EXPECT_COMPLEX_CLOSE((rho * rho).trace(), 1., 1e-12);

    auto const r = Eigen::Vector3d
    {
        std::sin(theta) * std::cos(phi),
        std::sin(theta) * std::sin(phi),
        std::cos(theta),
    };

    EXPECT_MATRIX_CLOSE(r, bloch_vector(rho), 1e-12);
}

//! @brief Exercise 2.73
TEST(chapter2_4, minimal_ensemble)
{
    qpp_e::maths::seed(46u);

    auto constexpr n = 4u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);

    auto constexpr m = 5u;
    auto constexpr range_m = std::views::iota(0u, m) | std::views::common;
    auto const _0_m = Eigen::seqN(0, m);
    auto constexpr policy = std::execution::par;

    auto lambda = Eigen::VectorXd::Zero(_2_pow_n).eval();
    lambda(_0_m) = Eigen::VectorXd::Map(qpp::randprob(m).data(), m);

    auto const P = qpp::randU(_2_pow_n);
    auto const rho = (P * lambda.asDiagonal() * P.adjoint()).eval();
    expect_density_operator(rho, 1e-12);

    auto psi = (P(Eigen::all, _0_m) * lambda(_0_m).cwiseSqrt().asDiagonal() * qpp::randU(m)).eval();
    auto const p = psi.colwise().squaredNorm().eval();
    psi.colwise().normalize();
    EXPECT_NEAR(p.sum(), 1., 1e-12);
    EXPECT_MATRIX_NOT_CLOSE(psi.adjoint() * psi, Eigen::MatrixXcd::Identity(m,m), 1e-1);
    if (print_text)
    {
        std::cerr << "Probability vector: " << qpp::disp(p) << '\n';
        std::cerr << "psi.adjoint() * psi:\n" << qpp::disp(psi.adjoint() * psi) << '\n';
    }

    auto lambda_inverse = Eigen::VectorXd::Zero(_2_pow_n).eval();
    lambda_inverse(_0_m) = lambda(_0_m).cwiseInverse();
    auto const rho_inverse = (P * lambda_inverse.asDiagonal() * P.adjoint()).eval();
    EXPECT_MATRIX_CLOSE(rho_inverse * rho * P(Eigen::all, _0_m), P(Eigen::all, _0_m), 1e-12);

    for(auto&& i : range_m)
    {
        EXPECT_COMPLEX_CLOSE(p[i], 1. / psi.col(i).dot(rho_inverse * psi.col(i)), 1e-12);
    }

    auto const rho_psi = std::transform_reduce(policy, range_m.begin(), range_m.end()
        , Eigen::MatrixXcd::Zero(_2_pow_n, _2_pow_n).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (p[i] * psi.col(i) * psi.col(i).adjoint()).eval();
    });
    expect_density_operator(rho_psi, 1e-12);
    EXPECT_MATRIX_CLOSE(rho_psi, rho, 1e-12);
}

//! @brief Equations 2.177 and 2.178
TEST(chapter2_4, partial_trace)
{
    qpp_e::maths::seed(12u);

    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr m = 2u;
    auto constexpr _2_pow_m = qpp_e::maths::pow(2u, m);

    auto const a1 = qpp::randket(_2_pow_n);
    auto const a2 = qpp::randket(_2_pow_n);
    auto const b1 = qpp::randket(_2_pow_m);
    auto const b2 = qpp::randket(_2_pow_m);

    auto const op_a = (a1 * a2.adjoint()).eval();
    auto const op_b = (b1 * b2.adjoint()).eval();

    EXPECT_MATRIX_CLOSE(qpp::ptrace2(qpp::kron(op_a, op_b), { _2_pow_n, _2_pow_m }), qpp::trace(op_b) * op_a, 1e-12);
}

//! @brief Equation 2.184
TEST(chapter2_4, reduced_density_operator)
{
    qpp_e::maths::seed(7u);

    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr m = 2u;
    auto constexpr _2_pow_m = qpp_e::maths::pow(2u, m);

    auto const rho = qpp::randrho(_2_pow_n);
    auto const sigma = qpp::randrho(_2_pow_m);
    auto const rho_sigma = qpp::kron(rho, sigma);

    EXPECT_MATRIX_CLOSE(qpp::ptrace2(rho_sigma, { _2_pow_n, _2_pow_m }), rho, 1e-12);
    EXPECT_MATRIX_CLOSE(qpp::ptrace1(rho_sigma, { _2_pow_n, _2_pow_m }), sigma, 1e-12);
}

//! @brief Equations 2.185 through 2.191
TEST(chapter2_4, reduced_density_operator_entangled)
{
    auto const rho = qpp::prj(qpp::st.b00);
    auto const rho1 = qpp::ptrace2(rho);
    EXPECT_MATRIX_CLOSE(rho1, 0.5 * qpp::gt.Id2, 1e-12);
    EXPECT_COMPLEX_CLOSE(qpp::trace(rho1 * rho1), 0.5, 1e-12);
}

namespace
{
    //! @brief Compute mean and variance
    auto statistics(qpp_e::maths::Matrix auto const& M, qpp_e::maths::Matrix auto const& state)
    {
        auto const mean = state.dot(M * state);
        EXPECT_NEAR(mean.imag(), 0., 1e-12);
        auto const variance = state.dot(M * M * state) - mean * mean;
        EXPECT_GE(variance.real(), -1e-12);
        EXPECT_NEAR(variance.imag(), 0., 1e-12);
        return std::tuple{ mean.real(), variance.real(), std::sqrt(std::max(variance.real(), 0.)) };
    }
}

//! @brief Box 2.6 and equations 2.179 through 2.183
TEST(chapter2_4, partial_trace_observables)
{
    qpp_e::maths::seed(98u);

    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr range_n = std::views::iota(0u, _2_pow_n) | std::views::common;
    auto constexpr l = 2u;
    auto constexpr _2_pow_l = qpp_e::maths::pow(2u, l);

    auto const M = qpp::randH(_2_pow_n);
    auto const [m, m_vec] = qpp::heig(M);
    auto const M_tilde = qpp::kron(M, qpp::gt.Id(_2_pow_l));

    auto P = std::vector<qpp::cmat>(m_vec.cols());
    auto P_tilde = std::vector<qpp::cmat>(m_vec.cols());

    for (auto&& i : range_n)
    {
        auto const state = qpp::kron(m_vec.col(i), qpp::randket(_2_pow_l)).col(0).eval();
        auto const [ mean, variance, STD ] = statistics(M_tilde, state);
        EXPECT_COMPLEX_CLOSE(mean, m[i], 1e-12);
        EXPECT_NEAR(variance, 0., 1e-12);

        P[i] = m_vec.col(i) * m_vec.col(i).adjoint();
        P_tilde[i] = qpp::kron(P[i], qpp::gt.Id(_2_pow_l));
    }

    auto const state_a = qpp::randket(_2_pow_n);
    auto const [result, probabilities, resulting_state] = qpp::measure(state_a, P);

    auto const state_b = qpp::randket(_2_pow_l);
    auto constexpr target = std::views::iota(0u, n) | std::views::common;
    auto const [result_composite, probabilities_composite, resulting_state_composite] = qpp::measure(qpp::kron(state_a, state_b), P, { target.begin(), target.end() });

    for (auto&& i : std::views::iota(0u, n))
    {
        EXPECT_COMPLEX_CLOSE(probabilities[i], probabilities_composite[i], 1e-12);
    }

    if constexpr (print_text)
    {
        std::cerr << ">> Measurement result: " << result << '\n';
        std::cerr << ">> Probabilities: ";
        std::cerr << qpp::disp(probabilities, ", ") << '\n';
        std::cerr << ">> Resulting states:\n";
        for (auto&& it : resulting_state)
            std::cerr << qpp::disp(it) << "\n\n";
        std::cerr << "-----------------------------\n";
        std::cerr << ">> Composite Measurement result: " << result_composite << '\n';
        std::cerr << ">> Composite Probabilities: ";
        std::cerr << qpp::disp(probabilities_composite, ", ") << '\n';
        std::cerr << ">> Composite Resulting states:\n";
        for (auto&& it : resulting_state_composite)
            std::cerr << qpp::disp(it) << "\n\n";
    }

    auto const rho_ab = qpp::randrho(_2_pow_n * _2_pow_l);
    auto const rho_a = qpp::ptrace2(rho_ab, { _2_pow_n, _2_pow_l });
    EXPECT_COMPLEX_CLOSE(qpp::trace(M * rho_a), qpp::trace(M_tilde * rho_ab), 1e-12);
}

//! @brief Exercise 2.74
TEST(chapter2_4, composite_of_pure_states)
{
    qpp_e::maths::seed(25u);

    auto constexpr n = 3u;
    auto constexpr _2_pow_n = qpp_e::maths::pow(2u, n);
    auto constexpr m = 2u;
    auto constexpr _2_pow_m = qpp_e::maths::pow(2u, m);

    auto const a = qpp::randket(_2_pow_n);
    auto const rho_a = qpp::prj(a);
    expect_density_operator(rho_a, 1e-12);
    EXPECT_COMPLEX_CLOSE(qpp::trace(rho_a * rho_a), 1., 1e-12);

    auto const b = qpp::randket(_2_pow_m);
    auto const rho_b = qpp::prj(b);
    expect_density_operator(rho_b, 1e-12);
    EXPECT_COMPLEX_CLOSE(qpp::trace(rho_b * rho_b), 1., 1e-12);

    auto const rho_A = qpp::ptrace2(qpp::kron(rho_a, rho_b), { _2_pow_n, _2_pow_m });
    expect_density_operator(rho_A, 1e-12);
    EXPECT_COMPLEX_CLOSE(qpp::trace(rho_A * rho_A), 1., 1e-12);
}

namespace
{
    auto constexpr bell_state_reduce_density_operator(qpp_e::maths::Matrix auto const& state)
    {
        using namespace qpp::literals;

        auto const rho = qpp::prj(state);
        auto const reduced_rho = (0.5 * (0_prj + 1_prj)).eval();
        EXPECT_MATRIX_CLOSE(qpp::ptrace1(rho), reduced_rho, 1e-12);
        EXPECT_MATRIX_CLOSE(qpp::ptrace2(rho), reduced_rho, 1e-12);
        EXPECT_MATRIX_CLOSE(reduced_rho, 0.5 * Eigen::Matrix2cd::Identity(), 1e-12);
    }
}

//! @brief Exercise 2.75
TEST(chapter2_4, reduce_density_operator_of_bell_states)
{
    bell_state_reduce_density_operator(qpp::st.b00);
    bell_state_reduce_density_operator(qpp::st.b01);
    bell_state_reduce_density_operator(qpp::st.b10);
    bell_state_reduce_density_operator(qpp::st.b11);
}

//! @brief Equations 2.192 through 2.201
TEST(chapter2_4, reduce_density_operator_and_quantum_teleportation)
{
    using namespace qpp::literals;
    auto constexpr policy = std::execution::par;

    auto const state = qpp::randket();
    auto const& a = state[0];
    auto const& b = state[1];

    auto const psi2 = (0.5
                *((qpp::kron(00_ket, a * 0_ket + b * 1_ket))
                + (qpp::kron(01_ket, a * 1_ket + b * 0_ket))
                + (qpp::kron(10_ket, a * 0_ket - b * 1_ket))
                + (qpp::kron(11_ket, a * 1_ket - b * 0_ket))
                )).eval();
    EXPECT_COMPLEX_CLOSE(psi2.squaredNorm(), 1., 1e-12);

    auto const [result0, probabilities0, resulting_state0] = qpp::measure(psi2, qpp::gt.Id(4), { 0, 1 }, 2, false);
    EXPECT_MATRIX_CLOSE(Eigen::Vector4d::Map(probabilities0.data()).eval(), Eigen::Vector4d::Constant(0.25), 1e-12);

    EXPECT_MATRIX_CLOSE(resulting_state0[0], qpp::kron(00_ket, a * 0_ket + b * 1_ket), 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state0[1], qpp::kron(01_ket, a * 1_ket + b * 0_ket), 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state0[2], qpp::kron(10_ket, a * 0_ket - b * 1_ket), 1e-12);
    EXPECT_MATRIX_CLOSE(resulting_state0[3], qpp::kron(11_ket, a * 1_ket - b * 0_ket), 1e-12);

    auto const rho_before_alice_measure = qpp::prj(psi2);
    expect_density_operator(rho_before_alice_measure, 1e-12);

    auto const [result1, probabilities1, resulting_state1] = qpp::measure(rho_before_alice_measure, qpp::gt.Id(4), { 0, 1 }, 2, false);
    EXPECT_MATRIX_CLOSE(Eigen::Vector4d::Map(probabilities1.data()).eval(), Eigen::Vector4d::Constant(0.25), 1e-12);

    auto constexpr range = std::views::iota(0u, 4u) | std::views::common;
    auto const rho = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::MatrixXcd::Zero(8, 8).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (probabilities1[i] * resulting_state1[i]).eval();
    });
    auto const expected_rho = std::transform_reduce(policy, range.begin(), range.end()
        , Eigen::MatrixXcd::Zero(8, 8).eval()
        , std::plus<>{}
        , [&](auto&& i)
    {
        return (probabilities0[i] * qpp::prj(resulting_state0[i])).eval();
    });
    EXPECT_MATRIX_CLOSE(rho, expected_rho, 1e-12);

    auto const rhoB = qpp::ptrace1(rho, { 4, 2 });
    EXPECT_MATRIX_CLOSE(rhoB, 0.5 * Eigen::Matrix2d::Identity(), 1e-12);
}
