#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>

#include <unsupported/Eigen/EulerAngles>
#include <unsupported/Eigen/MatrixFunctions>

#include <execution>
#include <numbers>
#include <ranges>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Equations 4.1 through 4.3
TEST(chapter4_2, important_single_qubit_gates)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto const I = Eigen::Matrix2cd::Identity();
    EXPECT_MATRIX_CLOSE(qpp::gt.Id2, I, 1e-12);

    auto const X = Eigen::Matrix2cd{{0., 1.}, {1., 0.}};
    EXPECT_MATRIX_CLOSE(qpp::gt.X, X, 1e-12);

    auto const Y = Eigen::Matrix2cd{{0., -1i}, {1i, 0.}};
    EXPECT_MATRIX_CLOSE(qpp::gt.Y, Y, 1e-12);

    auto const Z = Eigen::Matrix2cd{{1., 0.}, {0., -1.}};
    EXPECT_MATRIX_CLOSE(qpp::gt.Z, Z, 1e-12);

    auto const H = (inv_sqrt2 * Eigen::Matrix2cd{{1., 1.}, {1., -1.}}).eval();
    EXPECT_MATRIX_CLOSE(qpp::gt.H, H, 1e-12);

    auto const S = Eigen::Matrix2cd{{1., 0.}, {0., 1i}};
    EXPECT_MATRIX_CLOSE(qpp::gt.S, S, 1e-12);

    auto const T = Eigen::Matrix2cd{{1., 0.}, {0., std::exp(0.25i * pi)}};
    EXPECT_MATRIX_CLOSE(qpp::gt.T, T, 1e-12);

    auto const TT = (std::exp(0.125i * pi) * Eigen::Matrix2cd{{std::exp(-0.125i * pi), 0.}, {0., std::exp(0.125i * pi)}}).eval();
    EXPECT_MATRIX_CLOSE(TT, T, 1e-12);

    EXPECT_MATRIX_CLOSE((inv_sqrt2 * (X + Z)).eval(), H, 1e-12);
    EXPECT_MATRIX_CLOSE((T*T).eval(), S, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> I:\n" << qpp::disp(I) << "\n\n";
        std::cerr << ">> X:\n" << qpp::disp(X) << "\n\n";
        std::cerr << ">> Y:\n" << qpp::disp(Y) << "\n\n";
        std::cerr << ">> Z:\n" << qpp::disp(Z) << "\n\n";
        std::cerr << ">> H:\n" << qpp::disp(H) << "\n\n";
        std::cerr << ">> S:\n" << qpp::disp(S) << "\n\n";
        std::cerr << ">> T:\n" << qpp::disp(T) << "\n\n";
    }
}

//! @brief Exercise 4.1
TEST(chapter4_2, pauli_matrices_eigen_vectors)
{
    using namespace std::literals::complex_literals;

    auto const [ lambda_X, v_X ] = qpp::heig(qpp::gt.X);
    EXPECT_MATRIX_CLOSE(lambda_X, Eigen::Vector2cd(-1., 1.), 1e-12);
    EXPECT_COLLINEAR(v_X.col(0), Eigen::Vector2cd(-1., 1.), 1e-12);
    EXPECT_COLLINEAR(v_X.col(1), Eigen::Vector2cd(1., 1.), 1e-12);

    auto const [ lambda_Y, v_Y ] = qpp::heig(qpp::gt.Y);
    EXPECT_MATRIX_CLOSE(lambda_Y, Eigen::Vector2cd(-1., 1.), 1e-12);
    EXPECT_COLLINEAR(v_Y.col(0), Eigen::Vector2cd(1., -1i), 1e-12);
    EXPECT_COLLINEAR(v_Y.col(1), Eigen::Vector2cd(1., 1i), 1e-12);

    auto const [ lambda_Z, v_Z ] = qpp::heig(qpp::gt.Z);
    EXPECT_MATRIX_CLOSE(lambda_Z, Eigen::Vector2cd(-1., 1.), 1e-12);
    EXPECT_COLLINEAR(v_Z.col(0), Eigen::Vector2cd(0., 1.), 1e-12);
    EXPECT_COLLINEAR(v_Z.col(1), Eigen::Vector2cd(1., 0.), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> X: " << qpp::disp(lambda_X.transpose()) << "\n" << qpp::disp(v_X) << "\n\n";
        std::cerr << ">> Y: " << qpp::disp(lambda_Y.transpose()) << "\n" << qpp::disp(v_Y) << "\n\n";
        std::cerr << ">> Z: " << qpp::disp(lambda_Z.transpose()) << "\n" << qpp::disp(v_Z) << "\n\n";
    }
}

//! @brief Equations 4.4 through 4.7 and Exercise 4.2
TEST(chapter4_2, rotation_operators)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    auto const theta = qpp::rand(0., 4.*pi);
    auto const cos = std::cos(0.5 * theta);
    auto const sin = std::sin(0.5 * theta);

    auto const Rx = Eigen::Matrix2cd{{cos, -1.i * sin}, {-1.i * sin, cos}};
    EXPECT_MATRIX_CLOSE(Rx, qpp::gt.RX(theta), 1e-12);
    EXPECT_MATRIX_CLOSE(Rx, (-0.5i * theta * qpp::gt.X).exp().eval(), 1e-12);
    EXPECT_MATRIX_CLOSE(Rx, (cos * qpp::gt.Id2 - 1.i * sin * qpp::gt.X).eval(), 1e-12);

    auto const Ry = Eigen::Matrix2cd{{cos, -1. * sin}, {sin, cos}};
    EXPECT_MATRIX_CLOSE(Ry, qpp::gt.RY(theta), 1e-12);
    EXPECT_MATRIX_CLOSE(Ry, (-0.5i * theta * qpp::gt.Y).exp().eval(), 1e-12);
    EXPECT_MATRIX_CLOSE(Ry, (cos * qpp::gt.Id2 - 1.i * sin * qpp::gt.Y).eval(), 1e-12);

    auto const Rz = Eigen::Vector2cd{ std::exp(-0.5i*theta), std::exp(0.5i*theta) }.asDiagonal().toDenseMatrix();
    EXPECT_MATRIX_CLOSE(Rz, qpp::gt.RZ(theta), 1e-12);
    EXPECT_MATRIX_CLOSE(Rz, (-0.5i * theta * qpp::gt.Z).exp().eval(), 1e-12);
    EXPECT_MATRIX_CLOSE(Rz, (cos * qpp::gt.Id2 - 1.i * sin * qpp::gt.Z).eval(), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> Rx:\n" << qpp::disp(Rx) << "\n\n";
        std::cerr << ">> Ry:\n" << qpp::disp(Ry) << "\n\n";
        std::cerr << ">> Rz:\n" << qpp::disp(Rz) << "\n\n";
    }
}

//! @brief Exercise 4.3
TEST(chapter4_2, t_rotation_z)
{
    using namespace std::numbers;
    EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(qpp::gt.T, qpp::gt.RZ(0.25 * pi), 1e-12);
}

//! @brief Exercise 4.4
TEST(chapter4_2, h_as_rotations)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    auto constexpr phi = 0.5 * pi;

    auto const H = (std::exp(phi * 1.i) * qpp::gt.RX(phi) * qpp::gt.RZ(phi) * qpp::gt.RX(phi)).eval();
    EXPECT_MATRIX_CLOSE(H, qpp::gt.H, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> H:\n" << qpp::disp(H) << "\n\n";
        std::cerr << ">> H (QPP):\n" << qpp::disp(qpp::gt.H) << "\n\n";
    }
}

//! @brief Equation 4.8 and Exercise 4.5
TEST(chapter4_2, generalized_rotations)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed();

    auto const n = Eigen::Vector3d::Random().normalized().eval();
    auto const n_dot_sigma = (n[0] * qpp::gt.X + n[1] * qpp::gt.Y + n[2] * qpp::gt.Z).eval();
    EXPECT_MATRIX_CLOSE((n_dot_sigma * n_dot_sigma).eval(), Eigen::Matrix2cd::Identity(), 1e-12);

    auto const theta = qpp::rand(0., 2.*pi);
    auto const Rtheta = std::cos(0.5*theta) * qpp::gt.Id2 - 1.i * std::sin(0.5*theta) * n_dot_sigma;

    EXPECT_MATRIX_CLOSE(Rtheta, qpp::gt.Rn(theta, {n[0], n[1], n[2]}), 1e-12);
    EXPECT_MATRIX_CLOSE(Rtheta, (-0.5i * theta * n_dot_sigma).exp().eval(), 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> Rtheta:\n" << qpp::disp(Rtheta) << "\n\n";
    }
}

//! @brief Exercise 4.6
TEST(chapter4_2, bloch_sphere_interpretation_of_rotations)
{
    using namespace std::numbers;

    qpp_e::maths::seed(63u);

    auto constexpr bloch_vector = [](auto&& psi)
    {
        auto const theta = 2. * std::acos(std::abs(psi[0]));
        auto const phi = std::arg(psi[1]) - std::arg(psi[0]);

        return (Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()) * Eigen::Vector3d::UnitZ()).eval();
    };

    auto const state = qpp::randket();
    auto const lambda = bloch_vector(state);

    auto const alpha = qpp::rand(0., 2.*pi);
    auto const n = Eigen::Vector3d::Random().normalized().eval();

    auto const lambda_rotated_state = bloch_vector((qpp::gt.Rn(alpha, {n[0], n[1], n[2]}) * state).eval());
    auto const rotated_lambda = (Eigen::AngleAxisd(alpha, n).cast<Eigen::dcomplex>() * lambda).eval();

    EXPECT_MATRIX_CLOSE(lambda_rotated_state, rotated_lambda, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> state: " << qpp::disp(state.transpose()) << "\n";
        std::cerr << ">> lambda: " << qpp::disp(lambda.transpose()) << "\n";
        std::cerr << ">> lambda_rotated_state: " << qpp::disp(lambda_rotated_state.transpose()) << "\n";
        std::cerr << ">> rotated_lambda: " << qpp::disp(rotated_lambda.transpose()) << "\n";
    }
}

//! @brief Exercise 4.7
TEST(chapter4_2, x_y_relation)
{
    using namespace std::numbers;

    qpp_e::maths::seed(82u);

    EXPECT_MATRIX_CLOSE((qpp::gt.X * qpp::gt.Y * qpp::gt.X).eval(), -qpp::gt.Y, 1e-12);
    /* Or, equivalently */
    EXPECT_MATRIX_CLOSE((qpp::gt.X * qpp::gt.Y + qpp::gt.Y * qpp::gt.X).eval(), Eigen::Matrix2cd::Zero(), 1e-12);

    auto const theta = qpp::rand(0., 2.*pi);
    EXPECT_MATRIX_CLOSE((qpp::gt.X * qpp::gt.RY(theta) * qpp::gt.X).eval(), qpp::gt.RY(-theta), 1e-12);
}

namespace
{
    //! @brief Compute rotation parameters from unitary matrix
    auto unitary_to_rotation(qpp_e::maths::Matrix auto const& U)
    {
        using namespace std::literals::complex_literals;
        using namespace std::numbers;

        /* U = exp(iH), with H hermitian */
        auto const H = Eigen::Matrix2cd{ -1.i * U.log() };

        auto const alpha_c = 0.5 * H.diagonal().sum();
        auto const alpha = std::remainder(alpha_c.real(), 2. * pi);

        auto const H_2 = (H - Eigen::Vector2cd::Constant(alpha_c).asDiagonal().toDenseMatrix()).eval();
        auto const H_22 = (H_2 * H_2).eval();
        auto const theta = 2. * std::sqrt(std::abs(H_22(0,0).real()));

        auto const n_dot_sigma = (H_2 / (-0.5 * theta)).eval();

        auto const n = Eigen::Vector3d
        {
            n_dot_sigma(1, 0).real(),
            n_dot_sigma(1, 0).imag(),
            n_dot_sigma(0, 0).real(),
        };

        return std::tuple{ alpha, theta, n };
    }

    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis>
    auto euler_angles(auto const& M)
    {
        using scalar_t = std::decay_t<decltype(M)>::Scalar;
        using system_t = Eigen::EulerSystem<_AlphaAxis, _BetaAxis, _GammaAxis>;
        using angles_t = Eigen::EulerAngles<scalar_t, system_t>;

        return angles_t{ M };
    }

    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis>
    auto euler_decomposition(qpp_e::maths::RealNumber auto const& alpha, qpp_e::maths::RealNumber auto const& theta, qpp_e::maths::Matrix auto const& n)
    {
        using namespace std::numbers;

        auto const q = Eigen::Quaterniond{ Eigen::AngleAxisd{theta, n} };
        auto const euler = euler_angles<_AlphaAxis, _BetaAxis, _GammaAxis>(q);
        auto const qq = Eigen::Quaterniond{ euler };

        auto res = Eigen::Vector4d{};
        /* Eigen Euler Angles module performs range conversion, which messes up with the "single-qubit Euler decomposition" */
        res[0] = qq.isApprox(q, 1e-12) ? alpha : (alpha + pi);
        res.tail<3>() = euler.angles();

        return res;
    }
}

//! @brief Exercise 4.8 and Equations 4.9 and 4.10
TEST(chapter4_2, unitary_matrix_as_rotation)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed(3112u);

    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto constexpr decompose = [](auto&& U)
    {
        /* U = exp(iH), with H hermitian */
        auto const H = Eigen::Matrix2cd{ -1.i * U.log() };
        EXPECT_MATRIX_CLOSE(H, H.adjoint(), 1e-12);

        auto const alpha_c = 0.5 * H.diagonal().sum();
        EXPECT_COMPLEX_CLOSE(alpha_c, alpha_c.real(), 1e-12);
        auto const alpha = std::remainder(alpha_c.real(), 2. * pi);

        auto const H_2 = (H - Eigen::Vector2cd::Constant(alpha_c).asDiagonal().toDenseMatrix()).eval();
        EXPECT_MATRIX_CLOSE(H_2, H_2.adjoint(), 1e-12);
        EXPECT_COMPLEX_CLOSE(H_2(1,1), -H_2(0,0), 1e-12);

        auto const H_22 = (H_2 * H_2).eval();
        EXPECT_TRUE(H_22.isDiagonal(1e-12));
        EXPECT_COMPLEX_CLOSE(H_22(0,0), H_22(1,1), 1e-12);
        EXPECT_COMPLEX_CLOSE(H_22(0,0), H_22(0,0).real(), 1e-12);
        EXPECT_GT(H_22(0,0).real(), -1e-12);

        auto const theta = 2. * std::sqrt(std::abs(H_22(0,0).real()));

        auto const n_dot_sigma = (H_2 / (-0.5 * theta)).eval();
        EXPECT_MATRIX_CLOSE((n_dot_sigma * n_dot_sigma), Eigen::Matrix2cd::Identity(), 1e-12);

        auto const n = Eigen::Vector3d
        {
            n_dot_sigma(1, 0).real(),
            n_dot_sigma(1, 0).imag(),
            n_dot_sigma(0, 0).real(),
        };
        EXPECT_COMPLEX_CLOSE(n.squaredNorm(), 1., 1.e-12);
        EXPECT_MATRIX_CLOSE(n_dot_sigma, (n[0] * qpp::gt.X + n[1] * qpp::gt.Y + n[2] * qpp::gt.Z).eval(), 1e-12);

        auto const rotation = (std::exp(1.i * alpha) * qpp::gt.Rn(theta, { n[0], n[1], n[2] })).eval();
        EXPECT_MATRIX_CLOSE(rotation, U, 1e-12);

        if constexpr (print_text)
        {
            std::cerr << ">> U:\n" << qpp::disp(U) << "\n\n";
            std::cerr << ">> H:\n" << qpp::disp(H) << "\n\n";
            std::cerr << ">> alpha: " << alpha << "\n\n";
            std::cerr << ">> H_2:\n" << qpp::disp(H_2) << "\n\n";
            std::cerr << ">> H_22:\n" << qpp::disp(H_22) << "\n\n";
            std::cerr << ">> theta: " << theta << "\n\n";
            std::cerr << ">> n_dot_sigma:\n" << qpp::disp(n_dot_sigma) << "\n\n";
            std::cerr << ">> n: " << qpp::disp(n.transpose()) << "\n\n";
            std::cerr << ">> rotation:\n" << qpp::disp(rotation) << "\n\n";
        }

        return std::tuple{ alpha, theta, n };
    };

    /* Part 1 */
    auto const U = qpp::randU();
    auto const [alpha_U, theta_U, n_U] = decompose(U);
    auto const [alpha_Ub, theta_Ub, n_Ub] = unitary_to_rotation(U);
    EXPECT_COMPLEX_CLOSE(alpha_U, alpha_Ub, 1e-12);
    EXPECT_COMPLEX_CLOSE(theta_U, theta_Ub, 1e-12);
    EXPECT_MATRIX_CLOSE(n_U, n_Ub, 1e-12);

    /* Part 2 */
    auto const [alpha_H, theta_H, n_H] = decompose(qpp::gt.H);
    EXPECT_COMPLEX_CLOSE(alpha_H, 0.5 * pi, 1e-12);
    EXPECT_COMPLEX_CLOSE(theta_H, pi, 1e-12);
    EXPECT_MATRIX_CLOSE(n_H, Eigen::Vector3d(inv_sqrt2, 0., inv_sqrt2), 1e-12);

    /* Part 3 */
    auto const [alpha_S, theta_S, n_S] = decompose(qpp::gt.S);
    EXPECT_COMPLEX_CLOSE(alpha_S, 0.25 * pi, 1e-12);
    EXPECT_COMPLEX_CLOSE(theta_S, 0.5 * pi, 1e-12);
    EXPECT_MATRIX_CLOSE(n_S, Eigen::Vector3d::UnitZ(), 1e-12);
}

//! @brief Theorem 4.1, Equations 4.11 and 4.12, and Exercise 4.9
TEST(chapter4_2, z_y_decomposition)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed();

    auto const U = qpp::randU();
    auto const [alpha, theta, n] = unitary_to_rotation(U);

    auto const e = euler_decomposition<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_Z>(alpha, theta, n);

    auto const rotation = (std::exp(1.i * e[0]) * qpp::gt.RZ(e[1]) * qpp::gt.RY(e[2]) * qpp::gt.RZ(e[3])).eval();
    EXPECT_MATRIX_CLOSE(rotation, U, 1e-12);

    auto const a = e[0] - 0.5 * (e[1] + e[3]);
    auto const b = e[0] + 0.5 * (-e[1] + e[3]);
    auto const c = e[0] + 0.5 * (e[1] - e[3]);
    auto const d = e[0] + 0.5 * (e[1] + e[3]);
    auto const cos = std::cos(0.5 * e[2]);
    auto const sin = std::sin(0.5 * e[2]);
    auto const UU = Eigen::Matrix2cd{
        {std::exp(1.i * a) * cos, -std::exp(1.i * b) * sin},
        {std::exp(1.i * c) * sin,  std::exp(1.i * d) * cos}
    };
    EXPECT_MATRIX_CLOSE(UU, U, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> U:\n" << qpp::disp(U) << "\n\n";
        std::cerr << ">> alpha: " << alpha << "\n\n";
        std::cerr << ">> theta: " << theta << "\n\n";
        std::cerr << ">> n: " << qpp::disp(n.transpose()) << "\n\n";
        std::cerr << ">> euler: " << qpp::disp(e.transpose()) << "\n\n";
        std::cerr << ">> rotation:\n" << qpp::disp(rotation) << "\n\n";
        std::cerr << ">> UU:\n" << qpp::disp(UU) << "\n\n";
    }
}

//! @brief Check composition of rotations
TEST(chapter4_2, rotation_composition)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed();

    auto const n1 = Eigen::Vector3d::Random().normalized().eval();
    auto const theta1 = qpp::rand(0., 2.*pi);

    auto const n2 = Eigen::Vector3d::Random().normalized().eval();
    auto const theta2 = qpp::rand(0., 2.*pi);

    auto const composed_rot = Eigen::AngleAxisd(theta1, n1) * Eigen::AngleAxisd(theta2, n2);
    auto const n = composed_rot.vec().normalized().eval();
    auto const theta = 2. * std::atan2(composed_rot.vec().norm(), composed_rot.w());

    auto const Ua = (qpp::gt.Rn(theta1, { n1[0], n1[1], n1[2] }) * qpp::gt.Rn(theta2, { n2[0], n2[1], n2[2] })).eval();
    auto const Ub = qpp::gt.Rn(theta, { n[0], n[1], n[2] });

    EXPECT_MATRIX_CLOSE(Ua, Ub, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> theta : " << theta << ", n : " << qpp::disp(n.transpose()) << "\n\n";
        std::cerr << ">> theta1: " << theta1 << ", n1: " << qpp::disp(n1.transpose()) << "\n\n";
        std::cerr << ">> theta2: " << theta2 << ", n2: " << qpp::disp(n2.transpose()) << "\n\n";

        std::cerr << ">> Ua:\n" << qpp::disp(Ua) << "\n\n";
        std::cerr << ">> Ub:\n" << qpp::disp(Ub) << "\n\n";
    }
}
