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

//! @brief Exercise 4.7 bis
//! @details Same as Exercise 4.7, with X and Z. Needed for Corollary 4.2
TEST(chapter4_2, x_z_relation)
{
    using namespace std::numbers;

    qpp_e::maths::seed(28u);

    EXPECT_MATRIX_CLOSE((qpp::gt.X * qpp::gt.Z * qpp::gt.X).eval(), -qpp::gt.Z, 1e-12);
    /* Or, equivalently */
    EXPECT_MATRIX_CLOSE((qpp::gt.X * qpp::gt.Z + qpp::gt.Z * qpp::gt.X).eval(), Eigen::Matrix2cd::Zero(), 1e-12);

    auto const theta = qpp::rand(0., 2.*pi);
    EXPECT_MATRIX_CLOSE((qpp::gt.X * qpp::gt.RZ(theta) * qpp::gt.X).eval(), qpp::gt.RZ(-theta), 1e-12);
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
    }

    //! @brief Compute Euler angles for a compatible input M (rotation matrix, unit quaternion, ...)
    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis>
    auto euler_angles(auto const& M)
    {
        using scalar_t = std::decay_t<decltype(M)>::Scalar;
        using system_t = Eigen::EulerSystem<_AlphaAxis, _BetaAxis, _GammaAxis>;
        using angles_t = Eigen::EulerAngles<scalar_t, system_t>;

        return angles_t{ M };
    }

    //! @brief Compute Euler decomposition and phase from phase, angle and unit axis
    //! @see unitary_to_rotation (output)
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

    //! @brief Compute Euler generalized decomposition of a rotation matrix R
    //! @details The unit axis of rotation are not necessarily orthogonal
    //! Such a decomposition does not always exist (see necessary and sufficient condition in the code)
    //! Return NaN if the condition is not met
    //! Computed from "On Coordinate-Free Rotation Decomposition: Euler Angles about Arbitrary Axes" by Giulia Piovan and Francesco Bullo
    auto generalized_euler_decomposition(
        qpp_e::maths::Matrix auto const& R
        , qpp_e::maths::Matrix auto const& r1
        , qpp_e::maths::Matrix auto const& r2
        , qpp_e::maths::Matrix auto const& r3)
    {
        auto theta = Eigen::Vector3d::Zero().eval();

        auto const r2xr3 = r2.cross(r3).eval();
        auto const r2xr2xr3 = r2.cross(r2xr3).eval();

        auto const a = -r1.dot(r2xr2xr3);
        auto const b = r1.dot(r2xr3);
        auto const c = r1.dot(R * r3 - r3 - r2xr2xr3);

        theta[1] = std::atan2(b, a) + std::atan2(std::sqrt(a * a + b * b - c * c), c);

        auto const collinear = qpp_e::maths::collinear((R.transpose() * r1).eval(), r3, 1e-12, true);

        if(collinear)
        {
            auto const w2 = (R.transpose() * r2).eval();
            theta[2] = -std::atan2(w2.dot(r3.cross(r2)), r2.dot(w2) - r2.dot(r3) * w2.dot(r3));
        }
        else
        {
            auto const r2xr1 = r2.cross(r1).eval();
            auto const r2xr2xr1 = r2.cross(r2xr1).eval();

            auto const cos = std::cos(theta[1]);
            auto const sin = std::sin(theta[1]);
            auto const v1 = (r3 + sin * r2xr3 + (1. - cos) * r2xr2xr3).eval();
            auto const w1 = (R * r3).eval();
            auto const v3 = (r1 - sin * r2xr1 + (1. - cos) * r2xr2xr1).eval();
            auto const w3 = (R.transpose() * r1).eval();

            theta[0] =  std::atan2(w1.dot(r1.cross(v1)), v1.dot(w1) - v1.dot(r1) * w1.dot(r1));
            theta[2] = -std::atan2(w3.dot(r3.cross(v3)), v3.dot(w3) - v3.dot(r3) * w3.dot(r3));
        }

        auto const condition_left = std::abs(r1.dot((R - r2 * r2.transpose()) * r3));
        auto const condition_right = std::sqrt((1. - std::pow(r1.dot(r2), 2)) * (1. - std::pow(r3.dot(r2), 2)));

        EXPECT_EQ((condition_left > condition_right), theta.hasNaN());

        if constexpr (print_text)
        {
            std::cerr << ">> condition_left : " << condition_left << "\n";
            std::cerr << ">> condition_right : " << condition_right << "\n";
            std::cerr << ">> condition met : " << std::boolalpha << (condition_left <= condition_right) << "\n";
            std::cerr << ">> theta : " << qpp::disp(theta.transpose()) << "\n";
        }

        return theta;
    }

}

//! @brief Exercise 4.8 and Equations 4.9 and 4.10
TEST(chapter4_2, unitary_matrix_as_rotation)
{
    using namespace std::numbers;

    qpp_e::maths::seed(3112u);

    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    /* Part 1 */
    auto const U = qpp::randU();
    unitary_to_rotation(U);

    /* Part 2 */
    auto const [alpha_H, theta_H, n_H] = unitary_to_rotation(qpp::gt.H);
    EXPECT_COMPLEX_CLOSE(alpha_H, 0.5 * pi, 1e-12);
    EXPECT_COMPLEX_CLOSE(theta_H, pi, 1e-12);
    EXPECT_MATRIX_CLOSE(n_H, Eigen::Vector3d(inv_sqrt2, 0., inv_sqrt2), 1e-12);

    /* Part 3 */
    auto const [alpha_S, theta_S, n_S] = unitary_to_rotation(qpp::gt.S);
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

//! @brief Exercise 4.10
TEST(chapter4_2, x_y_decomposition)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed();

    auto const U = qpp::randU();
    auto const [alpha, theta, n] = unitary_to_rotation(U);

    auto const e = euler_decomposition<Eigen::EULER_X, Eigen::EULER_Y, Eigen::EULER_X>(alpha, theta, n);

    auto const rotation = (std::exp(1.i * e[0]) * qpp::gt.RX(e[1]) * qpp::gt.RY(e[2]) * qpp::gt.RX(e[3])).eval();
    EXPECT_MATRIX_CLOSE(rotation, U, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> U:\n" << qpp::disp(U) << "\n\n";
        std::cerr << ">> alpha: " << alpha << "\n\n";
        std::cerr << ">> theta: " << theta << "\n\n";
        std::cerr << ">> n: " << qpp::disp(n.transpose()) << "\n\n";
        std::cerr << ">> euler: " << qpp::disp(e.transpose()) << "\n\n";
        std::cerr << ">> rotation:\n" << qpp::disp(rotation) << "\n\n";
    }
}

//! @brief Check for Angle-Axis in a different basis
TEST(chapter4_2, basis_change_angle_axis_rotation)
{
    using namespace std::numbers;

    qpp_e::maths::seed();

    auto const b_H_o = Eigen::Quaterniond::UnitRandom().toRotationMatrix().eval();
    auto const o_X = Eigen::Vector3d::Random().normalized().eval();
    auto const b_X = (b_H_o * o_X).eval();

    auto const theta = qpp::rand(0., 2 * pi);

    auto const o_R_o = Eigen::AngleAxisd(theta, o_X).toRotationMatrix();
    auto const b_R_b = (b_H_o * o_R_o * b_H_o.inverse()).eval();
    auto const b_R_b_alt = Eigen::AngleAxisd(theta, b_X).toRotationMatrix();

    EXPECT_MATRIX_CLOSE(b_R_b, b_R_b_alt, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> b_R_b:\n" << qpp::disp(b_R_b) << "\n\n";
        std::cerr << ">> b_R_b_alt:\n" << qpp::disp(b_R_b_alt) << "\n\n";
    }
}

//! @brief Exercise 4.11 and Equation 4.13
//! @details Equation 4.13 is not verified in general.
//! @see Errata of N&C: https://michaelnielsen.org/qcqi/errata/errata/errata.html, for the correct formula
//! @details However, it holds on some necessary and sufficient condition: @see generalized_euler_decomposition
TEST(chapter4_2, n_m_decomposition)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed(7981584u);

    auto const U = qpp::randU();
    auto const [alpha_U, theta_U, n_U] = unitary_to_rotation(U);

    auto const Q = Eigen::Quaterniond{ Eigen::AngleAxisd(theta_U, n_U) };
    auto const R = Q.toRotationMatrix();

    auto const n1 = Eigen::Vector3d::Random().normalized().eval();
    auto const n2 = Eigen::Vector3d::Random().normalized().eval();

    auto theta = generalized_euler_decomposition(R, n1, n2, n1);
    auto const computed_Q = Eigen::AngleAxisd(theta[0], n1) * Eigen::AngleAxisd(theta[1], n2) * Eigen::AngleAxisd(theta[2], n1);
    auto const computed_R = computed_Q.toRotationMatrix();
    EXPECT_MATRIX_CLOSE(computed_R, R, 1e-12);

    auto const alpha = Q.isApprox(computed_Q, 1e-12) ? alpha_U : (alpha_U + pi);

    auto const computed_U = (std::exp(1.i * alpha)
        * qpp::gt.Rn(theta[0], { n1[0], n1[1], n1[2]})
        * qpp::gt.Rn(theta[1], { n2[0], n2[1], n2[2]})
        * qpp::gt.Rn(theta[2], { n1[0], n1[1], n1[2]})).eval();
    EXPECT_MATRIX_CLOSE(computed_U, U, 1e-12);

    if constexpr (print_text)
    {
        std::cerr << ">> theta: " << qpp::disp(theta.transpose()) << "\n\n";
        std::cerr << ">> Q: " << Q << "\n";
        std::cerr << ">> computed_Q: " << computed_Q << "\n\n";
        std::cerr << ">> R:\n" << qpp::disp(R) << "\n\n";
        std::cerr << ">> computed_R:\n" << qpp::disp(computed_R) << "\n\n";
        std::cerr << ">> U:\n" << qpp::disp(U) << "\n\n";
        std::cerr << ">> computed_U:\n" << qpp::disp(computed_U) << "\n\n";
    }
}

namespace
{

    template <Eigen::EulerAxis _AlphaAxis, Eigen::EulerAxis _BetaAxis, Eigen::EulerAxis _GammaAxis>
    auto abc_decomposition(qpp_e::maths::Matrix auto const& U)
    {
        using namespace std::literals::complex_literals;

        auto const [alpha, theta, n] = unitary_to_rotation(U);

        auto const e = euler_decomposition<_AlphaAxis, _BetaAxis, _GammaAxis>(alpha, theta, n);

        auto const rotation = (std::exp(1.i * e[0]) * qpp::gt.RZ(e[1]) * qpp::gt.RY(e[2]) * qpp::gt.RZ(e[3])).eval();
        EXPECT_MATRIX_CLOSE(rotation, U, 1e-12);

        auto const A = (qpp::gt.RZ(e[1]) * qpp::gt.RY(0.5 * e[2])).eval();
        auto const B = (qpp::gt.RY(-0.5 * e[2]) * qpp::gt.RZ(-0.5 * (e[3]+e[1]))).eval();
        auto const C = qpp::gt.RZ(0.5 * (e[3]-e[1]));
        auto const ABC = (A * B * C).eval();
        EXPECT_MATRIX_CLOSE(ABC, Eigen::Matrix2cd::Identity(), 1e-12);

        auto const B_inverse_signs = (qpp::gt.RY(0.5 * e[2]) * qpp::gt.RZ(0.5 * (e[3]+e[1]))).eval();
        auto const XBX = (qpp::gt.X * B * qpp::gt.X).eval();
        EXPECT_MATRIX_CLOSE(XBX, B_inverse_signs, 1e-12);

        auto const AXBXC = (A * qpp::gt.X * B * qpp::gt.X * C).eval();
        auto const eiaAXBXC = (std::exp(1.i * e[0]) * AXBXC).eval();
        EXPECT_MATRIX_CLOSE(eiaAXBXC, U, 1e-12);

        if constexpr (print_text)
        {
            std::cerr << ">> U:\n" << qpp::disp(U) << "\n\n";
            std::cerr << ">> alpha: " << alpha << "\n\n";
            std::cerr << ">> euler: " << qpp::disp(e.transpose()) << "\n\n";
            std::cerr << ">> A:\n" << qpp::disp(A) << "\n\n";
            std::cerr << ">> B:\n" << qpp::disp(B) << "\n\n";
            std::cerr << ">> C:\n" << qpp::disp(C) << "\n\n";
            std::cerr << ">> ABC:\n" << qpp::disp(ABC) << "\n\n";
            std::cerr << ">> AXBXC:\n" << qpp::disp(AXBXC) << "\n\n";
            std::cerr << ">> eiaAXBXC:\n" << qpp::disp(eiaAXBXC) << "\n\n";
        }

        return std::tuple{ alpha, A, B, C };
    }

}

//! @brief Corollary 4.2 and Equations 4.14 through 4.17
TEST(chapter4_2, abc__decomposition)
{
    qpp_e::maths::seed(31385u);

    auto const U = qpp::randU();
    abc_decomposition<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_Z>(U);
}

//! @brief Exercise 4.12
TEST(chapter4_2, H_abc_decomposition)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto const [ alpha, A, B, C ] = abc_decomposition<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_Z>(qpp::gt.H);

    EXPECT_COMPLEX_CLOSE(alpha, 0.5 * pi, 1e-12);

    auto const expected_A = Eigen::Rotation2D{ 0.125 * pi }.toRotationMatrix().cast<Eigen::dcomplex>().eval();
    EXPECT_MATRIX_CLOSE(expected_A, A, 1e-12);

    auto const expected_B = (qpp::gt.RY(-0.25 * pi) * qpp::gt.RZ(-0.5 * pi)).eval();
    EXPECT_MATRIX_CLOSE(expected_B, B, 1e-12);

    auto const expected_C = Eigen::Matrix2cd
    {
        { inv_sqrt2 - 1.i * inv_sqrt2, 0. },
        { 0., inv_sqrt2 + 1.i * inv_sqrt2 }
    };
    EXPECT_MATRIX_CLOSE(expected_C, C, 1e-12);
}
