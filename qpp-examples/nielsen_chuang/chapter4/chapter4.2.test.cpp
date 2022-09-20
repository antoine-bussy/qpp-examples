#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>
#include <qpp-examples/qube/debug.hpp>
#include <qpp-examples/qube/decompositions.hpp>

#include <unsupported/Eigen/MatrixFunctions>

#include <numbers>

using namespace qpp_e::qube::stream;

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

    debug() << ">> I:\n" << qpp::disp(I) << "\n\n";
    debug() << ">> X:\n" << qpp::disp(X) << "\n\n";
    debug() << ">> Y:\n" << qpp::disp(Y) << "\n\n";
    debug() << ">> Z:\n" << qpp::disp(Z) << "\n\n";
    debug() << ">> H:\n" << qpp::disp(H) << "\n\n";
    debug() << ">> S:\n" << qpp::disp(S) << "\n\n";
    debug() << ">> T:\n" << qpp::disp(T) << "\n\n";
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

    debug() << ">> X: " << qpp::disp(lambda_X.transpose()) << "\n" << qpp::disp(v_X) << "\n\n";
    debug() << ">> Y: " << qpp::disp(lambda_Y.transpose()) << "\n" << qpp::disp(v_Y) << "\n\n";
    debug() << ">> Z: " << qpp::disp(lambda_Z.transpose()) << "\n" << qpp::disp(v_Z) << "\n\n";
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

    debug() << ">> Rx:\n" << qpp::disp(Rx) << "\n\n";
    debug() << ">> Ry:\n" << qpp::disp(Ry) << "\n\n";
    debug() << ">> Rz:\n" << qpp::disp(Rz) << "\n\n";
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

    debug() << ">> H:\n" << qpp::disp(H) << "\n\n";
    debug() << ">> H (QPP):\n" << qpp::disp(qpp::gt.H) << "\n\n";
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

    debug() << ">> Rtheta:\n" << qpp::disp(Rtheta) << "\n\n";
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

    debug() << ">> state: " << qpp::disp(state.transpose()) << "\n";
    debug() << ">> lambda: " << qpp::disp(lambda.transpose()) << "\n";
    debug() << ">> lambda_rotated_state: " << qpp::disp(lambda_rotated_state.transpose()) << "\n";
    debug() << ">> rotated_lambda: " << qpp::disp(rotated_lambda.transpose()) << "\n";
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

//! @brief Exercise 4.8 and Equations 4.9 and 4.10
TEST(chapter4_2, unitary_matrix_as_rotation)
{
    using namespace std::numbers;

    qpp_e::maths::seed(3112u);

    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    /* Part 1 */
    auto const U = qpp::randU();
    qpp_e::qube::unitary_to_rotation(U);

    /* Part 2 */
    auto const [alpha_H, theta_H, n_H] = qpp_e::qube::unitary_to_rotation(qpp::gt.H);
    EXPECT_COMPLEX_CLOSE(alpha_H, 0.5 * pi, 1e-12);
    EXPECT_COMPLEX_CLOSE(theta_H, pi, 1e-12);
    EXPECT_MATRIX_CLOSE(n_H, Eigen::Vector3d(inv_sqrt2, 0., inv_sqrt2), 1e-12);

    /* Part 3 */
    auto const [alpha_S, theta_S, n_S] = qpp_e::qube::unitary_to_rotation(qpp::gt.S);
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
    auto const [alpha, theta, n] = qpp_e::qube::unitary_to_rotation(U);

    auto const e = qpp_e::qube::euler_decomposition<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_Z>(alpha, theta, n);

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

    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
    debug() << ">> alpha: " << alpha << "\n\n";
    debug() << ">> theta: " << theta << "\n\n";
    debug() << ">> n: " << qpp::disp(n.transpose()) << "\n\n";
    debug() << ">> euler: " << qpp::disp(e.transpose()) << "\n\n";
    debug() << ">> rotation:\n" << qpp::disp(rotation) << "\n\n";
    debug() << ">> UU:\n" << qpp::disp(UU) << "\n\n";
}

//! @brief Exercise 4.10
TEST(chapter4_2, x_y_decomposition)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed();

    auto const U = qpp::randU();
    auto const [alpha, theta, n] = qpp_e::qube::unitary_to_rotation(U);

    auto const e = qpp_e::qube::euler_decomposition<Eigen::EULER_X, Eigen::EULER_Y, Eigen::EULER_X>(alpha, theta, n);

    auto const rotation = (std::exp(1.i * e[0]) * qpp::gt.RX(e[1]) * qpp::gt.RY(e[2]) * qpp::gt.RX(e[3])).eval();
    EXPECT_MATRIX_CLOSE(rotation, U, 1e-12);

    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
    debug() << ">> alpha: " << alpha << "\n\n";
    debug() << ">> theta: " << theta << "\n\n";
    debug() << ">> n: " << qpp::disp(n.transpose()) << "\n\n";
    debug() << ">> euler: " << qpp::disp(e.transpose()) << "\n\n";
    debug() << ">> rotation:\n" << qpp::disp(rotation) << "\n\n";
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

    debug() << ">> b_R_b:\n" << qpp::disp(b_R_b) << "\n\n";
    debug() << ">> b_R_b_alt:\n" << qpp::disp(b_R_b_alt) << "\n\n";
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
    auto const [alpha_U, theta_U, n_U] = qpp_e::qube::unitary_to_rotation(U);

    auto const Q = Eigen::Quaterniond{ Eigen::AngleAxisd(theta_U, n_U) };
    auto const R = Q.toRotationMatrix();

    auto const n1 = Eigen::Vector3d::Random().normalized().eval();
    auto const n2 = Eigen::Vector3d::Random().normalized().eval();

    auto const theta = qpp_e::qube::generalized_euler_decomposition(alpha_U, theta_U, n_U, n1, n2, n1);
    auto const computed_Q = Eigen::AngleAxisd(theta[1], n1) * Eigen::AngleAxisd(theta[2], n2) * Eigen::AngleAxisd(theta[3], n1);
    auto const computed_R = computed_Q.toRotationMatrix();
    EXPECT_MATRIX_CLOSE(computed_R, R, 1e-12);

    auto const& alpha = theta[0];

    auto const computed_U = (std::exp(1.i * alpha)
        * qpp::gt.Rn(theta[1], { n1[0], n1[1], n1[2]})
        * qpp::gt.Rn(theta[2], { n2[0], n2[1], n2[2]})
        * qpp::gt.Rn(theta[3], { n1[0], n1[1], n1[2]})).eval();
    EXPECT_MATRIX_CLOSE(computed_U, U, 1e-12);

    debug() << ">> theta: " << qpp::disp(theta.transpose()) << "\n\n";
    debug() << ">> Q: " << Q << "\n";
    debug() << ">> computed_Q: " << computed_Q << "\n\n";
    debug() << ">> R:\n" << qpp::disp(R) << "\n\n";
    debug() << ">> computed_R:\n" << qpp::disp(computed_R) << "\n\n";
    debug() << ">> U:\n" << qpp::disp(U) << "\n\n";
    debug() << ">> computed_U:\n" << qpp::disp(computed_U) << "\n\n";
}

//! @brief Corollary 4.2 and Equations 4.14 through 4.17
TEST(chapter4_2, abc_decomposition)
{
    qpp_e::maths::seed(31385u);

    auto const U = qpp::randU();
    qpp_e::qube::abc_decomposition<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_Z>(U);
}

//! @brief Exercise 4.12
TEST(chapter4_2, H_abc_decomposition)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    auto constexpr inv_sqrt2 = 0.5 * sqrt2;

    auto const [ alpha, A, B, C ] = qpp_e::qube::abc_decomposition<Eigen::EULER_Z, Eigen::EULER_Y, Eigen::EULER_Z>(qpp::gt.H);

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

//! @brief Exercises 4.13 and 4.14
TEST(chapter4_2, circuit_identites)
{
    using namespace std::numbers;

    auto const HXH = (qpp::gt.H * qpp::gt.X * qpp::gt.H).eval();
    EXPECT_MATRIX_CLOSE(HXH, qpp::gt.Z, 1e-12);

    auto const _HYH = (-qpp::gt.H * qpp::gt.Y * qpp::gt.H).eval();
    EXPECT_MATRIX_CLOSE(_HYH, qpp::gt.Y, 1e-12);

    auto const HZH = (qpp::gt.H * qpp::gt.Z * qpp::gt.H).eval();
    EXPECT_MATRIX_CLOSE(HZH, qpp::gt.X, 1e-12);

    auto const HTH = (qpp::gt.H * qpp::gt.T * qpp::gt.H).eval();
    EXPECT_MATRIX_CLOSE_UP_TO_PHASE_FACTOR(HTH, qpp::gt.RX(0.25 * pi), 1e-12);

    debug() << ">> HXH:\n" << qpp::disp(HXH) << "\n\n";
    debug() << ">> -HYH:\n" << qpp::disp(_HYH) << "\n\n";
    debug() << ">> HZH:\n" << qpp::disp(HZH) << "\n\n";
    debug() << ">> HTH:\n" << qpp::disp(HTH) << "\n\n";
    debug() << ">> Rx(pi/4):\n" << qpp::disp(qpp::gt.RX(0.25 * pi)) << "\n\n";
}

namespace
{
    auto single_qubit_operation_composition(qpp_e::maths::RealNumber auto const& beta1, qpp_e::maths::Matrix auto const& n1
                                        , qpp_e::maths::RealNumber auto const& beta2, qpp_e::maths::Matrix auto const& n2)
    {
        auto const composed_rot = Eigen::AngleAxisd(beta1, n1) * Eigen::AngleAxisd(beta2, n2);
        auto const n12 = composed_rot.vec().normalized().eval();
        auto const beta12 = 2. * std::atan2(composed_rot.vec().norm(), composed_rot.w());

        auto const Ua = (qpp::gt.Rn(beta1, { n1[0], n1[1], n1[2] }) * qpp::gt.Rn(beta2, { n2[0], n2[1], n2[2] })).eval();
        auto const Ub = qpp::gt.Rn(beta12, { n12[0], n12[1], n12[2] });

        EXPECT_MATRIX_CLOSE(Ua, Ub, 1e-12);

        debug() << ">> beta12: " << beta12 << ", n12: " << qpp::disp(n12.transpose()) << "\n\n";
        debug() << ">> beta1: " << beta1 << ", n1:" << qpp::disp(n1.transpose()) << "\n\n";
        debug() << ">> beta2: " << beta2 << ", n2:" << qpp::disp(n2.transpose()) << "\n\n";

        debug() << ">> Ua:\n" << qpp::disp(Ua) << "\n\n";
        debug() << ">> Ub:\n" << qpp::disp(Ub) << "\n\n";

        return std::tuple{ beta12, n12 };
    }
}

//! @brief Exercise 4.15 and Equations 4.19 through 4.22
TEST(chapter4_2, composition_of_single_qubit_operations)
{
    using namespace std::literals::complex_literals;
    using namespace std::numbers;

    qpp_e::maths::seed();

    /* Part 1 */
    auto const n1 = Eigen::Vector3d::Random().normalized().eval();
    auto const beta1 = qpp::rand(0., 2.*pi);

    auto const n2 = Eigen::Vector3d::Random().normalized().eval();
    auto const beta2 = qpp::rand(0., 2.*pi);

    auto const [ beta12, n12 ] = single_qubit_operation_composition(beta1, n1, beta2, n2);

    auto const c1 = std::cos(0.5 * beta1);
    auto const s1 = std::sin(0.5 * beta1);
    auto const c2 = std::cos(0.5 * beta2);
    auto const s2 = std::sin(0.5 * beta2);
    auto const c12 = std::cos(0.5 * beta12);
    auto const s12 = std::sin(0.5 * beta12);

    auto const computed_c12 = c1 * c2 - s1 * s2 * n1.dot(n2);
    auto const computed_s12_n12 = ((s1 * c2) * n1 + (c1 * s2) * n2 - s1 * s2 * n2.cross(n1)).eval();

    EXPECT_COMPLEX_CLOSE(computed_c12, c12, 1e-12);
    EXPECT_MATRIX_CLOSE(computed_s12_n12, (s12 * n12).eval(), 1e-12);

    /* Part 2 */
    auto const beta = qpp::rand(0., 2.*pi);
    auto const z = Eigen::Vector3d::UnitZ();
    auto const n = Eigen::Vector3d::Random().normalized().eval();

    auto const [ betaZ, nZ ] = single_qubit_operation_composition(beta, z, beta, n);

    auto const c = std::cos(0.5 * beta);
    auto const s = std::sin(0.5 * beta);
    auto const cZ = std::cos(0.5 * betaZ);
    auto const sZ = std::sin(0.5 * betaZ);

    auto const computed_cZ = c * c  - s * s * z.dot(n);
    auto const computed_sZ_nZ = (s * c * (z + n) - s * s * n.cross(z)).eval();

    EXPECT_COMPLEX_CLOSE(computed_cZ, cZ, 1e-12);
    EXPECT_MATRIX_CLOSE(computed_sZ_nZ, (sZ * nZ).eval(), 1e-12);
}
