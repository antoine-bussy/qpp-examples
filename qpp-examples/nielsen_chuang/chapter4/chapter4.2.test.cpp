#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>

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
