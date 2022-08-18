#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/random.hpp>

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
