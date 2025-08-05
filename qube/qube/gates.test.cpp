#include <gtest/gtest.h>

#include "gates.hpp"
#include "maths/gtest_macros.hpp"
#include "maths/random.hpp"
#include <qpp/qpp.hpp>

TEST(gates_test, or_CNOT)
{
    using namespace qpp::literals;

    auto const U = qube::or_CNOT();

    EXPECT_MATRIX_CLOSE(U * U.adjoint(), Eigen::MatrixXcd::Identity(8, 8), 1e-12);

    EXPECT_EQ(U * 000_ket, 000_ket);
    EXPECT_EQ(U * 001_ket, 001_ket);
    EXPECT_EQ(U * 010_ket, 011_ket);
    EXPECT_EQ(U * 011_ket, 010_ket);
    EXPECT_EQ(U * 100_ket, 101_ket);
    EXPECT_EQ(U * 101_ket, 100_ket);
    EXPECT_EQ(U * 110_ket, 111_ket);
    EXPECT_EQ(U * 111_ket, 110_ket);
}

namespace
{
    auto test_or_CTRL(qpp::idx nq)
    {
        using namespace qpp::literals;

        qube::maths::seed();

        auto const D = qube::maths::pow(2ul, nq);
        auto const psi = qpp::randket(D);
        auto const U = qpp::randU(D);
        auto const U_CTRL = qube::or_CTRL(U);

        auto constexpr epsilon = 1e-12;

        EXPECT_MATRIX_CLOSE(U_CTRL * U_CTRL.adjoint(), Eigen::MatrixXcd::Identity(4*D, 4*D), epsilon);

        EXPECT_MATRIX_CLOSE(U_CTRL * qpp::kron(00_ket, psi), qpp::kron(00_ket,     psi), epsilon);
        EXPECT_MATRIX_CLOSE(U_CTRL * qpp::kron(01_ket, psi), qpp::kron(01_ket, U * psi), epsilon);
        EXPECT_MATRIX_CLOSE(U_CTRL * qpp::kron(10_ket, psi), qpp::kron(10_ket, U * psi), epsilon);
        EXPECT_MATRIX_CLOSE(U_CTRL * qpp::kron(11_ket, psi), qpp::kron(11_ket, U * psi), epsilon);
    }
}

TEST(gates_test, or_CTRL)
{
    test_or_CTRL(1);
}

TEST(gates_test, or_CTRL_2)
{
    test_or_CTRL(2);
}

TEST(gates_test, or_CTRL_3)
{
    test_or_CTRL(3);
}
