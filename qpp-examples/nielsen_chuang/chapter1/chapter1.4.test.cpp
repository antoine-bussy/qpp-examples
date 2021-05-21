#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>

#include <numbers>

namespace
{
    auto constexpr print_text = false;
}

//! @brief Test of Q++ mket function
TEST(chapter1_4, mket)
{
    using namespace qpp::literals;
    
    EXPECT_MATRIX_EQ(qpp::mket({0, 0, 0}), 000_ket);
    EXPECT_MATRIX_EQ(qpp::mket({0, 0, 1}), 001_ket);
    EXPECT_MATRIX_EQ(qpp::mket({0, 1, 0}), 010_ket);
    EXPECT_MATRIX_EQ(qpp::mket({0, 1, 1}), 011_ket);

    EXPECT_MATRIX_EQ(qpp::mket({1, 0, 0}), 100_ket);
    EXPECT_MATRIX_EQ(qpp::mket({1, 0, 1}), 101_ket);
    EXPECT_MATRIX_EQ(qpp::mket({1, 1, 0}), 110_ket);
    EXPECT_MATRIX_EQ(qpp::mket({1, 1, 1}), 111_ket);
}

//! @brief Figure 1.14
TEST(chapter1_4, toffoli_gate)
{
    using namespace qpp::literals;
    
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 000_ket, 000_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 001_ket, 001_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 010_ket, 010_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 011_ket, 011_ket);

    EXPECT_MATRIX_EQ(qpp::gt.TOF * 100_ket, 100_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 101_ket, 101_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 110_ket, 111_ket);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * 111_ket, 110_ket);

    for(auto&& a : { 0u, 1u })
        for(auto&& b : { 0u, 1u })
            for(auto&& c : { 0u, 1u })
                EXPECT_MATRIX_EQ(qpp::gt.TOF * qpp::mket({a, b, c}), qpp::mket({a, b, (c + a * b) % 2}));

    auto toffoli_matrix = Eigen::Matrix<qpp::cplx, 8, 8>::Identity().eval();
    toffoli_matrix(Eigen::lastN(2), Eigen::lastN(2)) = Eigen::Vector4cd{ 0, 1, 1, 0 }.reshaped(2,2);
    EXPECT_MATRIX_EQ(qpp::gt.TOF, toffoli_matrix);
    EXPECT_MATRIX_EQ(qpp::gt.TOF * qpp::gt.TOF.adjoint(), (Eigen::Matrix<qpp::cplx, 8, 8>::Identity()));

    if constexpr (print_text)
        std::cerr << ">> Toffoli gate:\n" << qpp::disp(qpp::gt.TOF) << '\n';
}

//! @brief Figure 1.15
TEST(chapter1_4, nand)
{
    for(auto&& a : { 0u, 1u })
        for(auto&& b : { 0u, 1u })
            EXPECT_MATRIX_EQ(qpp::gt.TOF * qpp::mket({a, b, 1}), qpp::mket({a, b, !(a && b)}));
}

//! @brief Figure 1.16
TEST(chapter1_4, fanout)
{
    for(auto&& a : { 0u, 1u })
        EXPECT_MATRIX_EQ(qpp::gt.TOF * qpp::mket({1, a, 0}), qpp::mket({1, a, a}));
}

namespace
{
    auto consteval functions()
    {
        return std::array
        {
            std::array{ 0u, 0u },
            std::array{ 0u, 1u },
            std::array{ 1u, 0u },
            std::array{ 1u, 1u },
        };
    }

    auto matrix(auto const& f)
    {
        auto F = Eigen::Matrix2cd::Zero().eval();
        F(f[0], 0) = F(f[1], 1) = 1;
        return F;
    }

    auto matrixU(auto const& F)
    {
        using namespace qpp::literals;

        auto Uf = Eigen::Matrix4cd::Zero().eval();
        Uf.col(0) = qpp::kron(0_ket, F * 0_ket);
        Uf.col(1) = qpp::kron(0_ket, qpp::gt.X * F * 0_ket);
        Uf.col(2) = qpp::kron(1_ket, F * 1_ket);
        Uf.col(3) = qpp::kron(1_ket, qpp::gt.X * F * 1_ket);
        return Uf;
    }
}

//! @brief Figure 1.17
TEST(chapter1_4, function)
{
    using namespace qpp::literals;

    for(auto&& f : functions())
    {
        auto const F = matrix(f);
        EXPECT_MATRIX_EQ(F * 0_ket, qpp::mket({f[0]}));
        EXPECT_MATRIX_EQ(F * 1_ket, qpp::mket({f[1]}));

        auto const Uf = matrixU(F);
        EXPECT_MATRIX_EQ(Uf * Uf.adjoint(), Eigen::Matrix4cd::Identity());

        EXPECT_MATRIX_EQ(Uf * 00_ket, qpp::mket({0u, f[0]}));
        EXPECT_MATRIX_EQ(Uf * 01_ket, qpp::mket({0u, (1u + f[0]) % 2u}));
        EXPECT_MATRIX_EQ(Uf * 10_ket, qpp::mket({1u, f[1]}));
        EXPECT_MATRIX_EQ(Uf * 11_ket, qpp::mket({1u, (1u + f[1]) % 2u}));

        if constexpr (print_text)
        {
            std::cerr << "-----------------------------\n";
            std::cerr << ">> F:\n" << qpp::disp(F) << '\n';
            std::cerr << ">> Uf:\n" << qpp::disp(Uf) << '\n';
        }
    }
}

//! @brief Figure 1.17 and equation 1.37
TEST(chapter1_4, function_parallelism)
{
    using namespace qpp::literals;
    auto constexpr inv_sqrt2 = 0.5 * std::numbers::sqrt2;

    for(auto&& f : functions())
    {
        auto const Uf = matrixU(matrix(f));

        auto const x = ((0_ket + 1_ket) * inv_sqrt2).eval();
        auto const y = 0_ket;

        auto const psi = (Uf * qpp::kron(x, y)).eval();
        auto const expected_psi = ((qpp::mket({0u, f[0]}) + qpp::mket({1u, f[1]})) * inv_sqrt2).eval();
        EXPECT_MATRIX_EQ(psi, expected_psi);

        if constexpr (print_text)
        {
            std::cerr << "-----------------------------\n";
            std::cerr << ">> Uf:\n" << qpp::disp(Uf) << '\n';
        }
    }
}
