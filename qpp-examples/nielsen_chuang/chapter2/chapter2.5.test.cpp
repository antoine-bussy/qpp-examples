#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <qpp/qpp.h>
#include <qpp-examples/maths/gtest_macros.hpp>
#include <qpp-examples/maths/arithmetic.hpp>
#include <qpp-examples/maths/random.hpp>

#include <numbers>

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
    EXPECT_MATRIX_CLOSE(eigsA(Eigen::lastN(_2_pow_m)), eigsB, 1e-12);

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
