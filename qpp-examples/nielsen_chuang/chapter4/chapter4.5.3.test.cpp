#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ranges>

#include <qpp/qpp.hpp>
#include <qube/debug.hpp>
#include <qube/maths/arithmetic.hpp>
#include <qube/maths/norm.hpp>
#include <qube/maths/random.hpp>

using namespace qube::stream;

//! @brief Equation 4.61
TEST(chapter4_5, operator_error)
{
    qube::maths::seed();

    auto constexpr nq = 3ul;
    auto constexpr n = qube::maths::pow(2ul, nq);

    auto const U = qpp::randU(n);
    auto const V = qpp::randU(n);
    auto const W = (U - V).eval();

    auto const error = qube::maths::operator_norm_2(W);
    auto const psi_max = W.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeFullV).matrixV().col(0).eval();
    auto const e_max = qpp::norm(W * psi_max);
    auto constexpr epsilon = 1e-12;
    EXPECT_LT(e_max, error + epsilon);

    for ([[maybe_unused]] auto&& i: std::views::iota(0, 50))
    {
        auto const psi = qpp::randket(n);
        auto const e = (W * psi).norm();
        EXPECT_LT(e, error + epsilon);
    }
}
