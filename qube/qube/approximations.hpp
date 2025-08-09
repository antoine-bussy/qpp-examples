#pragma once
/*!
@file
Approximation functions.
 */

#include "debug.hpp"

#include "maths/concepts.hpp"
#include "maths/gtest_macros.hpp"

#include <numbers>
#include <ranges>

namespace qube
{

    //! @brief Approximate an angle alpha as a integer multiple of theta.
    //! @tparam Scalar Floating-point type, expected to be at least double precision.
    template <typename Scalar = double> requires std::is_floating_point_v<Scalar>
    auto angle_approximation(
        qube::maths::RealNumber auto const& alpha_
        , qube::maths::RealNumber auto const& theta
        , qube::maths::RealNumber auto const& precision)
        -> std::tuple<Scalar, unsigned long int, Scalar>
    {
        using scalar_t = Scalar;
        auto constexpr pi = std::numbers::pi_v<scalar_t>;
        auto alpha = std::fmod(alpha_, 2. * pi);

        auto const delta = precision;

        auto constexpr check_and_cast = [](auto const& value, std::string const& name = "value")
        {
            if (value < 0.)
                throw std::invalid_argument(name + " must be positive");
            if (value > static_cast<scalar_t>(std::numeric_limits<unsigned long int>::max()))
                throw std::overflow_error(name + " would overflow unsigned long int");
            return static_cast<unsigned long int>(value);
        };

        auto const N = check_and_cast(std::ceil(2. * pi / delta) + 2., "N");
        auto const R = std::views::iota(1ul, N) |
            std::views::transform([&](auto&& k)
            {
                return std::abs(std::fmod(k * theta, 2. * pi));
            });
        auto const k = 1ul + static_cast<unsigned long int>(std::ranges::distance(R.cbegin(), std::ranges::min_element(R)));
        auto const theta_k = std::fmod(k * theta, 2. * pi);

        EXPECT_LT(std::abs(theta_k), delta);
        EXPECT_NE(theta_k, 0.);

        if (theta_k < 0.)
            alpha -= 2. * pi;

        auto m = check_and_cast(std::floor(alpha/theta_k), "m");
        // If alpha is too small, m can be zero, which is not useful for approximation.
        // Choose m = 1 means that we approximate alpha as theta_k.
        if (m == 0ul)
            m = 1ul;

        auto const mk = check_and_cast(m * k, "m * k");
        auto const approx = std::fmod(mk * theta, 2. * pi);

        debug() << ">> alpha: " << alpha << "\n";
        debug() << ">> approx: " << approx << "\n";
        debug() << ">> k: " << k << "\n";
        debug() << ">> m: " << m << "\n";
        debug() << ">> mk: " << mk << "\n";
        debug() << ">> theta_k: " << theta_k << "\n";

        return { approx, mk, theta_k };
    }

} /* namespace qube */
