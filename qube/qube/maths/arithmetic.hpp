#pragma once
/*!
@file
Arithmetic functions.
 */

#include "concepts.hpp"
#include <cmath>

namespace qube::maths
{

    inline auto constexpr pow(arithmetic auto const& base, std::integral auto const& exp) -> std::decay_t<decltype(base)>
    {
        using base_t = std::decay_t<decltype(base)>;
        assert(exp >= 0);
        if constexpr (std::is_signed_v<base_t>)
            if (base == -1)
                return (exp % 2 == 0) ? 1 : -1;

        if (base == 2)
            return (1 << exp);

        return std::pow(base, exp);
    }

} /* namespace qube::maths */
