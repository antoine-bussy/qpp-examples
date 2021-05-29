#pragma once
/*!
@file
Arithmetic functions.
 */

#include "concepts.hpp"
#include <cmath>

namespace qpp_e::maths
{

    inline auto constexpr pow(arithmetic auto const& base, std::integral auto const& exp) -> std::decay_t<decltype(base)>
    {
        assert(exp >= 0);
        if (base == 2)
            return (1 << exp);
        else if (base == -1)
            return (exp % 2 == 0) ? 1 : -1;
        else
            return std::pow(base, exp);
    }

} /* namespace qpp_e::maths */
