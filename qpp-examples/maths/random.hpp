#pragma once
/*!
@file
Randomness functions.
 */

#include <qpp/qpp.hpp>

namespace qube::maths
{
    template < class Sseq >
    inline auto qpp_seed(Sseq& seq)
    {
        return qpp::RandomDevices::get_instance().get_prng().seed(seq);
    }

    inline auto seed(unsigned int s = std::time(0))
    {
        std::srand(s);
        return qpp_seed(s);
    }

} /* namespace qube::maths */
