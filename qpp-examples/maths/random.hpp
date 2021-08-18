#pragma once
/*!
@file
Randomness functions.
 */

#include <qpp/qpp.h>

namespace qpp_e::maths
{
    template < class Sseq >
    inline auto qpp_seed(Sseq& seq)
    {
        return qpp::RandomDevices::get_instance().get_prng().seed(seq);
    }

    inline auto seed(unsigned int s)
    {
        std::srand(s);
        return qpp_seed(s);
    }

} /* namespace qpp_e::maths */
