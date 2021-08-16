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
        auto& gen =
#ifdef NO_THREAD_LOCAL_
            qpp::RandomDevices::get_instance().get_prng();
#else
            qpp::RandomDevices::get_thread_local_instance().get_prng();
#endif
        return gen.seed(seq);
    }

    inline auto seed(unsigned int s)
    {
        std::srand(s);
        return qpp_seed(s);
    }

} /* namespace qpp_e::maths */
