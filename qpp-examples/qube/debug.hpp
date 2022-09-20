#pragma once
/*!
@file
Debug functions.
 */

#include <iostream>

namespace qpp_e::qube
{
    struct null_stream_t
    {
        //! @brief Generic do-nothing streaming operator
        null_stream_t& operator<<(auto&&) { return *this; }
        null_stream_t& operator<<(std::ostream&(*f)(std::ostream&)) { std::cerr << f; return *this; }
    };

    struct err_stream_t
    {
        //! @brief Error streaming operator
        err_stream_t& operator<<(auto&& t) { std::cerr << t; return *this; }
        err_stream_t& operator<<(std::ostream&(*f)(std::ostream&)) { std::cerr << f; return *this; }
    };

    inline auto consteval debug()
    {
#if QPP_E_DEBUG_STREAM
        return err_stream_t{};
#else
        return null_stream_t{};
#endif
    }

    namespace stream
    {

        using qpp_e::qube::debug;

    } /* namespace stream */

} /* namespace qpp_e::qube */
