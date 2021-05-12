#pragma once
/*!
@file
Maths concepts.
 */

#include <concepts>
#include <Eigen/Core>

namespace qpp_e::maths
{
    template < class Derived >
    concept Matrix = std::derived_from<Derived, Eigen::MatrixBase<Derived>>;

    template < class T >
    concept RealNumber = std::is_arithmetic_v<T>;

    template < class T >
    concept ComplexNumber = requires (T c)
    {
        std::real(c);
        std::imag(c);
    };

} /* namespace qpp_e::maths */
