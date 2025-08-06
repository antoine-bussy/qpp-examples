#pragma once
/*!
@file
Maths concepts.
 */

#include <concepts>
#include <Eigen/Core>

namespace qube::maths
{
    template < class Derived >
    concept Matrix = requires(Derived m)
    {
        []<class D>(Eigen::MatrixBase<D> const&){}(m);
    };

    template < class T >
    concept RealNumber = std::is_arithmetic_v<T>;

    template < class T >
    concept ComplexNumber = requires (T c)
    {
        std::real(c);
        std::imag(c);
    };

    template < class T >
    concept arithmetic = std::is_arithmetic_v<T>;

    template < typename T >
    concept EigenIndexer = requires(T t)
    {
        requires std::convertible_to<decltype(t[Eigen::Index{0}]), Eigen::Index>;
        requires std::convertible_to<decltype(t.size()), Eigen::Index>;
    };

} /* namespace qube::maths */
