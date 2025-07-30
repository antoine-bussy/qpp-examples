#pragma once
/*!
@file
Norm functions.
 */

#include "concepts.hpp"
#include <Eigen/SVD>

namespace qube::maths
{

    //! @brief Compute the 2-norm of a matrix
    inline auto operator_norm_2(Matrix auto const& m)
    {
        return m.jacobiSvd().singularValues()[0];
    }

} /* namespace qube::maths */
