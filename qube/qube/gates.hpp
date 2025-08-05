#pragma once
/*!
@file
Custom Quantum Gates.
 */

#include <Eigen/Dense>

#include "maths/concepts.hpp"
#include "introspection.hpp"
#include <qpp/qpp.hpp>

namespace qube
{
    auto or_CTRL(maths::Matrix auto const& U)
    {
        assert(U.rows() == U.cols());

        auto const nU = static_cast<qpp::idx>(U.rows());
        auto const nqU = qpp::internal::get_num_subsys(nU, 2);
        assert(nU == maths::pow(2ul, nqU));
        auto const nq = nqU + 2ul; // 2 control qubits
        auto const n = maths::pow(2ul, nq);

        auto target = std::vector<qpp::idx>(nqU);
        std::ranges::iota(target, 2ul);

        auto circuit = qpp::QCircuit{ nq }
            .CTRL(U, { 0 }, target)
            .CTRL(U, { 1 }, target)
            .CTRL(U.adjoint(), { 0, 1 }, target);
        return extract_matrix(circuit, n);
    }

    /// @note Overkill, but generic.
    auto or_CNOT()
    {
        return or_CTRL(qpp::gt.X);
    }

} /* namespace qube */
