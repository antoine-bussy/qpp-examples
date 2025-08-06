#pragma once
/*!
@file
Introspection functions.
 */

#include "debug.hpp"

#include <qpp/qpp.hpp>
#include "maths/arithmetic.hpp"
#include "maths/concepts.hpp"
#include "maths/gtest_macros.hpp"


namespace qube
{

    //! @brief Extract indices of non-work qubits in a mket
    template < int NbOfOutputQubits = Eigen::Dynamic >
    auto extract_indices(unsigned long int nb_of_qubits, Eigen::VectorX<unsigned int> const& work_qubits_zero = {}
                                                  , Eigen::VectorX<unsigned int> const& work_qubits_one = {})
    {
        auto const nb_of_work_qubits_zero = work_qubits_zero.size();
        auto const nb_of_work_qubits_one = work_qubits_one.size();
        EXPECT_GE(nb_of_qubits, nb_of_work_qubits_zero + nb_of_work_qubits_one);
        auto const nb_of_non_work_qubits = nb_of_qubits - nb_of_work_qubits_zero - nb_of_work_qubits_one;

        if constexpr (NbOfOutputQubits != Eigen::Dynamic)
        {
            EXPECT_EQ(nb_of_non_work_qubits, NbOfOutputQubits);
        }

        auto mask = std::vector<qpp::idx>(nb_of_qubits, 2u);
        for(auto&& i : work_qubits_zero)
            mask[i] = 0u;
        for(auto&& i : work_qubits_one)
            mask[i] = 1u;

        auto constexpr OuputDim = (NbOfOutputQubits == Eigen::Dynamic)
                                    ? Eigen::Dynamic
                                    : qube::maths::pow(2l, static_cast<unsigned long int>(NbOfOutputQubits));
        auto const output_dim = qube::maths::pow(2l, nb_of_non_work_qubits);

        auto indices = Eigen::Vector<unsigned long int, OuputDim>::Zero(output_dim).eval();
        auto array = indices.array();

        auto J = Eigen::seqN(0, 1);

        for (auto&& i : mask)
        {
            array(J) *= 2;

            switch (i)
            {
                case 0:
                    break;
                case 1:
                    array(J) += 1;
                    break;
                case 2:
                default:
                    auto const _2J = Eigen::seqN(J.size(), J.size());
                    array(_2J) = array(J) + 1;
                    J = Eigen::seqN(0, 2 * J.size());
                    break;
            }
        }
        std::sort(indices.begin(), indices.end());

        debug() << "indices: " << qpp::disp(indices.transpose()) << "\n";

        return indices;
    }

    //! @brief Extract indices of non-work qubits in a mket
    template < int NbOfOutputQubits = Eigen::Dynamic >
    auto extract_indices_2(unsigned long int nb_of_qubits, Eigen::VectorX<unsigned int> const& work_qubits_zero = {}
                                                  , Eigen::VectorX<unsigned int> const& work_qubits_one = {})
    {
        auto const nb_of_work_qubits_zero = work_qubits_zero.size();
        auto const nb_of_work_qubits_one = work_qubits_one.size();
        EXPECT_GE(nb_of_qubits, nb_of_work_qubits_zero + nb_of_work_qubits_one);
        auto const nb_of_non_work_qubits = nb_of_qubits - nb_of_work_qubits_zero - nb_of_work_qubits_one;

        if constexpr (NbOfOutputQubits != Eigen::Dynamic)
        {
            EXPECT_EQ(nb_of_non_work_qubits, NbOfOutputQubits);
        }

        auto mask = std::vector<qpp::idx>(nb_of_qubits, 2u);
        for(auto&& i : work_qubits_zero)
            mask[i] = 0u;
        for(auto&& i : work_qubits_one)
            mask[i] = 1u;
        auto const dims = std::vector<qpp::idx>(nb_of_qubits, 2u);

        auto constexpr OuputDim = (NbOfOutputQubits == Eigen::Dynamic)
                                    ? Eigen::Dynamic
                                    : qube::maths::pow(2l, static_cast<unsigned long int>(NbOfOutputQubits));
        auto const output_dim = qube::maths::pow(2l, nb_of_non_work_qubits);

        auto indices = Eigen::Vector<unsigned long int, OuputDim>::Zero(output_dim).eval();

        auto constexpr fill_indices = [](auto&& callback, auto& mask, auto const& dims, auto& it_indices, auto const i) -> void
        {
            if(i == mask.size())
            {
                debug() << "mask: " << Eigen::VectorX<qpp::idx>::Map(mask.data(), mask.size())
                                .format(Eigen::IOFormat{ Eigen::StreamPrecision, Eigen::DontAlignCols, "", "", "", "", "", "" }) << "\n";
                *it_indices = qpp::multiidx2n(mask, dims);
                ++it_indices;
                return;
            }

            auto& b = mask[i];
            if(b != 2u)
                return callback(callback, mask, dims, it_indices, i+1);

            b = 0u;
            callback(callback, mask, dims, it_indices, i+1);
            b = 1u;
            callback(callback, mask, dims, it_indices, i+1);
            b = 2u;
        };

        auto it_indices = indices.begin();
        fill_indices(fill_indices, mask, dims, it_indices, 0u);
        EXPECT_EQ(it_indices, indices.end());

        debug() << "indices: " << qpp::disp(indices.transpose()) << "\n";

        return indices;
    }

    //! @brief Extract circuit matrix from engine
    template < int Dim = Eigen::Dynamic >
    auto extract_matrix(qpp::QEngine& engine, maths::EigenIndexer auto const& indices)
    {
        auto const total_dim = qube::maths::pow(2u, engine.get_circuit().get_nq());
        auto const dim = indices.size();
        EXPECT_EQ(total_dim % dim, 0);

        auto matrix = Eigen::Matrix<Eigen::dcomplex, Dim, Dim>::Zero(dim, dim).eval();
        auto j = 0u;
        // Use simple loop to accomodate more indices types
        for(auto i = 0u; i < dim; ++i)
        {
            auto const psi = Eigen::VectorXcd::Unit(total_dim, indices[i]);
            engine.reset(psi).execute();
            matrix.col(j) = engine.get_state()(indices, Eigen::all);
            ++j;
        }
        return matrix;
    }

    //! @brief Extract circuit matrix from engine
    template < int Dim = Eigen::Dynamic >
    auto extract_matrix(qpp::QEngine& engine, unsigned int dim = Dim)
    {
        EXPECT_GT(dim, 0);
        return extract_matrix<Dim>(engine, Eigen::seqN(0, dim));
    }

    //! @brief Extract circuit matrix from circuit
    template < int Dim = Eigen::Dynamic >
    auto extract_matrix(qpp::QCircuit const& circuit, maths::EigenIndexer auto const& indices)
    {
        auto engine = qpp::QEngine{ circuit };
        return extract_matrix<Dim>(engine, indices);
    }

    //! @brief Extract circuit matrix from circuit
    template < int Dim = Eigen::Dynamic >
    auto extract_matrix(qpp::QCircuit const& circuit, unsigned int dim = Dim)
    {
        EXPECT_NE(dim, static_cast<unsigned int>(Eigen::Dynamic));
        return extract_matrix<Dim>(circuit, Eigen::seqN(0, dim));
    }

} /* namespace qube */
