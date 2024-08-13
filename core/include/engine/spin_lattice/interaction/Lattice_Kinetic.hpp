#pragma once

#include <engine/Indexing.hpp>
#include <engine/spin_lattice/StateType.hpp>
#include <engine/spin_lattice/interaction/Functor_Prototypes.hpp>

#include <Eigen/Dense>

namespace Engine
{

namespace SpinLattice
{

namespace Interaction
{

struct Lattice_Kinetic
{
    using state_t = StateType;

    struct Data
    {
    };

    static bool valid_data( const Data & )
    {
        return true;
    }

    struct Cache
    {
        const ::Data::Geometry * geometry;
    };

    static bool is_contributing( const Data &, const Cache & )
    {
        return true;
    };

    // clang-tidy: ignore
    typedef int IndexType;

    using Index        = const IndexType *;
    using IndexStorage = Backend::optional<IndexType>;

    using Energy = Functor::Local::Energy_Functor<Functor::Local::DataRef<Lattice_Kinetic>>;

    template<Field field>
    using Gradient = Functor::Local::Gradient_Functor<field, Functor::Local::DataRef<Lattice_Kinetic>>;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Lattice Kinetic";

    template<typename IndexStorageVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield &, const Data &, Cache & cache, IndexStorageVector & indices )
    {
        using Indexing::check_atom_type;

        const auto N = geometry.n_cell_atoms;

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int ibasis = 0; ibasis < N; ++ibasis )
            {
                const int ispin = icell * N + ibasis;
                if( check_atom_type( geometry.atom_types[ispin] ) )
                {
                    Backend::get<IndexStorage>( indices[ispin] ) = ispin;
                }
            };
        }

        cache.geometry = &geometry;
    };
};

template<>
struct Functor::Local::DataRef<Lattice_Kinetic>
{
    using Interaction = Lattice_Kinetic;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              inverse_mass( cache.geometry->inverse_mass.data() ) {};

    const bool is_contributing;

protected:
    const scalar * inverse_mass;
};

template<>
inline scalar Lattice_Kinetic::Energy::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    if( is_contributing && index != nullptr )
    {
        const auto ispin = *index;

        return 0.5 * inverse_mass[ispin] * state.momentum[ispin].squaredNorm();
    }
    return 0.0;
}

template<>
inline Vector3
Lattice_Kinetic::Gradient<Field::Momentum>::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    if( is_contributing && index != nullptr )
    {
        const auto ispin = *index;

        return inverse_mass[ispin] * state.momentum[ispin];
    }
    return Vector3::Zero();
}

} // namespace Interaction

} // namespace SpinLattice

} // namespace Engine
