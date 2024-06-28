#pragma once

#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Span.hpp>
#include <engine/spin_lattice/interaction/Functor_Prototypes.hpp>

#include <Eigen/Dense>

namespace Engine
{

namespace SpinLattice
{

namespace Interaction
{

struct Lattice_Harmonic_Potential
{
    using state_t = StateType;

    struct Data
    {
        pairfield pairs{};
        vectorfield normals;
        scalarfield magnitudes;

        Data() = default;
        Data( pairfield pairs, vectorfield normals, scalarfield magnitudes )
                : pairs( std::move( pairs ) ),
                  normals( std::move( normals ) ),
                  magnitudes( std::move( magnitudes ) ) {};
    };

    static bool valid_data( const Data & data )
    {
        return data.pairs.empty()
               || ( 3 * data.pairs.size() == data.magnitudes.size() && 3 * data.pairs.size() == data.normals.size() );
    };

    struct Cache
    {
    };

    static bool is_contributing( const Data & data, const Cache & ) noexcept
    {
        return !data.pairs.empty();
    }

    struct IndexType
    {
        int ispin, jspin, ipair;
    };

    using Index        = Engine::Span<const IndexType>;
    using IndexStorage = Backend::vector<IndexType>;

    using Energy = Functor::Local::Energy_Functor<Functor::Local::DataRef<Lattice_Harmonic_Potential>>;

    template<Field field>
    using Gradient = Functor::Local::Gradient_Functor<field, Functor::Local::DataRef<Lattice_Harmonic_Potential>>;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "Lattice Harmonic Potential";

    template<typename IndexStorageVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache &,
        IndexStorageVector & indices )
    {
        using Indexing::idx_from_pair;

        for( unsigned int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( unsigned int i_pair = 0; i_pair < data.pairs.size(); ++i_pair )
            {
                int ispin = data.pairs[i_pair].i + icell * geometry.n_cell_atoms;
                int jspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    data.pairs[i_pair] );
                if( jspin >= 0 )
                {
                    Backend::get<IndexStorage>( indices[ispin] ).push_back( IndexType{ ispin, jspin, (int)i_pair } );
                    Backend::get<IndexStorage>( indices[jspin] ).push_back( IndexType{ jspin, ispin, (int)i_pair } );
                }
            }
        }
    };
};

template<>
struct Functor::Local::DataRef<Lattice_Harmonic_Potential>
{
    using Interaction = Lattice_Harmonic_Potential;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              normals( data.normals.data() ),
              magnitudes( data.magnitudes.data() )
    {
    }

    const bool is_contributing;

protected:
    const Vector3 * normals;
    const scalar * magnitudes;
};

template<>
inline scalar
Lattice_Harmonic_Potential::Energy::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    // don't need to check for `is_contributing` here, because the `transform_reduce` will short circuit correctly
    return 0.5
           * Backend::transform_reduce(
               index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
               [this, state] SPIRIT_LAMBDA( const Lattice_Harmonic_Potential::IndexType & idx ) -> scalar
               {
                   const auto & [ispin, jspin, iani] = idx;
                   scalar result                     = 0;
                   for( int alpha = 0; alpha < 3; ++alpha )
                   {
                       const int cidx = 3 * iani + alpha;
                       result += magnitudes[cidx] * normals[cidx].dot( state.displacement[ispin] )
                                 * normals[cidx].dot( state.displacement[jspin] );
                   }
                   return result;
               } );
}

template<>
inline Vector3 Lattice_Harmonic_Potential::Gradient<Field::Displacement>::operator()(
    const Index & index, quantity<const Vector3 *> state ) const
{
    // don't need to check for `is_contributing` here, because the `transform_reduce` will short circuit correctly
    return Backend::transform_reduce(
        index.begin(), index.end(), zero_value<Vector3>(), Backend::plus<Vector3>{},
        [this, state] SPIRIT_LAMBDA( const Lattice_Harmonic_Potential::IndexType & idx ) -> Vector3
        {
            const auto & [ispin, jspin, iani] = idx;
            Vector3 result                    = Vector3::Zero();
            for( int alpha = 0; alpha < 3; ++alpha )
            {
                const int cidx = 3 * iani + alpha;
                result += normals[cidx] * magnitudes[cidx] * normals[cidx].dot( state.displacement[jspin] );
            }
            return result;
        } );
}

} // namespace Interaction

} // namespace SpinLattice

} // namespace Engine
