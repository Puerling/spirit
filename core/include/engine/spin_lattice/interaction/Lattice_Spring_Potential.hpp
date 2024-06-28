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

struct Lattice_Spring_Potential
{
    using state_t = StateType;

    struct Data
    {
        pairfield pairs{};
        scalarfield magnitudes;
        scalarfield shell_magnitudes{};

        Data() = default;
        Data( pairfield pairs, scalarfield magnitudes )
                : pairs( std::move( pairs ) ), magnitudes( std::move( magnitudes ) ) {};

        Data( scalarfield shell_magnitudes ) : shell_magnitudes( std::move( shell_magnitudes ) ) {};
    };

    static bool valid_data( const Data & data )
    {
        if( !data.shell_magnitudes.empty() )
            return true;
        else
            return data.pairs.empty() || data.pairs.size() == data.magnitudes.size();
    };

    struct Cache
    {
        pairfield pairs{};
        scalarfield magnitudes{};
        scalarfield translation_distances{};
        vectorfield translation_vectors{};
    };

    static bool is_contributing( const Data &, const Cache & cache )
    {
        return !cache.pairs.empty();
    }

    struct IndexType
    {
        int ispin, jspin, ipair;
        bool invert;
    };

    using Index        = Engine::Span<const IndexType>;
    using IndexStorage = Backend::vector<IndexType>;

    using Energy = Functor::Local::Energy_Functor<Functor::Local::DataRef<Lattice_Spring_Potential>>;

    template<Field field>
    using Gradient = Functor::Local::Gradient_Functor<field, Functor::Local::DataRef<Lattice_Spring_Potential>>;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "Lattice Spring Potential";

    template<typename IndexStorageVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache,
        IndexStorageVector & indices )
    {
        using Indexing::idx_from_pair;

        // redundant neighbours are captured when expanding pairs below
        static constexpr bool use_redundant_neighbours = false;

        cache.pairs      = pairfield( 0 );
        cache.magnitudes = scalarfield( 0 );
        if( !data.shell_magnitudes.empty() )
        {
            // Generate neighbours by shells
            intfield shells( 0 );
            Neighbours::Get_Neighbours_in_Shells(
                geometry, data.shell_magnitudes.size(), cache.pairs, shells, use_redundant_neighbours );
            cache.magnitudes.reserve( cache.pairs.size() );
            for( std::size_t ipair = 0; ipair < cache.pairs.size(); ++ipair )
            {
                cache.magnitudes.push_back( data.shell_magnitudes[shells[ipair]] );
            }
        }
        else
        {
            // Use direct list of pairs
            cache.pairs      = data.pairs;
            cache.magnitudes = data.magnitudes;
        }

        cache.translation_vectors.resize( cache.pairs.size() );
        cache.translation_distances.resize( cache.pairs.size() );

        Backend::cpu::for_each_n(
            SPIRIT_CPU_PAR Backend::cpu::make_zip_iterator(
                begin( cache.pairs ), begin( cache.translation_vectors ), begin( cache.translation_distances ) ),
            cache.pairs.size(),
            Backend::cpu::make_zip_function(
                [&geometry]( const Pair & pair, Vector3 & translation_vector, scalar & translation_distance )
                {
                    translation_vector = geometry.cell_atoms[pair.j] - geometry.cell_atoms[pair.i]
                                         + static_cast<scalar>( pair.translations[0] ) * geometry.bravais_vectors[0]
                                         + static_cast<scalar>( pair.translations[1] ) * geometry.bravais_vectors[1]
                                         + static_cast<scalar>( pair.translations[2] ) * geometry.bravais_vectors[2];

                    translation_distance = translation_vector.norm();
                    translation_vector.normalize();
                } ) );

        for( unsigned int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( unsigned int i_pair = 0; i_pair < cache.pairs.size(); ++i_pair )
            {
                int ispin = cache.pairs[i_pair].i + icell * geometry.n_cell_atoms;
                int jspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    cache.pairs[i_pair] );
                if( jspin >= 0 )
                {
                    Backend::get<IndexStorage>( indices[ispin] )
                        .push_back( IndexType{ ispin, jspin, (int)i_pair, false } );
                    Backend::get<IndexStorage>( indices[jspin] )
                        .push_back( IndexType{ jspin, ispin, (int)i_pair, true } );
                }
            }
        }
    };
};

template<>
struct Functor::Local::DataRef<Lattice_Spring_Potential>
{
    using Interaction = Lattice_Spring_Potential;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              magnitudes( cache.magnitudes.data() ),
              translation_distances( cache.translation_distances.data() ),
              translation_vectors( cache.translation_vectors.data() )
    {
    }

    const bool is_contributing;

protected:
    const scalar * magnitudes;
    const scalar * translation_distances; // equilibrium distance
    const Vector3 * translation_vectors;  // equilibrium translation (normalized, ispin -> jspin)

    SPIRIT_HOSTDEVICE Vector3
    distance_vector( const Lattice_Spring_Potential::IndexType idx, const Vector3 * displacement ) const noexcept
    {
        return displacement[idx.jspin] - displacement[idx.ispin]
               + ( idx.invert ? -1.0 : 1.0 ) * translation_distances[idx.ipair] * translation_vectors[idx.ipair];
    }
};

// E_i = 0.25 * Σ_j [ V_ij * (|r_ij| - |R_ij|)^2 ]
//     = 0.25 * Σ_j [ V_ij * (|R_ij + u_j - u_i| - |R_ij|)^2 ]
template<>
inline scalar Lattice_Spring_Potential::Energy::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    // The prefactor has to be 1/4 = 1/2 * 1/2, such that V_ij has the same interpretation as a spring constant.
    // One factor 1/2 arises from deduplicating the double sum, the second factor 1/2 is the prefactor inherint to
    // the Hamiltonian of the harmonic oscillator.
    return 0.25
           * Backend::transform_reduce(
               index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
               [this, state] SPIRIT_LAMBDA( const Lattice_Spring_Potential::IndexType & idx ) -> scalar
               {
                   const auto & [ispin, jspin, iani, parity] = idx;
                   const auto dd = distance_vector( idx, state.displacement ).norm() - translation_distances[iani];

                   return magnitudes[iani] * dd * dd;
               } );
}

// ∇E_i = Σ_j [ V_ij * (|R_ij| - |r_ij|) / |r_ij| * r_ij ]
template<>
inline Vector3 Lattice_Spring_Potential::Gradient<Field::Displacement>::operator()(
    const Index & index, quantity<const Vector3 *> state ) const
{
    // don't need to check for `is_contributing` here, because the `transform_reduce` will short circuit correctly
    return Backend::transform_reduce(
        index.begin(), index.end(), zero_value<Vector3>(), Backend::plus<Vector3>{},
        [this, state] SPIRIT_LAMBDA( const Lattice_Spring_Potential::IndexType & idx ) -> Vector3
        {
            /*
             * The `d_norm==0` branch should never be taken, as the assumption for this interaction is that displacement
             * is much smaller than the translation distance. However, it exists as a safeguard to avoid `nan` values
             * from appearing. It relies on the assumption that the other forces will pull these atoms apart again.
             * Setting the gradient to a finite value would introduce a preferred direction.
             * Additionally, reaching this state should be highly unlikely.
             */
            const auto & [ispin, jspin, iani, parity] = idx;
            const auto d                              = distance_vector( idx, state.displacement );
            if( const auto d_norm = d.norm(); d_norm != 0 )
                return magnitudes[iani] * ( translation_distances[iani] / d_norm - 1.0 ) * d;
            else
            {
                return Vector3::Zero();
            }
        } );
}

} // namespace Interaction

} // namespace SpinLattice

} // namespace Engine
