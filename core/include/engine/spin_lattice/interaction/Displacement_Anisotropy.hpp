#pragma once

#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/spin_lattice/StateType.hpp>
#include <engine/spin_lattice/interaction/Functor_Prototypes.hpp>

#include <Eigen/Dense>

namespace Engine
{

namespace SpinLattice
{

namespace Interaction
{

struct Displacement_Anisotropy
{
    using state_t = StateType;

    struct Data
    {
        pairfield pairs;
        scalarfield magnitudes;

        scalarfield shell_magnitudes;

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
    }

    struct Cache
    {
        pairfield pairs{};
        scalarfield magnitudes{};
        vectorfield translation_vectors{};
    };

    static bool is_contributing( const Data &, const Cache & cache )
    {
        return !cache.pairs.empty();
    }

    struct IndexType
    {
        int ispin, jspin, ipair;
        bool onsite;
    };

    using Index        = Engine::Span<const IndexType>;
    using IndexStorage = Backend::vector<IndexType>;

    using Energy = Functor::Local::Energy_Functor<Functor::Local::DataRef<Displacement_Anisotropy>>;

    template<Field field>
    using Gradient = Functor::Local::Gradient_Functor<field, Functor::Local::DataRef<Displacement_Anisotropy>>;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "Cube Anisotropy";

    template<typename IndexStorageVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache,
        IndexStorageVector & indices )
    {
        using Indexing::idx_from_pair;

        /*
         * Redundant neighbours are needed, because the pair has a strong sense of directionality.
         * The interaction is really an onsite interaction that is induced by partner (neighbouring) atoms.
         * Because of this the pair (i -> j) is different from the pair (j -> i) other than for the
         * Heisenberg Exchange and DMI interactions.
         */
        static constexpr bool use_redundant_neighbours = true;

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

        Backend::cpu::for_each_n(
            SPIRIT_CPU_PAR Backend::cpu::make_zip_iterator( begin( cache.pairs ), begin( cache.translation_vectors ) ),
            cache.pairs.size(),
            Backend::cpu::make_zip_function(
                [&geometry]( const Pair & pair, Vector3 & translation_vector )
                {
                    translation_vector = geometry.cell_atoms[pair.j] - geometry.cell_atoms[pair.i]
                                         + static_cast<scalar>( pair.translations[0] ) * geometry.bravais_vectors[0]
                                         + static_cast<scalar>( pair.translations[1] ) * geometry.bravais_vectors[1]
                                         + static_cast<scalar>( pair.translations[2] ) * geometry.bravais_vectors[2];
                } ) );

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( unsigned int ipair = 0; ipair < data.pairs.size(); ++ipair )
            {
                int ispin = cache.pairs[ipair].i + icell * geometry.n_cell_atoms;
                int jspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    cache.pairs[ipair] );
                if( jspin >= 0 )
                {
                    /*
                     * breaking the correspondence that the first spin is the index of the site on which it operates
                     * allows us to omit a parameter for inverting the distance vector, because it directly corresponds
                     * to the ordered pair (i -> j). Consequently, the first index i is always the `onsite` index,
                     * i.e. the index of the spin whose orientation contributes to the interaction (see gradient
                     * implementation below).
                     */
                    Backend::get<IndexStorage>( indices[ispin] )
                        .push_back( IndexType{ ispin, jspin, (int)ipair, true } );
                    Backend::get<IndexStorage>( indices[jspin] )
                        .push_back( IndexType{ ispin, jspin, (int)ipair, false } );
                }
            }
        }
    }
};

template<>
struct Functor::Local::DataRef<Displacement_Anisotropy>
{
    using Interaction = Displacement_Anisotropy;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              magnitudes( cache.magnitudes.data() ),
              translation_vectors( cache.translation_vectors.data() )
    {
    }

    const bool is_contributing;

protected:
    const scalar * magnitudes;
    const Vector3 * translation_vectors; // equilibrium translation

    static constexpr scalar thresh = 1e-6;

    SPIRIT_HOSTDEVICE Vector3
    distance_vector( const Interaction::IndexType idx, const Vector3 * displacement ) const noexcept
    {
        return ( displacement[idx.jspin] - displacement[idx.ispin] ) + translation_vectors[idx.ipair];
    }
};

// E_i = -0.5 * Σ_j [ V_ij * ( n_i · r_ij )^2 ]
template<>
inline scalar Displacement_Anisotropy::Energy::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    return -0.5
           * Backend::transform_reduce(
               index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
               [this, state] SPIRIT_LAMBDA( const Interaction::IndexType & idx ) -> scalar
               {
                   if( !idx.onsite )
                       return 0.0;
                   else
                   {
                       const auto dot
                           = state.spin[idx.ispin].dot( distance_vector( idx, state.displacement ).normalized() );
                       return magnitudes[idx.ipair] * dot * dot;
                   }
               } );
}

// ∇E_i = - Σ_j [ V_ij * ( n_i · e_ij ) * e_ij ]
template<>
inline Vector3
Displacement_Anisotropy::Gradient<Field::Spin>::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    return -1.0
           * Backend::transform_reduce(
               index.begin(), index.end(), zero_value<Vector3>(), Backend::plus<Vector3>{},
               [this, state] SPIRIT_LAMBDA( const Interaction::IndexType & idx ) -> Vector3
               {
                   if( !idx.onsite )
                       return Vector3::Zero();
                   else
                   {
                       const auto e_ij = distance_vector( idx, state.displacement ).normalized();
                       return magnitudes[idx.ipair] * state.spin[idx.ispin].dot( e_ij ) * e_ij;
                   }
               } );
}

//

// ∇E_k = - Σ_ij 1/|r_ij| * [ V_ij * ( n_i · e_ij ) * ( δ_jk - δ_ki ) * ( 1 - e_ij e_ij^T ) n_i ]
template<>
inline Vector3 Displacement_Anisotropy::Gradient<Field::Displacement>::operator()(
    const Index & index, quantity<const Vector3 *> state ) const
{
    // TODO: find out why the precision is so low (|∇(E) - ∇_FD(E)| > 1e-7) when comparing against the finite difference gradient
    return -1.0
           * Backend::transform_reduce(
               index.begin(), index.end(), zero_value<Vector3>(), Backend::plus<Vector3>{},
               [this, state] SPIRIT_LAMBDA( const Interaction::IndexType & idx ) -> Vector3
               {
                   auto r_ij = distance_vector( idx, state.displacement );
                   if( const scalar distance = r_ij.norm(); distance < thresh )
                       return Vector3::Zero();
                   else
                   {
                       r_ij.normalize();
                       if( !idx.onsite )
                       {
                           const scalar dot = r_ij.dot( state.spin[idx.ispin /*sic!*/] );
                           return magnitudes[idx.ipair] * dot / distance
                                  * /*( 1 - e_ij e_ij^T )n_j=*/( state.spin[idx.ispin /*sic!*/] - dot * r_ij );
                       }
                       else
                       {
                           const scalar dot = r_ij.dot( state.spin[idx.ispin] );
                           return -magnitudes[idx.ipair] * dot / distance
                                  * /*( 1 - e_ij e_ij^T )n_i=*/( state.spin[idx.ispin] - dot * r_ij );
                       }
                   }
               } );
}

} // namespace Interaction

} // namespace SpinLattice

} // namespace Engine
