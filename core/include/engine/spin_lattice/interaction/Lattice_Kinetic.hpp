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
        intfield indices{};
        scalarfield magnitudes{}; // inverse masses
        vectorfield normals{};

        Data() = default;
        Data( intfield indices, scalarfield magnitudes, vectorfield normals )
                : indices( std::move( indices ) ),
                  magnitudes( std::move( magnitudes ) ),
                  normals( std::move( normals ) ) {};
    };

    static bool valid_data( const Data & data )
    {
        using std::begin, std::end;

        if( 3 * data.indices.size() != data.magnitudes.size() || 3 * data.indices.size() != data.normals.size() )
            return false;
        if( std::any_of( begin( data.indices ), end( data.indices ), []( const int & i ) { return i < 0; } ) )
            return false;

        return true;
    }

    struct Cache
    {
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.indices.empty();
    };

    struct IndexType
    {
        int ispin, iani;
    };

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
        const ::Data::Geometry & geometry, const intfield &, const Data & data, Cache &, IndexStorageVector & indices )
    {
        using Indexing::check_atom_type;

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( unsigned int iani = 0; iani < data.indices.size(); ++iani )
            {
                int ispin = icell * geometry.n_cell_atoms + data.indices[iani];
                if( check_atom_type( geometry.atom_types[ispin] ) )
                    Backend::get<IndexStorage>( indices[ispin] ) = IndexType{ ispin, (int)iani };
            }
        }
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
              normals( data.normals.data() ),
              magnitudes( data.magnitudes.data() ) {};

    const bool is_contributing;

protected:
    const Vector3 * normals;
    const scalar * magnitudes;
};

template<>
inline scalar Lattice_Kinetic::Energy::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    scalar result = 0.0;
    if( is_contributing && index != nullptr )
    {
        const auto & [ispin, iani] = *index;
#pragma unroll
        for( auto i = 0; i < 3; ++i )
        {
            const auto d = normals[3 * iani + i].dot( state.momentum[ispin] );
            result += magnitudes[3 * iani + i] * d * d;
        }

        return 0.5 * result;
    }
    else
    {
        return result;
    }
}

template<>
inline Vector3
Lattice_Kinetic::Gradient<Field::Momentum>::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    Vector3 result = Vector3::Zero();
    if( is_contributing && index != nullptr )
    {
        const auto & [ispin, iani] = *index;
#pragma unroll
        for( auto i = 0; i < 3; ++i )
        {
            const auto d = normals[3 * iani + i].dot( state.momentum[ispin] );
            result += ( d * magnitudes[3 * iani + i] ) * normals[3 * iani + i];
        }
    }
    return result;
}

} // namespace Interaction

} // namespace SpinLattice

} // namespace Engine
