#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <data/Misc.hpp>
#include <engine/FFT.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/spin/Hamiltonian_Defines.hpp>
#include <engine/spin/interaction/Functors.hpp>
#include <engine/spin/interaction/Wrapper.hpp>
#include <engine/spin/interaction/all.hpp>
#include <utility/Span.hpp>
#include <utility/Variadic_Traits.hpp>

#include <variant>

namespace Engine
{

namespace Spin
{

using Interaction::tuple_bind;

// Hamiltonian for (pure) spin systems
template<typename state_type, typename... InteractionTypes>
class Hamiltonian
{
    static_assert( std::conjunction_v<std::is_same<state_type, typename InteractionTypes::state_t>...> );

    template<typename... Ts>
    using WrappedInteractions = std::tuple<Interaction::InteractionWrapper<Ts>...>;

    template<typename... Ts>
    using IndexTupleTemplate = std::tuple<typename Ts::Index...>;

    template<template<class> class Pred, template<class...> class Variadic>
    using filtered = typename Utility::variadic_filter<Pred, Variadic, InteractionTypes...>::type;

    template<template<class> class Pred>
    using indexed = typename Utility::variadic_filter_index_sequence<Pred, InteractionTypes...>::type;

    using LocalInteractions = filtered<Interaction::is_local, std::tuple>;

    template<class Tuple, std::size_t... I_local, std::size_t... I_nonlocal>
    Hamiltonian(
        std::shared_ptr<Data::Geometry> geometry, intfield boundary_conditions, std::index_sequence<I_local...>,
        std::index_sequence<I_nonlocal...>, Tuple && data )
            : geometry( std::move( geometry ) ),
              boundary_conditions( std::move( boundary_conditions ) ),
              local( std::make_tuple( std::get<I_local>( std::forward<Tuple>( data ) )... ) ),
              nonlocal( std::make_tuple( std::get<I_nonlocal>( std::forward<Tuple>( data ) )... ) ),
              indices{}
    {
        applyGeometry();
    };

    template<typename T>
    using is_local = Interaction::is_local<T>;

    template<typename T>
    using is_nonlocal = std::negation<Interaction::is_local<T>>;

public:
    using state_t = state_type;

    using local_indices    = indexed<is_local>;
    using nonlocal_indices = indexed<is_nonlocal>;

    using InteractionTuple = WrappedInteractions<InteractionTypes...>;

    using NonLocalInteractionTuple = filtered<is_nonlocal, WrappedInteractions>;

    using LocalInteractionTuple = filtered<is_local, WrappedInteractions>;
    using IndexTuple            = typename Utility::variadic_map<Trait::Index, std::tuple, LocalInteractions>::type;
    using IndexVector           = field<IndexTuple>;

    template<typename... DataTypes>
    Hamiltonian( std::shared_ptr<Data::Geometry> geometry, intfield boundary_conditions, DataTypes &&... data )
            : Hamiltonian(
                geometry, boundary_conditions, local_indices{}, nonlocal_indices{},
                std::make_tuple(
                    Interaction::InteractionWrapper<InteractionTypes>( std::forward<DataTypes>( data ) )... ) ){};

    // rule of five, because we use pointers to the geometry and the boundary_conditions in the cache
    // this choice should keep the interfaces a bit cleaner and allow adding more global dependencies
    // in the future.
    // TODO: Turn the Geometry into a stack member variable of this class or ensure otherwise that we
    // are notified whenever the address of the Geometry object might have changed
    ~Hamiltonian() = default;
    Hamiltonian( const Hamiltonian & other )
            : geometry( other.geometry ),
              boundary_conditions( other.boundary_conditions ),
              local( other.local ),
              indices( other.indices ),
              nonlocal( other.nonlocal )
    {
        setPtrAddress();
    };
    Hamiltonian & operator=( const Hamiltonian & other )
    {
        if( this != &other )
        {
            geometry            = other.geometry;
            boundary_conditions = other.boundary_conditions;
            local               = other.local;
            indices             = other.indices;
            nonlocal            = other.nonlocal;
            setPtrAddress();
        };
        return *this;
    };
    Hamiltonian( Hamiltonian && other ) noexcept
            : geometry( std::move( other.geometry ) ),
              boundary_conditions( std::move( other.boundary_conditions ) ),
              local( std::move( other.local ) ),
              indices( std::move( other.indices ) ),
              nonlocal( std::move( other.nonlocal ) )
    {
        setPtrAddress();
    };
    Hamiltonian & operator=( Hamiltonian && other ) noexcept
    {
        if( this != &other )
        {
            geometry            = std::move( other.geometry );
            boundary_conditions = std::move( other.boundary_conditions );
            local               = std::move( other.local );
            indices             = std::move( other.indices );
            nonlocal            = std::move( other.nonlocal );
            setPtrAddress();
        };
        return *this;
    };
    void setPtrAddress()
    {
        Interaction::setPtrAddress( local, geometry.get(), &boundary_conditions );
        Interaction::setPtrAddress( nonlocal, geometry.get(), &boundary_conditions );
    }

    void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin )
    {
        const auto nos = state.size();

        if( energy_per_spin.size() != nos )
        {
            energy_per_spin = scalarfield( nos, 0.0 );
        }
        else
        {
            Vectormath::fill( energy_per_spin, 0.0 );
        }

        std::transform(
            begin( indices ), end( indices ), begin( energy_per_spin ),
            Interaction::transform_op( tuple_bind<Accessor::Energy>( local ), scalar( 0.0 ), state ) );

        Interaction::Functors::apply( tuple_bind<Accessor::Energy>( nonlocal ), state, energy_per_spin );
    };

    void Energy_Contributions_per_Spin( const state_t & state, Data::vectorlabeled<scalarfield> & contributions )
    {
        auto active           = active_interactions();
        const auto & n_active = active.size();

        if( contributions.size() != n_active )
        {
            contributions = Data::vectorlabeled<scalarfield>( n_active, { "", scalarfield( state.size(), 0.0 ) } );
        }

        std::transform(
            begin( active ), end( active ), begin( contributions ),
            [&state]( const auto & interaction )
            {
                scalarfield energy_per_spin( state.size(), 0.0 );
                interaction->Energy_per_Spin( state, energy_per_spin );
                return std::make_pair( interaction->Name(), energy_per_spin );
            } );
    };

    [[nodiscard]] Data::vectorlabeled<scalar> Energy_Contributions( const state_t & state )
    {
        auto active           = active_interactions();
        const auto & n_active = active.size();
        Data::vectorlabeled<scalar> contributions( n_active, { "", 0.0 } );

        std::transform(
            begin( active ), end( active ), begin( contributions ),
            [&state]( const auto & interaction )
            { return std::make_pair( interaction->Name(), interaction->Energy( state ) ); } );

        return contributions;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    [[nodiscard]] scalar Energy_Single_Spin( const int ispin, const state_t & state )
    {
        return Interaction::transform_op( tuple_bind<Accessor::Energy_Single_Spin>( local ), scalar( 0.0 ), state )(
                   indices[ispin] )
               + Interaction::Functors::apply_reduce(
                   tuple_bind<Accessor::Energy_Single_Spin>( nonlocal ), scalar( 0.0 ), ispin, state );
    };

    [[nodiscard]] scalar Energy( const state_t & state )
    {
        return std::transform_reduce(
                   begin( indices ), end( indices ), scalar( 0 ), std::plus{},
                   Interaction::transform_op( tuple_bind<Accessor::Energy>( local ), scalar( 0.0 ), state ) )
               + Interaction::Functors::apply_reduce(
                   tuple_bind<Accessor::Energy_Total>( nonlocal ), scalar( 0.0 ), state );
    };

    void Hessian( const state_t & state, MatrixX & hessian )
    {
        hessian.setZero();
        Hessian_Impl( state, Interaction::dense_hessian_functor( hessian ) );
    };

    void Sparse_Hessian( const state_t & state, SpMatrixX & hessian )
    {
        field<Common::Interaction::triplet> tripletList;
        tripletList.reserve( geometry->n_cells_total * Sparse_Hessian_Size_per_Cell() );
        Hessian_Impl( state, Interaction::sparse_hessian_functor( tripletList ) );
        hessian.setFromTriplets( begin( tripletList ), end( tripletList ) );
    };

    std::size_t Sparse_Hessian_Size_per_Cell() const
    {
        auto func = []( const auto &... interaction ) -> std::size_t
        { return ( 0 + ... + interaction.Sparse_Hessian_Size_per_Cell() ); };
        return std::apply( func, local ) + std::apply( func, nonlocal );
    };

    void Gradient( const state_t & state, vectorfield & gradient )
    {
        std::transform(
            begin( indices ), end( indices ), begin( gradient ),
            Interaction::transform_op( tuple_bind<Accessor::Gradient>( local ), Vector3::Zero(), state ) );

        Interaction::Functors::apply( tuple_bind<Accessor::Gradient>( nonlocal ), state, gradient );
    };

    // provided for backwards compatibility, this function no longer serves a purpose
    [[nodiscard]] scalar Gradient_and_Energy( const state_t & state, vectorfield & gradient )
    {
        Gradient( state, gradient );
        return Energy( state );
    };

    [[nodiscard]] std::size_t active_count() const
    {
        auto f = []( const auto &... interaction )
        { return ( std::size_t( 0 ) + ... + ( interaction.is_contributing() ? 1 : 0 ) ); };
        return std::apply( f, local ) + std::apply( f, nonlocal );
    }

    [[nodiscard]] auto active_interactions() -> std::vector<std::unique_ptr<Interaction::StandaloneAdapter<state_t>>>
    {
        std::vector<std::unique_ptr<Interaction::StandaloneAdapter<state_t>>> interactions( active_count() );
        auto it = Interaction::generate_local( local, indices, begin( interactions ) );
        Interaction::generate_nonlocal( nonlocal, it );
        return interactions;
    };

    [[nodiscard]] auto active_interactions() const
        -> std::vector<std::unique_ptr<const Interaction::StandaloneAdapter<state_t>>>
    {
        std::vector<std::unique_ptr<Interaction::StandaloneAdapter<state_t>>> interactions( active_count() );
        auto it = Interaction::generate_local( local, indices, begin( interactions ) );
        Interaction::generate_nonlocal( nonlocal, it );
        return interactions;
    };

    // compile time getter
    template<class T>
    [[nodiscard]] constexpr auto getInteraction() -> std::unique_ptr<Interaction::StandaloneAdapter<state_t>>
    {
        if constexpr( hasInteraction_Local<T>() )
            return Interaction::make_standalone( std::get<Interaction::InteractionWrapper<T>>( local ), indices );
        else if constexpr( hasInteraction_NonLocal<T>() )
            return Interaction::make_standalone( std::get<Interaction::InteractionWrapper<T>>( nonlocal ) );
        else
            return nullptr;
    };

    // compile time getter
    template<class T>
    [[nodiscard]] constexpr auto getInteraction() const
        -> std::unique_ptr<const Interaction::StandaloneAdapter<state_t>>
    {
        if constexpr( hasInteraction_Local<T>() )
            return Interaction::make_standalone( std::get<Interaction::InteractionWrapper<T>>( local ), indices );
        else if constexpr( hasInteraction_NonLocal<T>() )
            return Interaction::make_standalone( std::get<Interaction::InteractionWrapper<T>>( nonlocal ) );
        else
            return nullptr;
    };

    template<class T>
    [[nodiscard]] static constexpr bool hasInteraction()
    {
        return Utility::has_type<T, decltype( local )>::value || Utility::has_type<T, decltype( nonlocal )>::value;
    };

    void applyGeometry()
    {
        if( geometry == nullptr )
        {
            setPtrAddress();
            return;
        }

        if( indices.size() != static_cast<std::size_t>( geometry->nos ) )
        {
            indices = IndexVector( geometry->nos, IndexTuple{} );
        }
        else
        {
            std::for_each(
                begin( indices ), end( indices ),
                [&interactions = local]( IndexTuple & index )
                {
                    std::apply(
                        [&index]( const auto &... interaction ) { ( ..., interaction.clearIndex( index ) ); },
                        interactions );
                } );
        }

        std::apply(
            [this]( auto &... interaction ) -> void
            { ( ..., interaction.applyGeometry( *geometry, boundary_conditions, indices ) ); },
            local );

        std::apply(
            [this]( auto &... interaction ) -> void
            { ( ..., interaction.applyGeometry( *geometry, boundary_conditions ) ); },
            nonlocal );
    }

    template<typename T>
    [[nodiscard]] auto data() const -> std::enable_if_t<hasInteraction<T>(), const typename T::Data &>
    {
        if constexpr( hasInteraction_Local<T>() )
            return std::get<Interaction::InteractionWrapper<T>>( local ).data;
        else if constexpr( hasInteraction_NonLocal<T>() )
            return std::get<Interaction::InteractionWrapper<T>>( nonlocal ).data;
        // std::unreachable();
    };

    template<typename T>
    [[nodiscard]] auto set_data( const typename T::Data & data ) -> std::enable_if_t<hasInteraction<T>(), bool>
    {
        bool success = false;
        if constexpr( hasInteraction_Local<T>() )
            success = std::get<Interaction::InteractionWrapper<T>>( local ).set_data( data );
        else if constexpr( hasInteraction_NonLocal<T>() )
            success = std::get<Interaction::InteractionWrapper<T>>( nonlocal ).set_data( data );
        applyGeometry<T>();
        return success;
    };

    template<typename T>
    [[nodiscard]] auto cache() const -> std::enable_if_t<hasInteraction<T>(), const typename T::Cache &>
    {
        if constexpr( hasInteraction_Local<T>() )
            return std::get<Interaction::InteractionWrapper<T>>( local ).cache;
        else if constexpr( hasInteraction_NonLocal<T>() )
            return std::get<Interaction::InteractionWrapper<T>>( nonlocal ).cache;
        // std::unreachable();
    };

    [[nodiscard]] const auto & getBoundaryConditions() const
    {
        return boundary_conditions;
    }

    void setBoundaryConditions( const intfield & bc )
    {
        boundary_conditions = bc;
        applyGeometry();
    }

    [[nodiscard]] const auto & getGeometry() const
    {
        return *geometry;
    }

    void setGeometry( const ::Data::Geometry & g )
    {
        *geometry = g;
        applyGeometry();
    }

private:
    template<typename Callable>
    void Hessian_Impl( const state_t & state, Callable hessian )
    {
        std::for_each(
            begin( indices ), end( indices ),
            Interaction::for_each_op( tuple_bind<Accessor::Hessian>( local ), state, hessian ) );

        Interaction::Functors::apply( tuple_bind<Accessor::Hessian>( nonlocal ), state, hessian );
    };

    template<typename InteractionType>
    void applyGeometry()
    {
        static_assert( hasInteraction<InteractionType>() );

        if( geometry == nullptr )
        {
            setPtrAddress();
            return;
        }

        // TODO(feature-template_hamiltonian): turn this into an error
        if( geometry->nos != indices.size() )
            return;

        if constexpr( is_local<InteractionType>::value )
        {
            std::get<Interaction::InteractionWrapper<InteractionType>>( local ).applyGeometry(
                *geometry, boundary_conditions, indices );
        }
        else if constexpr( !is_local<InteractionType>::value )
        {
            std::get<Interaction::InteractionWrapper<InteractionType>>( nonlocal )
                .applyGeometry( *geometry, boundary_conditions );
        };
        // std::unreachable();
    };

    template<class T>
    [[nodiscard]] static constexpr bool hasInteraction_Local()
    {
        return is_local<T>::value && Utility::has_type<Interaction::InteractionWrapper<T>, decltype( local )>::value;
    };

    template<class T>
    [[nodiscard]] static constexpr bool hasInteraction_NonLocal()
    {
        return !is_local<T>::value
               && Utility::has_type<Interaction::InteractionWrapper<T>, decltype( nonlocal )>::value;
    };

    std::shared_ptr<Data::Geometry> geometry;
    intfield boundary_conditions;

    LocalInteractionTuple local{};
    IndexVector indices{};
    NonLocalInteractionTuple nonlocal{};
};

// Single Type wrapper around Variant Hamiltonian type
// Should the visitors split up into standalone function objects?
class HamiltonianVariant
{
public:
    using state_t    = vectorfield;
    using Gaussian   = Hamiltonian<state_t, Interaction::Gaussian>;
    using Heisenberg = Hamiltonian<
        state_t, Interaction::Zeeman, Interaction::Anisotropy, Interaction::Biaxial_Anisotropy,
        Interaction::Cubic_Anisotropy, Interaction::Exchange, Interaction::DMI, Interaction::Quadruplet,
        Interaction::DDI>;

private:
    using Variant = std::variant<Gaussian, Heisenberg>;

public:
    explicit constexpr HamiltonianVariant( Gaussian && gaussian ) noexcept(
        std::is_nothrow_move_constructible_v<Gaussian> )
            : hamiltonian( gaussian ){};

    explicit constexpr HamiltonianVariant( Heisenberg && heisenberg ) noexcept(
        std::is_nothrow_move_constructible_v<Gaussian> )
            : hamiltonian( heisenberg ){};

    [[nodiscard]] std::string_view Name() const noexcept
    {
        if( std::holds_alternative<Gaussian>( hamiltonian ) )
            return "Gaussian";

        if( std::holds_alternative<Heisenberg>( hamiltonian ) )
            return "Heisenberg";

        // std::unreachable();

        return "Unknown";
    };

    [[nodiscard]] scalar Energy( const state_t & state )
    {
        return std::visit( [&state]( auto & h ) { return h.Energy( state ); }, hamiltonian );
    }

    void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin )
    {
        std::visit(
            [&state, &energy_per_spin]( auto & h ) { h.Energy_per_Spin( state, energy_per_spin ); }, hamiltonian );
    }

    [[nodiscard]] scalar Energy_Single_Spin( const int ispin, const state_t & state )
    {
        return std::visit( [ispin, &state]( auto & h ) { return h.Energy_Single_Spin( ispin, state ); }, hamiltonian );
    }

    void Gradient( const state_t & state, vectorfield & gradient )
    {
        std::visit( [&state, &gradient]( auto & h ) { h.Gradient( state, gradient ); }, hamiltonian );
    }

    void Hessian( const state_t & state, MatrixX & hessian )
    {
        std::visit( [&state, &hessian]( auto & h ) { h.Hessian( state, hessian ); }, hamiltonian );
    }

    void Sparse_Hessian( const state_t & state, SpMatrixX & hessian )
    {
        std::visit( [&state, &hessian]( auto & h ) { h.Sparse_Hessian( state, hessian ); }, hamiltonian );
    }

    void Gradient_and_Energy( const state_t & state, vectorfield & gradient, scalar & energy )
    {
        std::visit(
            [&state, &gradient, &energy]( auto & h ) { energy = h.Gradient_and_Energy( state, gradient ); },
            hamiltonian );
    };

    void Energy_Contributions_per_Spin( const state_t & state, Data::vectorlabeled<scalarfield> & contributions )
    {
        std::visit(
            [&state, &contributions]( auto & h ) { h.Energy_Contributions_per_Spin( state, contributions ); },
            hamiltonian );
    };

    [[nodiscard]] Data::vectorlabeled<scalar> Energy_Contributions( const state_t & state )
    {
        return std::visit( [&state]( auto & h ) { return h.Energy_Contributions( state ); }, hamiltonian );
    };

    void onGeometryChanged()
    {
        std::visit( []( auto & h ) { return h.applyGeometry(); }, hamiltonian );
    };

    [[nodiscard]] std::size_t active_count() const
    {
        return std::visit( []( const auto & h ) { return h.active_count(); }, hamiltonian );
    }

    [[nodiscard]] auto active_interactions() -> std::vector<std::unique_ptr<Interaction::StandaloneAdapter<state_t>>>
    {
        return std::visit( []( auto & h ) { return h.active_interactions(); }, hamiltonian );
    };

    [[nodiscard]] auto getBoundaryConditions() const -> const intfield &
    {
        return std::visit(
            []( const auto & h ) -> decltype( auto ) { return h.getBoundaryConditions(); }, hamiltonian );
    };

    void setBoundaryConditions( const intfield & boundary_conditions )
    {
        std::visit(
            [&boundary_conditions]( auto & h ) { return h.setBoundaryConditions( boundary_conditions ); },
            hamiltonian );
    };

    [[nodiscard]] auto getGeometry() const -> const ::Data::Geometry &
    {
        return std::visit( []( const auto & h ) -> decltype( auto ) { return h.getGeometry(); }, hamiltonian );
    };

    void setGeometry( const ::Data::Geometry & geometry )
    {
        std::visit( [&geometry]( auto & h ) { h.setGeometry( geometry ); }, hamiltonian );
    };

    template<class T>
    [[nodiscard]] bool hasInteraction()
    {
        return std::visit( []( auto & h ) { return h.template hasInteraction<T>(); }, hamiltonian );
    };

    template<class T>
    [[nodiscard]] auto getInteraction() -> std::unique_ptr<Interaction::StandaloneAdapter<state_t>>
    {
        return std::visit( []( auto & h ) { return h.template getInteraction<T>(); }, hamiltonian );
    };

    template<typename T>
    [[nodiscard]] auto data() const -> const typename T::Data *
    {
        return std::visit(
            []( const auto & h ) -> const typename T::Data *
            {
                if constexpr( h.template hasInteraction<T>() )
                    return &h.template cache<T>();
                else
                    return nullptr;
            },
            hamiltonian );
    };

    template<typename T>
    [[nodiscard]] auto cache() -> const typename T::Cache *
    {
        return std::visit(
            []( const auto & h ) -> const typename T::Cache *
            {
                if constexpr( h.template hasInteraction<T>() )
                    return &h.template cache<T>();
                else
                    return nullptr;
            },
            hamiltonian );
    };

    template<typename T>
    [[nodiscard]] bool set_data( const typename T::Data & data )
    {
        return std::visit(
            [&data]( auto & h ) -> bool
            {
                if constexpr( h.template hasInteraction<T>() )
                    return h.template set_data<T>();
                else
                    return false;
            },
            hamiltonian );
    };

private:
    Variant hamiltonian;
};

} // namespace Spin

} // namespace Engine

#endif
