#pragma once

#include <engine/Vectormath_Defines.hpp>
#include <engine/common/Hamiltonian.hpp>
#include <engine/spin_lattice/Interaction_Standalone_Adaptor.hpp>
#include <engine/spin_lattice/StateType.hpp>
#include <engine/spin_lattice/interaction/Displacement_Anisotropy.hpp>
#include <engine/spin_lattice/interaction/Exchange.hpp>
#include <engine/spin_lattice/interaction/Lattice_Harmonic_Potential.hpp>
#include <engine/spin_lattice/interaction/Lattice_Kinetic.hpp>
#include <engine/spin_lattice/interaction/Lattice_Spring_Potential.hpp>
#include <engine/spin_lattice/interaction/Quadruplet.hpp>
#include <utility/Variadic_Traits.hpp>

namespace Engine
{

namespace SpinLattice
{

namespace Functor = Common::Functor;
namespace Accessor
{

using Common::Accessor::Energy;
using Common::Accessor::Energy_Single_Spin;
using Common::Accessor::Energy_Total;

template<Field... field>
struct Bind
{
    template<typename T>
    using Gradient = typename T::template Gradient<field...>;
};

} // namespace Accessor

// TODO: look into mixins and decide if they are more suitable to compose the `Hamiltonian` and `StandaloneAdaptor` types

// Hamiltonian for (pure) spin systems
template<typename state_type, typename StandaloneAdaptorType, typename... InteractionTypes>
class Hamiltonian : public Common::Hamiltonian<state_type, StandaloneAdaptorType, InteractionTypes...>
{
    using base_t = Common::Hamiltonian<state_type, StandaloneAdaptorType, InteractionTypes...>;

public:
    using Common::Hamiltonian<state_type, StandaloneAdaptorType, InteractionTypes...>::Hamiltonian;

    using state_t = state_type;

    template<Field field>
    void Gradient( const state_t & state, vectorfield & gradient )
    {
        const auto nos = get<field>( state ).size();

        if( gradient.size() != nos )
            gradient = vectorfield( nos, Vector3::Zero() );
        else
            Backend::fill( gradient.begin(), gradient.end(), Vector3::Zero() );

        Backend::transform(
            SPIRIT_PAR this->indices.begin(), this->indices.end(), gradient.begin(),
            Functor::transform_op(
                Functor::tuple_dispatch<Accessor::Bind<field>::template Gradient>( this->local ),
                Vector3( Vector3::Zero() ), state ) );

        Functor::apply(
            Functor::tuple_dispatch<Accessor::Bind<field>::template Gradient>( this->nonlocal ), state, gradient );
    };
};

struct HamiltonianVariantTypes
{
    using state_t     = quantity<vectorfield>;
    using AdaptorType = SpinLattice::Interaction::StandaloneAdaptor<state_t>;

    template<typename... Interactions>
    using HamiltonianTemplate = Hamiltonian<state_t, AdaptorType, Interactions...>;

    using Lattice_Spring   = HamiltonianTemplate<Interaction::Lattice_Kinetic, Interaction::Lattice_Spring_Potential>;
    using Lattice_Harmonic = HamiltonianTemplate<Interaction::Lattice_Kinetic, Interaction::Lattice_Harmonic_Potential>;

    using RotInvariant = HamiltonianTemplate<
        Interaction::Lattice_Kinetic, Interaction::Lattice_Spring_Potential, Interaction::Exchange,
        Interaction::Quadruplet, Interaction::Displacement_Anisotropy>;

    using Variant = std::variant<Lattice_Spring, Lattice_Harmonic, RotInvariant>;
};

// Single Type wrapper around Variant Hamiltonian type
// Should the visitors split up into standalone function objects?
class HamiltonianVariant : public Common::HamiltonianVariant<HamiltonianVariant, HamiltonianVariantTypes>
{
public:
    using state_t          = typename HamiltonianVariantTypes::state_t;
    using Lattice_Spring   = typename HamiltonianVariantTypes::Lattice_Spring;
    using Lattice_Harmonic = typename HamiltonianVariantTypes::Lattice_Harmonic;
    using RotInvariant     = typename HamiltonianVariantTypes::RotInvariant;
    using Variant          = typename HamiltonianVariantTypes::Variant;
    using AdaptorType      = typename HamiltonianVariantTypes::AdaptorType;

private:
    using base_t = Common::HamiltonianVariant<HamiltonianVariant, HamiltonianVariantTypes>;

public:
    explicit HamiltonianVariant( Lattice_Spring && hamiltonian ) noexcept(
        std::is_nothrow_move_constructible_v<Lattice_Spring> )
            : base_t( std::move( hamiltonian ) ) {};

    explicit HamiltonianVariant( Lattice_Harmonic && hamiltonian ) noexcept(
        std::is_nothrow_move_constructible_v<Lattice_Harmonic> )
            : base_t( std::move( hamiltonian ) ) {};

    explicit HamiltonianVariant( RotInvariant && hamiltonian ) noexcept(
        std::is_nothrow_move_constructible_v<RotInvariant> )
            : base_t( std::move( hamiltonian ) ) {};

    [[nodiscard]] std::string_view Name() const noexcept
    {
        if( std::holds_alternative<Lattice_Spring>( hamiltonian ) )
            return "Lattice (spring)";

        if( std::holds_alternative<Lattice_Harmonic>( hamiltonian ) )
            return "Lattice (harmonic)";

        if( std::holds_alternative<RotInvariant>( hamiltonian ) )
            return "Rotationally Invariant";

        // std::unreachable();

        return "Unknown";
    };

    template<Field field>
    void Gradient( const state_t & state, vectorfield & gradient )
    {
        std::visit( [&state, &gradient]( auto & h ) { h.template Gradient<field>( state, gradient ); }, hamiltonian );
    }

    void Gradient( const state_t & state, quantity<vectorfield> & gradient )
    {
        Gradient_Impl( make_enum_sequence<Field>(), state, gradient );
    }

    void Gradient_and_Energy( const state_t & state, quantity<vectorfield> & gradient, scalar & energy )
    {
        Gradient_Impl( make_enum_sequence<Field>(), state, gradient );
        energy = this->Energy( state );
    };

private:
    template<Field... field>
    void
    Gradient_Impl( const enum_sequence<Field, field...> &, const state_t & state, quantity<vectorfield> & gradient )
    {
        std::visit(
            [&state, &gradient]( auto & h ) { ( ..., h.template Gradient<field>( state, get<field>( gradient ) ) ); },
            hamiltonian );
    }
};

} // namespace SpinLattice

} // namespace Engine
