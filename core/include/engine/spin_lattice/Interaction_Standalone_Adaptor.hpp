#pragma once

#include <engine/common/Interaction_Standalone_Adaptor.hpp>
#include <engine/common/Interaction_Wrapper.hpp>
#include <engine/spin_lattice/interaction/Functor_Prototypes.hpp>

namespace Engine
{

namespace SpinLattice
{

namespace Interaction
{

using Common::Interaction::InteractionWrapper;
using Common::Interaction::is_local;

template<typename state_type>
struct StandaloneAdaptor : public Common::Interaction::StandaloneAdaptor<state_type>
{
    using state_t = state_type;

    virtual void Gradient( const state_t & state, quantity<vectorfield> & gradient )    = 0;
    virtual void Gradient_Spin( const state_t & state, vectorfield & gradient )         = 0;
    virtual void Gradient_Displacement( const state_t & state, vectorfield & gradient ) = 0;
    virtual void Gradient_Momentum( const state_t & state, vectorfield & gradient )     = 0;

protected:
    constexpr StandaloneAdaptor() = default;
};

template<typename InteractionType>
class StandaloneAdaptor_NonLocal
        : public Common::Interaction::StandaloneAdaptor_NonLocal<
              InteractionType, SpinLattice::Interaction::StandaloneAdaptor<typename InteractionType::state_t>>
{
    static_assert(
        !is_local<InteractionType>::value, "interaction type for non-local standalone adaptor must be non-local" );

    using base_t = Common::Interaction::StandaloneAdaptor_NonLocal<
        InteractionType, SpinLattice::Interaction::StandaloneAdaptor<typename InteractionType::state_t>>;

    using Interaction = InteractionType;
    using Data        = typename InteractionType::Data;
    using Cache       = typename InteractionType::Cache;

    struct constructor_tag
    {
        explicit constructor_tag() = default;
    };

public:
    using state_t = typename InteractionType::state_t;

    template<typename AdaptorType>
    friend class Common::Interaction::StandaloneFactory;

    StandaloneAdaptor_NonLocal( constructor_tag, const Data & data, Cache & cache ) noexcept : base_t( data, cache ) {};

    void Gradient( const state_t & state, quantity<vectorfield> & gradient ) final
    {
        std::invoke( typename InteractionType::Gradient( this->data, this->cache ), state, gradient );
    }

    void Gradient_Spin( const state_t & state, vectorfield & gradient ) final
    {
        Gradient_Impl<Field::Spin>( state, gradient );
    }

    void Gradient_Displacement( const state_t & state, vectorfield & gradient ) final
    {
        Gradient_Impl<Field::Displacement>( state, gradient );
    }

    void Gradient_Momentum( const state_t & state, vectorfield & gradient ) final
    {
        Gradient_Impl<Field::Momentum>( state, gradient );
    }

private:
    template<Field field>
    void Gradient_Impl( const state_t & state, vectorfield & gradient )
    {
        std::invoke( typename InteractionType::template Gradient<field>( this->data, this->cache ), state, gradient );
    }
};

template<typename InteractionType, typename IndexVector>
class StandaloneAdaptor_Local : public Common::Interaction::StandaloneAdaptor_Local<
                                    InteractionType, StandaloneAdaptor<typename InteractionType::state_t>, IndexVector>
{
    static_assert( is_local<InteractionType>::value, "interaction type for local standalone adaptor must be local" );

    using base_t = Common::Interaction::StandaloneAdaptor_Local<
        InteractionType, StandaloneAdaptor<typename InteractionType::state_t>, IndexVector>;

    using Interaction = InteractionType;
    using Data        = typename InteractionType::Data;
    using Cache       = typename InteractionType::Cache;
    using IndexTuple  = typename IndexVector::value_type;

    // private constructor tag with factory function (for std::unique_ptr) declared as friend
    // to make this the only way to instanciate this object
    struct constructor_tag
    {
        explicit constructor_tag() = default;
    };

public:
    using state_t = typename InteractionType::state_t;

    template<typename AdaptorType>
    friend class Common::Interaction::StandaloneFactory;

    StandaloneAdaptor_Local( constructor_tag, const Data & data, Cache & cache, const IndexVector & indices ) noexcept
            : base_t( data, cache, indices ) {};

    void Gradient_Spin( const state_t & state, vectorfield & gradient ) final
    {
        Gradient_Impl<Field::Spin>( state, gradient );
    }

    void Gradient_Displacement( const state_t & state, vectorfield & gradient ) final
    {
        Gradient_Impl<Field::Displacement>( state, gradient );
    }

    void Gradient_Momentum( const state_t & state, vectorfield & gradient ) final
    {
        Gradient_Impl<Field::Momentum>( state, gradient );
    }

    void Gradient( const state_t & state, quantity<vectorfield> & gradient ) final
    {
        Gradient_Spin( state, gradient.spin );
        Gradient_Displacement( state, gradient.displacement );
        Gradient_Momentum( state, gradient.momentum );
    }

private:
    template<Field field>
    void Gradient_Impl( const state_t & state, vectorfield & gradient )
    {
        auto functor = typename InteractionType::template Gradient<field>( this->data, this->cache );
        typename state_traits<state_t>::const_pointer state_ptr = state.data();
        Backend::transform(
            SPIRIT_PAR this->indices.begin(), this->indices.end(), gradient.begin(),
            [state_ptr, functor] SPIRIT_LAMBDA( const IndexTuple & index )
            { return functor( Backend::get<typename InteractionType::Index>( index ), state_ptr ); } );
    }
};

} // namespace Interaction

} // namespace SpinLattice

namespace Common
{

namespace Interaction
{

template<typename state_t>
class StandaloneFactory<SpinLattice::Interaction::StandaloneAdaptor<state_t>>
{
    using AdaptorType = SpinLattice::Interaction::StandaloneAdaptor<state_t>;

public:
    constexpr StandaloneFactory() = default;

    template<typename InteractionType>
    static auto
    make_standalone( InteractionWrapper<InteractionType> & interaction ) noexcept -> std::unique_ptr<AdaptorType>
    {
        static_assert(
            !is_local<InteractionType>::value, "interaction type for non-local standalone adaptor must be non-local" );
        using T = SpinLattice::Interaction::StandaloneAdaptor_NonLocal<InteractionType>;
        return std::make_unique<T>( typename T::constructor_tag{}, interaction.data, interaction.cache );
    };

    template<typename InteractionType, typename IndexVector>
    static auto
    make_standalone( InteractionWrapper<InteractionType> & interaction, const IndexVector & indices ) noexcept
        -> std::unique_ptr<AdaptorType>
    {
        static_assert(
            is_local<InteractionType>::value, "interaction type for local standalone adaptor must be local" );
        using T = SpinLattice::Interaction::StandaloneAdaptor_Local<InteractionType, IndexVector>;
        return std::make_unique<T>( typename T::constructor_tag{}, interaction.data, interaction.cache, indices );
    };
};

} // namespace Interaction

} // namespace Common

} // namespace Engine
