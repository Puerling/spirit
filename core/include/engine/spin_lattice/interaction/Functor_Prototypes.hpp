#pragma once

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/Backend.hpp>
#include <engine/Span.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/interaction/Functor_Prototypes.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>
#include <engine/spin_lattice/StateType.hpp>

namespace Engine
{

namespace SpinLattice
{

namespace Interaction
{

namespace Functor
{

namespace NonLocal
{

using Common::Interaction::Functor::NonLocal::Reduce_Functor;

template<typename InteractionType>
struct DataRef
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    constexpr DataRef( const Data & data, Cache & cache ) noexcept : data( data ), cache( cache ) {};

    const Data & data;
    Cache & cache;
    bool is_contributing = Interaction::is_contributing( data, cache );
};

template<typename DataRef>
struct Energy_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    void operator()( const StateType & state, scalarfield & energy ) const;

    using DataRef::DataRef;
};

template<Field field, typename DataRef>
struct Gradient_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    void operator()( const StateType & state, quantity<vectorfield> & gradient ) const;

    using DataRef::DataRef;
};

template<typename DataRef>
struct Energy_Single_Spin_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    scalar operator()( int ispin, const StateType & state ) const;

    using DataRef::DataRef;
};

} // namespace NonLocal

namespace Local
{

using Common::Interaction::Functor::Local::Energy_Single_Spin_Functor;

template<typename InteractionType>
struct DataRef
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    constexpr DataRef( const Data & data, const Cache & cache ) noexcept : data( data ), cache( cache ) {};

    const Data & data;
    const Cache & cache;
    const bool is_contributing = Interaction::is_contributing( data, cache );
};

template<typename DataRef>
struct Energy_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    SPIRIT_HOSTDEVICE scalar operator()( const Index & index, quantity<const Vector3 *> state ) const;

    using DataRef::DataRef;
};

template<Field value, typename DataRef>
struct Gradient_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    static constexpr Field field = value;

    SPIRIT_HOSTDEVICE Vector3 operator()( const Index &, quantity<const Vector3 *> ) const
    {
        return Vector3::Zero();
    };

    using DataRef::DataRef;
};

namespace SpinWrapper
{

namespace Spin = Spin::Interaction::Functor::Local;

template<typename InteractionType>
struct Energy_Functor : public Spin::Energy_Functor<Spin::DataRef<typename InteractionType::base_t>>
{
private:
    using base_t = typename Spin::Energy_Functor<Spin::DataRef<typename InteractionType::base_t>>;

public:
    using Interaction = InteractionType;
    using Data        = typename Interaction::base_t::Data;
    using Cache       = typename Interaction::base_t::Cache;
    using Index       = typename Interaction::base_t::Index;

    SPIRIT_HOSTDEVICE scalar operator()( const Index & index, quantity<const Vector3 *> state ) const
    {
        return base_t::operator()( index, state.spin );
    };

    using base_t::base_t;
};

template<Field value, typename InteractionType>
struct Gradient_Functor : public Spin::Gradient_Functor<Spin::DataRef<typename InteractionType::base_t>>
{
private:
    using base_t = typename Spin::Gradient_Functor<Spin::DataRef<typename InteractionType::base_t>>;

public:
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    static constexpr Field field = value;

    SPIRIT_HOSTDEVICE Vector3 operator()( const Index &, quantity<const Vector3 *> ) const
    {
        return Vector3::Zero();
    };

    using base_t::base_t;
};

template<typename InteractionType>
struct Gradient_Functor<Field::Spin, InteractionType>
        : public Spin::Gradient_Functor<Spin::DataRef<typename InteractionType::base_t>>
{
private:
    using base_t = typename Spin::Gradient_Functor<Spin::DataRef<typename InteractionType::base_t>>;

public:
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    static constexpr Field field = Field::Spin;

    SPIRIT_HOSTDEVICE Vector3 operator()( const Index & index, quantity<const Vector3 *> state ) const
    {
        return base_t::operator()( index, state.spin );
    };

    using base_t::base_t;
};

} // namespace SpinWrapper

} // namespace Local

} // namespace Functor

} // namespace Interaction

} // namespace SpinLattice

} // namespace Engine
