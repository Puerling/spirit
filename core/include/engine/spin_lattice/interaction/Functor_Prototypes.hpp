#pragma once

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/Backend.hpp>
#include <engine/Span.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/interaction/Functor_Prototypes.hpp>
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

} // namespace Local

} // namespace Functor

} // namespace Interaction

} // namespace SpinLattice

} // namespace Engine
