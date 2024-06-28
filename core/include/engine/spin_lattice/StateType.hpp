#pragma once

#include <engine/Vectormath_Defines.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <type_traits>

namespace Engine
{

namespace SpinLattice
{

template<typename T>
struct quantity;

}

template<typename state_type>
struct state_traits;

template<typename T>
struct state_traits<SpinLattice::quantity<T>>
{
    using type          = SpinLattice::quantity<T>;
    using pointer       = SpinLattice::quantity<typename T::pointer>;
    using const_pointer = SpinLattice::quantity<typename T::const_pointer>;
};

namespace SpinLattice
{

// index type
enum struct Field
{
    Spin         = 0,
    Displacement = 1,
    Momentum     = 2,
};

// record type
template<typename T>
struct quantity
{
    T spin;
    T displacement;
    T momentum;
};

template<typename T>
struct quantity<field<T>>
{
    field<T> spin;
    field<T> displacement;
    field<T> momentum;

    auto data() -> typename state_traits<quantity<field<T>>>::pointer
    {
        return { spin.data(), displacement.data(), momentum.data() };
    }

    auto data() const -> typename state_traits<quantity<field<T>>>::const_pointer
    {
        return { spin.data(), displacement.data(), momentum.data() };
    }
};

// factories
template<typename T, typename U = T, typename V = U>
constexpr auto make_quantity( T && spin, U && displacement, V && momentum )
    -> quantity<std::common_type_t<std::decay_t<T>, std::decay_t<U>, std::decay_t<V>>>
{
    return { std::forward<T>( spin ), std::forward<U>( displacement ), std::forward<V>( momentum ) };
}

template<typename T>
constexpr auto make_quantity( T value ) -> quantity<T>
{
    return { value, value, value };
}

// indexing (individual)
template<Field field, typename T>
T & get( quantity<T> & q )
{
    if constexpr( field == Field::Spin )
        return q.spin;
    if constexpr( field == Field::Displacement )
        return q.displacement;
    if constexpr( field == Field::Momentum )
        return q.momentum;
}

template<Field field, typename T>
const T & get( const quantity<T> & q )
{
    if constexpr( field == Field::Spin )
        return q.spin;
    if constexpr( field == Field::Displacement )
        return q.displacement;
    if constexpr( field == Field::Momentum )
        return q.momentum;
}

// indexing (sequenced)
template<typename Enum, Enum... Values>
struct enum_sequence : std::integer_sequence<Enum, Values...>
{
};

template<typename Enum>
constexpr auto make_enum_sequence();

template<>
constexpr auto make_enum_sequence<Field>()
{
    return enum_sequence<Field, Field::Spin, Field::Displacement, Field::Momentum>{};
};

// algebraic operations
template<typename T, typename U = T>
auto operator+( const quantity<T> & lhs, const quantity<U> & rhs ) -> quantity<typename std::common_type<T, U>::type>
{
    return { lhs.spin + rhs.spin, lhs.displacement + rhs.displacement, lhs.momentum + rhs.momentum };
}

template<typename T, typename U = T>
auto operator+=( quantity<T> & lhs, const quantity<U> & rhs ) -> quantity<T> &
{
    lhs.spin += rhs.spin;
    lhs.displacement += rhs.displacement;
    lhs.momentum += rhs.momentum;
    return lhs;
}

// common usage
using StateType = quantity<vectorfield>;
using StatePtr  = quantity<Vector3 *>;
using StateCPtr = quantity<const Vector3 *>;

} // namespace SpinLattice

template<typename state_t>
state_t make_state( int nos );

template<>
inline SpinLattice::StateType make_state( int nos )
{
    return SpinLattice::make_quantity( vectorfield( nos ) );
};

static_assert( std::is_same_v<std::common_type_t<Vector2, Vector2>, Vector2> );

} // namespace Engine
