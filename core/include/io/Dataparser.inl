#pragma once

#include <io/Dataparser.hpp>

namespace IO
{

namespace Spin
{

namespace State
{

template<typename T>
class Buffer
{
    static_assert( std::is_same_v<std::remove_const_t<T>, vectorfield> );
    using pointer = std::conditional_t<std::is_const_v<T>, const T *, T *>;

public:
    explicit Buffer( T & state ) : m_state( &state ) {};

    scalar * data()
    {
        static_assert( !std::is_const_v<T> );

        if( size() == 0 )
            return nullptr;
        return m_state->data()->data();
    };

    const scalar * data() const
    {
        if( size() == 0 )
            return nullptr;
        return m_state->data()->data();
    };

    std::size_t size() const
    {
        if( m_state == nullptr )
            return 0;
        return m_state->size();
    };

    friend void copy( const Buffer & buffer, vectorfield & state )
    {
        if( buffer.m_state == &state )
            return;
        if( buffer.m_state == nullptr )
            state.clear();

        std::copy( buffer.m_state->begin(), buffer.m_state->end(), state.begin() );
    }

private:
    pointer m_state;
};

} // namespace State

} // namespace Spin

namespace SpinLattice
{

namespace State
{

class Buffer
{
public:
    using row_type  = std::array<scalar, 9>;
    using data_type = std::vector<row_type>;

    explicit Buffer( std::size_t size ) : m_data( size ) {};
    explicit Buffer( data_type data ) : m_data( std::move( data ) ) {};
    explicit Buffer( const StateType & state )
            : m_data( std::min( std::min( state.spin.size(), state.displacement.size() ), state.momentum.size() ) )
    {
        copy( state, *this );
    };

    scalar * data()
    {
        return m_data.data()->data();
    }

    const scalar * data() const
    {
        return m_data.data()->data();
    }

    std::size_t size() const
    {
        return m_data.size();
    }

    void resize( const std::size_t n )
    {
        return m_data.resize( n );
    }

    friend void copy( const Buffer & buffer, StateType & state )
    {
        Engine::Backend::cpu::for_each_n(
            Engine::Backend::cpu::make_zip_iterator(
                buffer.m_data.begin(), state.spin.begin(), state.displacement.begin(), state.momentum.begin() ),
            buffer.m_data.size(),
            Engine::Backend::cpu::make_zip_function(
                []( const row_type & row, Vector3 & spin, Vector3 & displacement, Vector3 & momentum )
                {
                    spin         = Vector3{ row[0], row[1], row[2] };
                    displacement = Vector3{ row[3], row[4], row[5] };
                    momentum     = Vector3{ row[6], row[7], row[8] };
                } ) );
    }

    friend void copy( const StateType & state, Buffer & buffer )
    {
        Engine::Backend::cpu::transform(
            Engine::Backend::cpu::make_zip_iterator(
                begin( state.spin ), begin( state.displacement ), begin( state.momentum ) ),
            Engine::Backend::cpu::make_zip_iterator(
                end( state.spin ), end( state.displacement ), end( state.momentum ) ),
            begin( buffer.m_data ),
            Engine::Backend::cpu::make_zip_function(
                []( const Vector3 & s, const Vector3 & u, const Vector3 & p )
                { return std::array<scalar, 9>{ s[0], s[1], s[2], u[0], u[1], u[2], p[0], p[1], p[2] }; } ) );
    }

private:
    data_type m_data{};
};

} // namespace State

} // namespace SpinLattice

} // namespace IO
