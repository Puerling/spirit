#pragma once

#include <engine/Span.hpp>

#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace Utility
{

namespace b64
{

namespace detail
{

using triplet_t = std::bitset<3 * 8>;
using quad_t    = std::bitset<4 * 8>;

static constexpr triplet_t b64_triplet{ 0b0011'1111 };
static constexpr std::byte b64_byte{ 0b0011'1111 };
static constexpr quad_t byte_quad{ 0b1111'1111 };

static constexpr char B64_PAD = '=';

constexpr std::array<std::byte, 256> make_decoding_table() noexcept
{
    constexpr std::array<std::uint8_t, 256> decoding = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x3E, 0xFF, 0xFF, 0xFF, 0x3F,
        0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF,
        0xFF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
        0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,

        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    };

    std::array<std::byte, 256> result{};
    // transform
    for( std::size_t i = 0; i < result.size(); ++i )
        result[i] = std::byte( decoding[i] );

    return result;
}

constexpr std::array<char, 64> make_encoding_table() noexcept
{
    constexpr std::array<char, 65> encoding = { "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                "abcdefghijklmnopqrstuvwxyz"
                                                "0123456789+/" };
    std::array<char, 64> result{};
    // copy
    for( std::size_t i = 0; i < result.size(); ++i )
        result[i] = encoding[i];

    return result;
}

template<typename T>
constexpr T next_multiple( const T number, const std::size_t multiple = 0 ) noexcept
{
    if( multiple == 0 )
        return number;

    const auto remainder = number % multiple;
    if( remainder == 0 )
        return number;

    return number + multiple - remainder;
};

} // namespace detail

static constexpr std::array<char, 64> to_base64 = detail::make_encoding_table();

static constexpr std::array<std::byte, 256> from_base64 = detail::make_decoding_table();

namespace detail
{

template<typename OutputIt>
inline void encode( const std::byte * data, std::size_t size, OutputIt dest )
{
    static_assert( std::is_assignable_v<std::decay_t<decltype( *dest )>, char> );

    const std::size_t pad_width = ( size % 3 != 0 ) ? 3 - ( size % 3 ) : 0;
    {
        triplet_t state{};
        int nbits = 0;
        for( ; size > 0; ++data, --size )
        {
            // read byte
            state = ( state << 8 ) | triplet_t{ std::to_integer<std::uint32_t>( *data ) };
            nbits += 8;

            // consume to encode 6 bits at a time
            while( nbits >= 6 )
            {
                nbits -= 6;
                *dest++ = to_base64[( ( state >> nbits ) & b64_triplet ).to_ulong()];
            }
        }
        // consume remaining bits
        if( nbits > 0 )
        {
            *dest++ = to_base64[( ( ( state << 8 ) >> ( nbits + 2 ) ) & b64_triplet ).to_ulong()];
        }
    }
    // padding
    for( std::size_t i = 0; i < pad_width; ++i )
        *dest++ = B64_PAD;
}

} // namespace detail

template<typename T, typename OutputIt>
void encode( const T * data, const std::size_t size, OutputIt dest )
{
    std::vector<std::byte> bytes( sizeof( T ) * size );
    std::memcpy( bytes.data(), data, bytes.size() );

    detail::encode( bytes.data(), bytes.size(), dest );
}

namespace detail
{

template<typename OutputIt>
void decode( const std::string_view data, OutputIt dest, std::size_t dest_maxsize )
{
    static_assert( std::is_assignable_v<std::decay_t<decltype( *dest )>, std::byte> );

    quad_t state{};
    int nbits = 0;
    for( const auto byte : data )
    {
        // read bytes
        if( std::isspace( byte ) )
        {
            continue;
        }

        state = ( state << 6 )
                | quad_t( std::to_integer<std::uint8_t>( from_base64[static_cast<std::uint8_t>( byte )] & b64_byte ) );
        nbits += 6;

        // consume bytes to write to dest
        while( nbits >= 8 )
        {
            nbits -= 8;
            *dest++ = std::byte( ( ( state >> nbits ) & byte_quad ).to_ulong() );

            if( --dest_maxsize == 0 )
                return;
        };
    }

    // consume remaining bits
    if( nbits > 0 )
    {
        if( const auto rem = nbits % 8; rem != 0 )
        {
            state <<= ( 8 - rem );
            nbits += ( 8 - rem );
        }
        while( nbits >= 8 )
        {
            nbits -= 8;
            *dest++ = std::byte( ( ( state >> nbits ) & byte_quad ).to_ulong() );

            if( --dest_maxsize == 0 )
                return;
        };
    }
}

} // namespace detail

template<typename T>
void decode( const std::string_view data, T * dest, const std::size_t size )
{
    std::vector<std::byte> bytes( 0 );
    bytes.reserve( sizeof( T ) * detail::next_multiple( size, 4 ) );

    detail::decode( data, std::back_inserter( bytes ), sizeof( T ) * size );

    std::memcpy( dest, bytes.data(), std::min( bytes.size(), sizeof( T ) * size ) );
}

} // namespace b64

} // namespace Utility
