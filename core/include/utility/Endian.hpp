#pragma once

namespace Utility
{

enum struct endianness
{
    little = 0,
    big    = 1,
};

inline endianness get_system_endianness()
{
    const int value{ 0x01 };
    const void * address{ static_cast<const void *>( &value ) };
    const unsigned char * least_significant_address{ static_cast<const unsigned char *>( address ) };

    return ( *least_significant_address == 0x01 ) ? endianness::little : endianness::big;
}

static const endianness ENDIAN = get_system_endianness();

} // namespace Utility
