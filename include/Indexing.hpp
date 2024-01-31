#pragma once

#include <Vectormath_Defines.hpp>

namespace Indexing
{

__inline__ __device__ bool cu_check_atom_type( int atom_type )
{
    // Else we just return true
    return true;
}

__inline__ __device__ bool cu_check_atom_type( const int atom_type, const int reference_type )
{
    // Else we just return true
    return true;
}

// Check atom types
inline bool check_atom_type( int atom_type )
{
    // Else we just return true
    return true;
}
inline bool check_atom_type( int atom_type, int reference_type )
{
    // Else we just return true
    return true;
}

} // namespace Indexing
