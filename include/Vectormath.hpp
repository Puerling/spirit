#pragma once

#include <Vectormath_Defines.hpp>

namespace Vectormath
{

/////////////////////////////////////////////////////////////////
//////// Vectormath-like operations

// sets sf := s
// sf is a scalarfield
// s is a scalar
void fill( scalarfield & sf, scalar s );

void fill( scalarfield & sf, scalar s, const intfield & mask );

// sets vf := v
// vf is a vectorfield
// v is a vector
void fill( vectorfield & vf, const Vector3 & v );
void fill( vectorfield & vf, const Vector3 & v, const intfield & mask );

} // namespace Vectormath
