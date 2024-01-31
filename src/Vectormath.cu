#include <Vectormath.hpp>
#include <Vectormath_Defines.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Vectormath
{

__global__ void cu_fill( scalar * sf, scalar s, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        sf[idx] = s;
    }
}
void fill( scalarfield & sf, scalar s )
{
    unsigned int n = sf.size();
    cu_fill<<<( n + 1023 ) / 1024, 1024>>>( sf.data(), s, n );
    CU_CHECK_AND_SYNC();
}
__global__ void cu_fill_mask( scalar * sf, scalar s, const int * mask, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        sf[idx] = mask[idx] * s;
    }
}
void fill( scalarfield & sf, scalar s, const intfield & mask )
{
    unsigned int n = sf.size();
    cu_fill_mask<<<( n + 1023 ) / 1024, 1024>>>( sf.data(), s, mask.data(), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_fill( Vector3 * vf1, Vector3 v2, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        vf1[idx] = v2;
    }
}
void fill( vectorfield & vf, const Vector3 & v )
{
    unsigned int n = vf.size();
    cu_fill<<<( n + 1023 ) / 1024, 1024>>>( vf.data(), v, n );
    CU_CHECK_AND_SYNC();
}
__global__ void cu_fill_mask( Vector3 * vf1, Vector3 v2, const int * mask, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        vf1[idx] = v2;
    }
}
void fill( vectorfield & vf, const Vector3 & v, const intfield & mask )
{
    unsigned int n = vf.size();
    cu_fill_mask<<<( n + 1023 ) / 1024, 1024>>>( vf.data(), v, mask.data(), n );
    CU_CHECK_AND_SYNC();
}

} // namespace Vectormath
