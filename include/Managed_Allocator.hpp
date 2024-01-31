#pragma once

#include <memory>
#include <stdexcept>
#include <fmt/core.h>

#include <cuda.h>
#include <cuda_runtime.h>

static void CudaHandleError( cudaError_t err, const char * file, int line, const char * function )
{
    if( err != cudaSuccess )
    {
        throw std::runtime_error(
                fmt::format( "{}:{} in function \'{}\':\n{:>49}{}", file, line, function, " ", cudaGetErrorString( err ) ) );
    }
}

#define CU_HANDLE_ERROR( err ) ( CudaHandleError( err, __FILE__, __LINE__, __func__ ) )

#define CU_CHECK_ERROR() ( CudaHandleError( cudaGetLastError(), __FILE__, __LINE__, __func__ ) )

#define CU_CHECK_AND_SYNC()                                                                                            \
    CU_CHECK_ERROR();                                                                                                  \
    CU_HANDLE_ERROR( cudaDeviceSynchronize() )

template<class T>
class managed_allocator : public std::allocator<T>
{
public:
    using value_type = T;

    template<typename Tp1>
    struct rebind
    {
        using other = managed_allocator<Tp1>;
    };

    value_type * allocate( size_t n )
    {
        value_type * result = nullptr;

        CU_HANDLE_ERROR( cudaMallocManaged( &result, n * sizeof( value_type ) ) );

        return result;
    }

    void deallocate( value_type * ptr, size_t )
    {
        CU_HANDLE_ERROR( cudaFree( ptr ) );
    }

    managed_allocator() throw() : std::allocator<T>() {}
    managed_allocator( const managed_allocator & a ) throw() : std::allocator<T>( a ) {}
    template<class U>
    managed_allocator( const managed_allocator<U> & a ) throw() : std::allocator<T>( a )
    {
    }
    ~managed_allocator() throw() = default;
};
