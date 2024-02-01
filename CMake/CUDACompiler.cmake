######### CUDA setup ###############################################
enable_language( CUDA )

find_package( CUDAToolkit 11 REQUIRED )

if( NOT DEFINED CMAKE_CUDA_ARCHITECTURES )
    set( CMAKE_CUDA_ARCHITECTURES ${SPIRIT_CUDA_ARCHITECTURES} )
endif()

set( _CUDA_OPTS "" )

if( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA" )
    string(APPEND _CUDA_OPTS
        "-lineinfo;"
        "--expt-relaxed-constexpr;"
        "--expt-extended-lambda;"
        "--diag-suppress=20011;"
        "--diag-suppress=20012;"
        "--diag-suppress=20014;"
        "--diag-suppress=1675;"
        "--display-error-number;"
    )
elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC") 
    string(APPEND _CUDA_OPTS
        "-cuda;"
    )
endif()

set( META_COMPILER         "${META_COMPILER}" )
set( META_COMPILER_VERSION "${META_COMPILER_VERSION} and CUDA ${CUDA_VERSION}" )
set( META_COMPILER_FULL    "${META_COMPILER_FULL} and (CUDA ${CUDA_VERSION}) for cuda arch \\\"${CMAKE_CUDA_ARCHITECTURES}\\\"" )

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${_CUDA_OPTS}>")

if( NOT DEFINED CMAKE_CUDA_STANDARD )
    set( CMAKE_CUDA_STANDARD 17 )
    set( CMAKE_CUDA_STANDARD_REQUIRED ON )
endif()

message( STATUS ">> Using CUDA. Flags: ${CMAKE_CUDA_FLAGS}" )
message( STATUS ">> CUDA toolkit path: ${CUDAToolkit_LIBRARY_ROOT}" )
message( STATUS ">> CUDA libraries: ${CUDA_LIBRARIES}" )
