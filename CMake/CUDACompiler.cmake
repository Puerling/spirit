######### CUDA setup ###############################################
enable_language( CUDA )

find_package( CUDA 8 REQUIRED )

# add_compile_options($<COMPILE_LANGUAGE:CUDA>:)
if( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA" ) 
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -arch=sm_${SPIRIT_CUDA_ARCH} --expt-relaxed-constexpr --expt-extended-lambda" )
    ### Deactivate CUDA warning inside Eigen such as "warning: __host__ annotation is ignored on a function("Quaternion") that is explicitly defaulted on its first declaration"
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=20011 --diag-suppress=20012 --diag-suppress=20014 --diag-suppress=1675")
    ### Display warning number when writing a warning
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --display-error-number" )

    set( META_COMPILER         "${META_COMPILER} and nvcc" )
    set( META_COMPILER_VERSION "${META_COMPILER_VERSION} and ${CUDA_VERSION}" )
    set( META_COMPILER_FULL    "${META_COMPILER_FULL} and nvcc (${CUDA_VERSION}) for cuda arch \\\"sm_${SPIRIT_CUDA_ARCH}\\\"" )

elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC") 
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -cuda" )

    set( META_COMPILER         "${META_COMPILER}" )
    set( META_COMPILER_VERSION "${META_COMPILER_VERSION} and ${CUDA_VERSION}" )
    set( META_COMPILER_FULL    "${META_COMPILER_FULL} and nvcc (${CUDA_VERSION}) for cuda arch \\\"sm_${SPIRIT_CUDA_ARCH}\\\"" )

else()
    set( CMAKE_CUDA_ARCHITECTURES ${SPIRIT_CUDA_ARCH} )
    set( META_COMPILER         "${META_COMPILER}" )
    set( META_COMPILER_VERSION "${META_COMPILER_VERSION} and CUDA ${CUDA_VERSION}" )
    set( META_COMPILER_FULL    "${META_COMPILER_FULL} and (CUDA ${CUDA_VERSION}) for cuda arch \\\"${SPIRIT_CUDA_ARCH}\\\"" )
endif()

if( NOT DEFINED CMAKE_CUDA_STANDARD )
    set( CMAKE_CUDA_STANDARD 17 )
    set( CMAKE_CUDA_STANDARD_REQUIRED ON )
endif()

message( STATUS ">> Using CUDA. Flags: ${CMAKE_CUDA_FLAGS}" )
message( STATUS ">> CUDA toolkit path: ${CUDA_TOOLKIT_ROOT_DIR}" )
message( STATUS ">> CUDA libraries: ${CUDA_LIBRARIES}" )
