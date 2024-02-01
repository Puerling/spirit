MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- Spirit --------------------------------------- <<" )


######### CMake Version ############################################
cmake_minimum_required( VERSION 3.10 FATAL_ERROR )
####################################################################
### Distinguish between Clang and AppleClang
cmake_policy( SET CMP0025 NEW )

set( SPIRIT_CUDA_ARCHITECTURES native )

#----- Write a compilation database (compile-commands.json)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

######### Meta information #########################################
#----- Get git revision
#----- Meta information about the project
set( META_PROJECT_NAME         "Spirit" )
#----- Compiler
set( META_COMPILER             "${CMAKE_CXX_COMPILER_ID}" )
set( META_COMPILER_VERSION     "${CMAKE_CXX_COMPILER_VERSION}" )
set( META_COMPILER_FULL        "${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER_VERSION})" )
####################################################################

######### Project name #############################################
message( STATUS ">> Building ${META_PROJECT_NAME} core library" )
project( ${META_PROJECT_NAME} )
####################################################################

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" )
######### Basic compiler-specific flags ############################
include(CompilerFlags)

######### CUDA setup ###############################################
include(CUDACompiler)

####################################################################
message( STATUS ">> Compiler information: ${META_COMPILER_FULL}" )


######### Gather headers and sources ###############################
set( SPIRIT_HEADERS )
set( SPIRIT_SOURCES )
### Add Subdirectories
add_subdirectory( ${PROJECT_SOURCE_DIR}/src )
add_subdirectory( ${PROJECT_SOURCE_DIR}/include )

set_source_files_properties( ${SPIRIT_SOURCES} PROPERTIES LANGUAGE CUDA )

####################################################################
set( libfmt_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/thirdparty/fmt/include )

### Flag needed to use fmt as header-only
set( SPIRIT_COMPILE_DEFINITIONS ${SPIRIT_COMPILE_DEFINITIONS} -DFMT_HEADER_ONLY )

set( Eigen_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/thirdparty )

############ Create object library #################################
add_library( ${META_PROJECT_NAME} OBJECT
    ${SPIRIT_HEADERS}
    ${SPIRIT_SOURCES}
)

set_target_properties( ${META_PROJECT_NAME} PROPERTIES
    CXX_STANDARD            17
    CXX_STANDARD_REQUIRED   ON
    CXX_EXTENSIONS          OFF )

target_include_directories( ${META_PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories( ${META_PROJECT_NAME} PRIVATE
    ${libfmt_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
)

target_link_libraries( ${META_PROJECT_NAME} INTERFACE ${CUDA_LIBRARIES} )
set_target_properties( ${META_PROJECT_NAME} PROPERTIES CUDA_SEPERABLE_COMPILATION ON )

######### C/C++ static library #####################################
message( STATUS ">> Building static C/C++ library ${META_PROJECT_NAME}_static" )
add_library( ${META_PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:${META_PROJECT_NAME}> )

set_target_properties( ${META_PROJECT_NAME}_static PROPERTIES
    CXX_STANDARD            17
    CXX_STANDARD_REQUIRED   ON
    CXX_EXTENSIONS          OFF )

### Forward public includes from the object library
target_include_directories( ${META_PROJECT_NAME}_static PUBLIC
    $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES> )
### Apply interface link dependencies from the object library
target_link_libraries( ${META_PROJECT_NAME}_static PRIVATE
    $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_LINK_LIBRARIES> )

### main executable ################################################
set_source_files_properties( ${CMAKE_CURRENT_LIST_DIR}/main.cpp PROPERTIES LANGUAGE CUDA )

add_executable("main" ${CMAKE_CURRENT_LIST_DIR}/main.cpp )
### Forward public includes from the object library
target_include_directories( "main" PUBLIC
    $<TARGET_PROPERTY:${META_PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES>
    ${libfmt_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
)

target_link_libraries("main" ${META_PROJECT_NAME}_static )
set_target_properties("main" PROPERTIES
    CXX_STANDARD             17
    CXX_STANDARD_REQUIRED    ON
    CXX_EXTENSIONS           OFF
)

target_compile_definitions( "main" PRIVATE ${SPIRIT_COMPILE_DEFINITIONS} )
