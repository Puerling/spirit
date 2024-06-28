add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/interaction)

set(SOURCE_SPIRIT_ENGINE_SPINLATTICE_INTERACTION
    ${SOURCE_SPIRIT_ENGINE_SPINLATTICE_INTERACTION}
    PARENT_SCOPE
)

set(SOURCE_SPIRIT_ENGINE_SPINLATTICE
    ${SOURCE_SPIRIT_ENGINE_SPINLATTICE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_LLG.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE # needed so the change of ${SOURCE} will persist to the parent scope
)

if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    set_source_files_properties(
        ${CMAKE_CURRENT_LIST_DIR}/Method_LLG.cpp
        PROPERTIES LANGUAGE CUDA )
endif()

