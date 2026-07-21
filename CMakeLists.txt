cmake_minimum_required(VERSION 3.20)
project(antipodal CXX)

# C++20 is required by the header-only API (concepts, ranges, designated initializers)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional dependencies — code paths gate on the ANTIPODAL_HAS_* defines below
option(ANTIPODAL_WITH_TBB    "Enable TBB-based parallel dispatcher"           ON)
option(ANTIPODAL_WITH_EMBREE "Enable Embree (used by future ray-based paths)" ON)
option(ANTIPODAL_WITH_OPENGL "Enable the raw-OpenGL compute-shader evaluator" ON)
option(ANTIPODAL_BUILD_TESTS "Build the antipodal test suite"                 ON)
option(ANTIPODAL_BUILD_EXAMPLES "Build the antipodal example programs"        ON)

# Header-only: consumers get includes + propagated deps via the INTERFACE target
add_library(antipodal INTERFACE)
add_library(antipodal::antipodal ALIAS antipodal)

target_include_directories(antipodal INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_compile_features(antipodal INTERFACE cxx_std_20)

if(ANTIPODAL_WITH_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(antipodal INTERFACE TBB::tbb)
    target_compile_definitions(antipodal INTERFACE ANTIPODAL_HAS_TBB=1)
endif()

if(ANTIPODAL_WITH_EMBREE)
    find_package(embree 4 REQUIRED)
    target_link_libraries(antipodal INTERFACE embree)
    target_compile_definitions(antipodal INTERFACE ANTIPODAL_HAS_EMBREE=1)
endif()

# The OpenGL evaluator is header-only and loads GL entry points through a
# caller-supplied loader, so the INTERFACE target links nothing extra here.
# The optional headless-context helper (gl_headless_context.hh) needs platform
# libs (Windows: opengl32/gdi32/user32; Linux: GL + EGL); consumers that use it
# link those themselves, as the test target does below.
if(ANTIPODAL_WITH_OPENGL)
    target_compile_definitions(antipodal INTERFACE ANTIPODAL_HAS_OPENGL=1)
endif()

# Tests only build when antipodal is the top-level project (skipped when consumed via add_subdirectory)
if(ANTIPODAL_BUILD_TESTS AND PROJECT_IS_TOP_LEVEL)
    enable_testing()

    add_executable(antipodal-tests
        test/main.cc
        test/test_common.cc
        test/test_intersector.cc
        test/test_gwn_mesh.cc
        test/test_gwn_mesh_robust.cc
        test/test_gwn_mesh_opengl.cc
        test/test_gwn_parametric.cc
        test/fwd_diff_diff/test_fdd_basics.cc
        test/fwd_diff_diff/test_fdd_math.cc
        test/fwd_diff/test_fd_basics.cc
        test/fwd_diff/test_fd_math.cc
        test/math/test_integrate_gk15.cc
        test/math/test_integrate_gl7.cc
    )
    target_link_libraries(antipodal-tests PRIVATE antipodal)
    target_include_directories(antipodal-tests PRIVATE test)

    # The OpenGL test spins up a headless context via gl_headless_context.hh,
    # which needs platform GL/EGL libraries linked into the test executable.
    if(ANTIPODAL_WITH_OPENGL)
        if(WIN32)
            target_link_libraries(antipodal-tests PRIVATE opengl32 gdi32 user32)
        elseif(UNIX)
            find_package(OpenGL REQUIRED)
            find_library(ANTIPODAL_EGL_LIB EGL REQUIRED)
            target_link_libraries(antipodal-tests PRIVATE OpenGL::GL ${ANTIPODAL_EGL_LIB})
        endif()
    endif()

    if(MSVC)
        target_compile_options(antipodal-tests PRIVATE /W4)
    else()
        target_compile_options(antipodal-tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME antipodal-tests COMMAND antipodal-tests)
endif()

# Examples only build when antipodal is the top-level project
if(ANTIPODAL_BUILD_EXAMPLES AND PROJECT_IS_TOP_LEVEL)
    find_package(Stb REQUIRED) # vcpkg 'stb' port -> Stb_INCLUDE_DIR (stb_image_write.h)

    add_executable(antipodal-example-slice examples/main.cc)
    target_link_libraries(antipodal-example-slice PRIVATE antipodal)
    target_include_directories(antipodal-example-slice PRIVATE ${Stb_INCLUDE_DIR})
endif()
