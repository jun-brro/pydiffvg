cmake_minimum_required(VERSION 3.12)

project(diffvg VERSION 0.0.1 DESCRIPTION "Differentiable Vector Graphics")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Python development
if(WIN32)
    find_package(Python 3.6 COMPONENTS Development REQUIRED)
else()
    find_package(Python 3.7 COMPONENTS Development REQUIRED)
endif()

# Use system-installed pybind11 instead of bundled submodule
find_package(pybind11 CONFIG REQUIRED)
# add_subdirectory(pybind11)  # deprecated when using system pybind11

option(DIFFVG_CUDA "Build diffvg with GPU code path?" ON)

if(DIFFVG_CUDA)
    message(STATUS "Build with CUDA support")
    find_package(CUDA 10 REQUIRED)
    set(CMAKE_CUDA_STANDARD 11)
    if(NOT WIN32)
        # Enforce C++11 for NVCC
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    endif()
else()
    message(STATUS "Build without CUDA support")
    find_package(Thrust REQUIRED)
endif()

# Include Python and pybind11 headers
include_directories(${PYTHON_INCLUDE_DIRS})
# pybind11::pybind11 target provides include paths

# Linking libraries
if(DIFFVG_CUDA)
    link_directories(${CUDA_LIBRARIES})
else()
    include_directories(${THRUST_INCLUDE_DIR})
endif()

# Compiler flags
if(NOT MSVC)
    add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
else()
    add_compile_options(/Wall /Zi)
    add_link_options(/DEBUG)
endif()
if(NOT DIFFVG_CUDA)
    add_compile_options("-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
endif()

# Source files
set(SRCS
    atomic.h
    color.h
    cdf.h
    cuda_utils.h
    diffvg.h
    edge_query.h
    filter.h
    matrix.h
    parallel.h
    pcg.h
    ptr.h
    sample_boundary.h
    scene.h
    shape.h
    solve.h
    vector.h
    within_distance.h
    winding_number.h
    atomic.cpp
    color.cpp
    diffvg.cpp
    parallel.cpp
    scene.cpp
    shape.cpp
)

# Build diffvg module
if(DIFFVG_CUDA)
    add_compile_definitions(COMPILE_WITH_CUDA)
    set_source_files_properties(
        diffvg.cpp
        scene.cpp
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
    )
    cuda_add_library(diffvg MODULE ${SRCS})
else()
    add_library(diffvg MODULE ${SRCS})
endif()

# macOS dynamic lookup hack
if(APPLE)
    set(DYNAMIC_LOOKUP "-undefined dynamic_lookup")
endif()

target_link_libraries(diffvg PRIVATE ${DYNAMIC_LOOKUP} pybind11::pybind11)

# RPATH settings
set_target_properties(diffvg PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(diffvg PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
if(UNIX AND NOT APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "@loader_path")
endif()

# Ensure C++11 standard and keep assertions
set_property(TARGET diffvg PROPERTY CXX_STANDARD 11)
string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

# (Optional) TensorFlow support if found
if(NOT WIN32)
    find_package(TensorFlow)
    if(TensorFlow_FOUND)
        add_subdirectory(pydiffvg_tensorflow/custom_ops)
    else()
        message(STATUS "Building without TensorFlow support (not found)")
    endif()
endif()
