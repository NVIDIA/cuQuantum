# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.22)
project(cudensitymat_jax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python executable: ${Python3_EXECUTABLE}")

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA toolkit directory: ${CUDAToolkit_INCLUDE_DIRS}")

# Find XLA directory
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import jax; print(jax.ffi.include_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE XLA_DIR
)
if(NOT XLA_DIR)
    message(FATAL_ERROR "XLA directory not found")
else()
    message(STATUS "XLA directory: ${XLA_DIR}")
endif()

# Find pybind11 directory
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE pybind11_INCLUDE_DIR
)
if(NOT pybind11_INCLUDE_DIR)
    message(FATAL_ERROR "Pybind11 include directory not found")
else()
    message(STATUS "Pybind11 include directory: ${pybind11_INCLUDE_DIR}")
endif()

set(pybind11_DIR ${pybind11_INCLUDE_DIR}/../share/cmake/pybind11)
find_package(pybind11 REQUIRED)

# FIXME: This should be made more robust and moved to setup.py.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os; import cuquantum; print(os.path.dirname(cuquantum.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE CUQUANTUM_PYTHON_ROOT
)

if(DEFINED ENV{CUDENSITYMAT_ROOT})
    set(CUDENSITYMAT_ROOT $ENV{CUDENSITYMAT_ROOT})
else()
    set(CUDENSITYMAT_ROOT ${CUQUANTUM_PYTHON_ROOT})
endif()

message(STATUS "cuQuantum Python directory: ${CUQUANTUM_PYTHON_ROOT}")
message(STATUS "cuDensityMat directory: ${CUDENSITYMAT_ROOT}")

pybind11_add_module(
    ${PROJECT_NAME}
    cudensitymat_jax.cpp
    pybind.cpp
)
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
    ${XLA_DIR}
    ${pybind11_INCLUDE_DIR}
    ${CUDENSITYMAT_ROOT}/include
)

find_library(
    CUDENSITYMAT_LIBRARY
    NAMES libcudensitymat.so
          libcudensitymat.so.0
    HINTS ${CUDENSITYMAT_ROOT}/lib64
          ${CUDENSITYMAT_ROOT}/lib
)

if(NOT CUDENSITYMAT_LIBRARY)
    message(FATAL_ERROR "cuDensityMat library not found")
else()
    message(STATUS "cuDensityMat library: ${CUDENSITYMAT_LIBRARY}")
endif()

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        BUILD_RPATH "$ORIGIN"
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    CUDA::cudart_static
    ${CUDENSITYMAT_LIBRARY}
)
