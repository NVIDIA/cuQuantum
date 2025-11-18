# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.27.0 FATAL_ERROR)

project(stabilizer_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(CuStabilizer REQUIRED)

add_executable(custabilizer_demo custabilizer_demo.cpp)

target_link_libraries(custabilizer_demo PRIVATE
  cuStabilizer::cuStabilizer
  CUDA::cudart
)

set_target_properties(custabilizer_demo PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

