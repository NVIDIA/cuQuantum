##
## SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## 1. Redistributions of source code must retain the above copyright notice, this
## list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above copyright notice,
## this list of conditions and the following disclaimer in the documentation
## and/or other materials provided with the distribution.
##
## 3. Neither the name of the copyright holder nor the names of its
## contributors may be used to endorse or promote products derived from
## this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
## FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##

# Single-GPU example only (assuming A100 GPU):
# cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
#
# Multi-GPU example with MPI (assuming A100 GPU):
# cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 -DENABLE_MPI=TRUE

cmake_minimum_required(VERSION 3.22.0 FATAL_ERROR)

# Guard against automatically setting CMAKE_CUDA_ARCHITECTURES by CMP0104
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES_DEFINED TRUE)
else()
  set(CMAKE_CUDA_ARCHITECTURES_DEFINED FALSE)
endif()

project(example_cuDensityMat LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES_DEFINED)
  if(
    (${CUDAToolkit_VERSION_MAJOR} LESS 12) OR 
    (${CUDAToolkit_VERSION_MAJOR} EQUAL 12 AND ${CUDAToolkit_VERSION_MINOR} LESS 8)
  )
    set(CMAKE_CUDA_ARCHITECTURES 70-real;75-real;80-real;86-real;90)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70-real;75-real;80-real;86-real;90-real;100-real;120-real;100-virtual)
  endif()
endif()
message(STATUS "Target CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")


# Find MPI
if(ENABLE_MPI)
  find_package(MPI)
  if(MPI_FOUND)
    message(STATUS "MPI found: Building a multi-GPU MPI-enabled version of the examples!")
  else()
    message(STATUS "MPI not found: Building a single-GPU version of the examples!")
  endif()
else()
  message(STATUS "Building a single-GPU version of the examples!")
endif()

# Find cuTensor
if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
  message(FATAL_ERROR "CUTENSOR_ROOT is not set!")
else()
  if(DEFINED ENV{CUTENSOR_ROOT})
    set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
  endif()
  message(STATUS "Looking for cuTensor in ${CUTENSOR_ROOT}")
  if(NOT EXISTS ${CUTENSOR_ROOT})
    message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
  endif()
endif()
set(CUTENSOR_INC_DIR ${CUTENSOR_ROOT}/include)
set(CUTENSOR_LIB_DIR ${CUTENSOR_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR})

# Find cuDensityMat
if(NOT DEFINED ENV{CUDENSITYMAT_ROOT} AND NOT DEFINED CUDENSITYMAT_ROOT)
  message(FATAL_ERROR "CUDENSITYMAT_ROOT is not set!")
else()
  if(DEFINED ENV{CUDENSITYMAT_ROOT})
    set(CUDENSITYMAT_ROOT "$ENV{CUDENSITYMAT_ROOT}")
  endif()
  message(STATUS "Looking for cuDensityMat in ${CUDENSITYMAT_ROOT}")
  if(NOT EXISTS ${CUDENSITYMAT_ROOT})
    message(FATAL_ERROR "Cannot find CUDENSITYMAT_ROOT")
  endif()
endif()
set(CUDENSITYMAT_INC_DIR ${CUDENSITYMAT_ROOT}/include)
set(CUDENSITYMAT_LIB_DIR ${CUDENSITYMAT_ROOT}/lib)

# Single-GPU example
add_executable(operator_action_example
  operator_action_example.cpp
)

set_target_properties(operator_action_example
  PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(operator_action_example
  PUBLIC
    ${CUDENSITYMAT_INC_DIR}
    ${CUTENSOR_INC_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_directories(operator_action_example
  PUBLIC
    ${CUDENSITYMAT_LIB_DIR}
    ${CUTENSOR_LIB_DIR}
)

target_link_libraries(operator_action_example
  PUBLIC
    cudensitymat
    cutensor
    CUDA::cublas
    CUDA::cudart
)

# Multi-GPU example
if(MPI_FOUND)

  add_executable(operator_action_mpi_example
    operator_action_mpi_example.cpp
  )

  set_target_properties(operator_action_mpi_example
    PROPERTIES
      CXX_STANDARD ${CMAKE_CXX_STANDARD}
      CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
      POSITION_INDEPENDENT_CODE ON
  )

  target_compile_definitions(operator_action_mpi_example
    PUBLIC
      MPI_ENABLED
  )

  target_include_directories(operator_action_mpi_example
    PUBLIC
      ${CUDENSITYMAT_INC_DIR}
      ${CUTENSOR_INC_DIR}
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
      ${MPI_CXX_INCLUDE_DIRS}
  )

  target_link_directories(operator_action_mpi_example
    PUBLIC
      ${CUDENSITYMAT_LIB_DIR}
      ${CUTENSOR_LIB_DIR}
  )

  target_link_libraries(operator_action_mpi_example
    PUBLIC
      cudensitymat
      cutensor
      CUDA::cublas
      CUDA::cudart
      MPI::MPI_CXX
  )

endif() # MPI_FOUND
