# 
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ---[ Check cmake version.
cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)

# ---[ Project specification.
project(cutensornet_example LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

# ##########################################
# cutensornet_example build mode
# ##########################################

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build.")
  set_property(
    CACHE
    CMAKE_BUILD_TYPE
    PROPERTY
      STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
else ()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif ()

# ##########################################
# cutensornet_example check-env utilities
# ##########################################

function(set_with_fallback VARIABLE FALLBACK)
  if (NOT DEFINED ${VARIABLE} OR ${VARIABLE} STREQUAL "")
    set(${VARIABLE} $ENV{${VARIABLE}} CACHE INTERNAL ${VARIABLE})
    if (${VARIABLE} STREQUAL "")
      if (NOT ${FALLBACK} STREQUAL "")
        set(${VARIABLE} $ENV{${FALLBACK}} CACHE INTERNAL ${VARIABLE})
      endif ()
    endif ()
  endif ()
endfunction()

# ##########################################
# cutensornet_example dependencies
# ##########################################

set_with_fallback(CUTENSOR_ROOT NONE)

if (CUTENSOR_ROOT STREQUAL "")
    message(FATAL_ERROR "Please set the environment variables CUTENSOR_ROOT to the path of the cuTENSOR installation.")
endif ()

set_with_fallback(CUTENSORNET_ROOT CUQUANTUM_ROOT)

if (CUTENSORNET_ROOT STREQUAL "")
    message(FATAL_ERROR "Please set the environment variables CUTENSORNET_ROOT or CUQUANTUM_ROOT to the path of the cuQuantum installation.")
endif ()

message(STATUS "Using CUTENSORNET_ROOT = ${CUTENSORNET_ROOT}")

# ##########################################
# cutensornet_example global flags
# ##########################################

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ##########################################
# cutensornet_example utility function
# ##########################################

function(add_cutensornet_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
  list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
  target_include_directories(
    ${EXAMPLE_TARGET}
      PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${CUTENSOR_ROOT}/include
        ${CUTENSORNET_ROOT}/include
  )
  target_link_directories(
    ${EXAMPLE_TARGET}
      PUBLIC
        ${CUTENSOR_ROOT}/lib/11
        ${CUTENSORNET_ROOT}/lib
        ${CUTENSORNET_ROOT}/lib64
  )
  target_link_libraries(
    ${EXAMPLE_TARGET}
      PUBLIC
        cutensornet
        cutensor
        cudart
        cublasLt
  )
  set_target_properties(
    ${EXAMPLE_TARGET}
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES
          "70;75;80"
  )
  # Install example
  install(
      TARGETS ${EXAMPLE_TARGET}
      RUNTIME
      DESTINATION ${CUTENSORNET_EXAMPLE_BINARY_INSTALL_DIR}
      PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
  )
  add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ##########################################
# cutensornet_example add all targets
# ##########################################

add_custom_target(cutensornet_examples)

add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet" tensornet_example.cu)
