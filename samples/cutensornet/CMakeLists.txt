# 
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
# 

# ---[ Check cmake version.
cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)

include(GNUInstallDirs)

# ---[ Project specIFication.
project(cutensornet_example LANGUAGES C CXX CUDA)

# ##########################################
# cuTENSOR_example build mode
# ##########################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# cuTENSOR_example dependencies
# ##########################################

find_package(CUDA 10.1 REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
  message(FATAL_ERROR "CUTENSOR_ROOT not set!")
else()
  if(DEFINED ENV{CUTENSOR_ROOT})
    set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
  endif()
  message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
  if(NOT EXISTS ${CUTENSOR_ROOT})
    message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
  endif()
endif()

if(NOT DEFINED ENV{CUTENSORNET_ROOT} AND NOT DEFINED CUTENSORNET_ROOT)
    message(FATAL_ERROR "CUTENSORNET_ROOT not set!")
else()
  if(DEFINED ENV{CUTENSORNET_ROOT})
      set(CUTENSORNET_ROOT "$ENV{CUTENSORNET_ROOT}")
  endif()
  message("-- Looking for cuTENSORNet in ${CUTENSORNET_ROOT}")
  if(NOT EXISTS ${CUTENSORNET_ROOT})
      message(FATAL_ERROR "Cannot find CUTENSORNET_ROOT")
  endif()
endif()

if(NOT TARGET cutensor)
  add_library(cutensor SHARED IMPORTED)
  if(WIN32)
    set(CUTENSOR_LIBRARY_NAME "cutensor.dll")
    set(CUTENSOR_LIBRARY_DEF "cutensor.lib")
  else()
    set(CUTENSOR_LIBRARY_NAME "libcutensor.so")
  endif()
  set_target_properties(cutensor PROPERTIES
      IMPORTED_LOCATION "${CUTENSOR_ROOT}/lib/11/${CUTENSOR_LIBRARY_NAME}"
      IMPORTED_IMPLIB "${CUTENSOR_ROOT}/lib/11/${CUTENSOR_LIBRARY_DEF}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
endif()

if(NOT TARGET cutensornet)
  add_library(cutensornet SHARED IMPORTED)
  if(WIN32)
  set(CUTENSORNET_LIBRARY_NAME "cutensornet.dll")
    set(CUTENSORNET_LIBRARY_DEF "cutensornet.lib")
  else()
    set(CUTENSORNET_LIBRARY_NAME "libcutensornet.so")
  endif()
  set_target_properties(cutensornet PROPERTIES
    IMPORTED_LOCATION "${CUTENSORNET_ROOT}/lib/${CUTENSORNET_LIBRARY_NAME}"
    IMPORTED_IMPLIB "${CUTENSORNET_ROOT}/lib/${CUTENSORNET_LIBRARY_DEF}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSORNET_ROOT}/include")
endif()

# Installation directories
if(NOT DEFINED CUTENSORNET_EXAMPLE_BINARY_INSTALL_DIR)
    message(WARNING "CUTENSORNET_EXAMPLE_BINARY_INSTALL_DIR not set, setting to cutensornet_example/bin")
    set(CUTENSORNET_EXAMPLE_BINARY_INSTALL_DIR "cutensornet_example/bin")
endif()

# ##########################################
# cuTENSOR_example building flags
# ##########################################

# Global CXX/CUDA flags

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS_ARCH_SM70 "-gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS_ARCH_SM75 "-gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS_ARCH_SM80 "-gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80")
set(CMAKE_CUDA_FLAGS_ARCH "${CMAKE_CUDA_FLAGS_ARCH_SM70} ${CMAKE_CUDA_FLAGS_ARCH_SM75} ${CMAKE_CUDA_FLAGS_ARCH_SM80}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_ARCH}")

# ##########################################
# cuTENSOR_example target
# ##########################################

function(add_cutensornet_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_include_directories(${EXAMPLE_TARGET}
        PUBLIC
            ${CUDA_INCLUDE_DIRS}
            ${CUTENSOR_ROOT}/include
            ${CUTENSORNET_ROOT}/include
    )
    target_link_libraries(${EXAMPLE_TARGET}
        PUBLIC
            cutensornet
            cutensor
            cudart
            cublasLt
    )
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
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

add_custom_target(cutensornet_examples)

add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet" tensornet_example.cu)

# ##########################################
# cuTENSOR_example directories
# ##########################################

# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ##########################################
# Install examples
# ##########################################

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF()
