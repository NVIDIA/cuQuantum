# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)
project(cutensornet_example LANGUAGES C CXX CUDA)

include(GNUInstallDirs)
find_package(CUDA 11.0 REQUIRED)

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
        ${CUTENSORNET_ROOT}/include
  )
  target_link_directories(
    ${EXAMPLE_TARGET}
      PUBLIC
        ${CUTENSORNET_ROOT}/lib
        ${CUTENSORNET_ROOT}/lib64
  )
  target_link_libraries(
    ${EXAMPLE_TARGET}
      PUBLIC
        cutensornet
        $<$<BOOL:${MPI_FOUND}>:MPI::MPI_CXX>
  )
  if((${CUDA_VERSION_MAJOR} GREATER_EQUAL 12) OR (${CUDA_VERSION_MAJOR} EQUAL 11 AND ${CUDA_VERSION_MINOR} GREATER_EQUAL 8))
    set_target_properties(
      ${EXAMPLE_TARGET}
        PROPERTIES
          POSITION_INDEPENDENT_CODE ON
          CUDA_ARCHITECTURES "70;75;80;86;90")
  else()
    set_target_properties(
      ${EXAMPLE_TARGET}
        PROPERTIES
          POSITION_INDEPENDENT_CODE ON
          CUDA_ARCHITECTURES "70;75;80;86")
  endif()
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
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.reuse" tensornet_example_reuse.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.gradients" tensornet_example_gradients.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.amplitudes" high_level/amplitudes_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.amplitudes" high_level/amplitudes_mcg_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.expectation" high_level/expectation_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.marginal" high_level/marginal_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.sampler" high_level/sampling_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mps_amplitudes" high_level/mps_amplitudes_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mps_expectation" high_level/mps_expectation_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mps_marginal" high_level/mps_marginal_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mps_sampler" high_level/mps_sampling_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mps_sampler_mpo" high_level/mpo_mps_sampling_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mps_sampler_qft" high_level/qft_mps_sampling_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.svd" approxTN/tensor_svd_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.qr" approxTN/tensor_qr_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.gate" approxTN/gate_split_example.cu)
add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mps" approxTN/mps_example.cu)

find_package(MPI)
if (MPI_FOUND)
  add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mpi" tensornet_example_mpi.cu)
  add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mpi.auto" tensornet_example_mpi_auto.cu)
  add_cutensornet_example(cutensornet_examples "cuTENSORNet.example.tensornet.mpi.sampling" high_level/sampling_mpi_example.cu)
else ()
  message(WARNING "An MPI installation was not detected. Please install CUDA-aware MPI if you would like to build the distributed example(s).")
endif ()
