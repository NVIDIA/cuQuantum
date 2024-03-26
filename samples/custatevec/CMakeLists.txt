# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)
project(custatevec_example LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

# ##########################################
# custatevec_example build mode
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
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# custatevec_example check-env utilities
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
# custatevec_example dependencies
# ##########################################

set_with_fallback(CUSTATEVEC_ROOT CUQUANTUM_ROOT)

if (CUSTATEVEC_ROOT STREQUAL "")
    message(FATAL_ERROR "Please set the environment variables CUSTATEVEC_ROOT or CUQUANTUM_ROOT to the path of the cuQuantum installation.")
endif ()

message(STATUS "Using CUSTATEVEC_ROOT = ${CUSTATEVEC_ROOT}")

# ##########################################
# custatevec_example global flags
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

set(CMAKE_CUDA_FLAGS_ARCH_SM70 "-gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS_ARCH_SM75 "-gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS_ARCH_SM80 "-gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80")
set(CMAKE_CUDA_FLAGS_ARCH_SM90 "-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90")
set(CMAKE_CUDA_FLAGS_ARCH "${CMAKE_CUDA_FLAGS_ARCH_SM70} ${CMAKE_CUDA_FLAGS_ARCH_SM75} ${CMAKE_CUDA_FLAGS_ARCH_SM80} ${CMAKE_CUDA_FLAGS_ARCH_SM90}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_ARCH}")

# ##########################################
# custatevec_example utility function
# ##########################################

function(add_custatevec_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
  list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
  target_include_directories(
    ${EXAMPLE_TARGET}
      PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${CUSTATEVEC_ROOT}/include
  )
  target_link_directories(
    ${EXAMPLE_TARGET}
      PUBLIC
        ${CUSTATEVEC_ROOT}/lib
        ${CUSTATEVEC_ROOT}/lib64
  )
  target_link_libraries(
    ${EXAMPLE_TARGET}
      PUBLIC
        custatevec
        cudart
        cublas
        cublasLt
  )
  set_target_properties(
    ${EXAMPLE_TARGET}
    PROPERTIES
      CUDA_ARCHITECTURES
        "70;75;80;90"
  )
  # Install example
  install(
      TARGETS ${EXAMPLE_TARGET}
      RUNTIME
      DESTINATION test
      PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
  )
  add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ##########################################
# custatevec_example add all targets
# ##########################################

add_custom_target(custatevec_examples)

add_custatevec_example(custatevec_examples "cuStateVec.example.initialize_sv"             initialize_sv.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.gate_application"          gate_application.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.permutation_matrix"        permutation_matrix.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.diagonal_matrix"           diagonal_matrix.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.batched_gate_application"  batched_gate_application.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.exponential_pauli"         exponential_pauli.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.expectation"               expectation.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.batched_expectation"       batched_expectation.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.expentation_pauli"         expectation_pauli.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.sampler"                   sampler.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.measure_zbasis"            measure_zbasis.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.batch_measure"             batch_measure.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.batched_collapse"          batched_collapse.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.batched_abs2sum"           batched_abs2sum.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.batched_measure"           batched_measure.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.accessor_get"              accessor_get.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.accessor_set"              accessor_set.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.test_matrix_type"          test_matrix_type.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.memory_handler"            memory_handler.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.swap_index_bits"           swap_index_bits.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.mgpu_swap_index_bits"      mgpu_swap_index_bits.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.mgpu_sampler"              mgpu_sampler.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.mgpu_batch_measure"        mgpu_batch_measure.cu)
add_custatevec_example(custatevec_examples "cuStateVec.example.subsv_migration"           subsv_migration.cu)
