/* Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cuda_runtime_api.h>

#include <complex>
#include <vector>
#include <iostream>


/** Error handling macro definitions */

#define HANDLE_CUDA_ERROR(x)                                \
{                                                           \
  const auto err = x;                                       \
  if (err != cudaSuccess)                                   \
  {                                                         \
    const char * error = cudaGetErrorString(err);           \
    printf("CUDA Error: %s in line %d\n", error, __LINE__); \
    fflush(stdout);                                         \
    std::abort();                                           \
  }                                                         \
};

#define HANDLE_CUDM_ERROR(x)                               \
{                                                          \
  const auto err = x;                                      \
  if (err != CUDENSITYMAT_STATUS_SUCCESS)                  \
  {                                                        \
    printf("cuDensityMat error in line %d\n", __LINE__);   \
    fflush(stdout);                                        \
    std::abort();                                          \
  }                                                        \
};

#ifdef MPI_ENABLED
#define HANDLE_MPI_ERROR(x)                                \
{                                                          \
  const auto err = x;                                      \
  if (err != MPI_SUCCESS)                                  \
  {                                                        \
    char error[MPI_MAX_ERROR_STRING];                      \
    int len;                                               \
    MPI_Error_string(err, error, &len);                    \
    printf("MPI Error: %s in line %d\n", error, __LINE__); \
    fflush(stdout);                                        \
    MPI_Abort(MPI_COMM_WORLD, err);                        \
  }                                                        \
};
#endif


/** Helper function for creating an array copy in GPU memory:
 *   RealComplexType = {float, double, complex<float>, complex<double>} */
template <typename RealComplexType>
void * createArrayGPU(const std::vector<RealComplexType> & cpuArray)
{
  void * gpuArray {nullptr};
  const std::size_t arraySize = cpuArray.size() * sizeof(RealComplexType);
  if (arraySize > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&gpuArray, arraySize));
    HANDLE_CUDA_ERROR(cudaMemcpy(gpuArray, static_cast<const void*>(cpuArray.data()),
                                 arraySize, cudaMemcpyHostToDevice));
  }
  return gpuArray;
}


/** Helper function for destroying a previously created array copy in GPU memory */
inline void destroyArrayGPU(void * gpuArray)
{
  HANDLE_CUDA_ERROR(cudaFree(gpuArray));
  return;
}


/** Helper function for printing a GPU array */
template <typename RealComplexType>
void printArrayGPU(void * gpuArray,
                   std::size_t arrayLen)
{
  std::vector<RealComplexType> cpuArray(arrayLen);
  const std::size_t arraySize = arrayLen * sizeof(RealComplexType);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaMemcpy(cpuArray.data(), gpuArray, arraySize, cudaMemcpyDeviceToHost));
  std::cout << "\nPrinting array " << gpuArray << "[" << arrayLen << "]:\n";
  for (std::size_t i = 0; i < arrayLen; ++i) {
    std::cout << " " << i << "   " << cpuArray[i] << std::endl;
  }
  std::cout << std::flush;
  return;
}
