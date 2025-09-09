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


/** Helper function for creating an array in GPU memory:
 *   RealComplexType = {float, double, complex<float>, complex<double>} */
template <typename RealComplexType>
void * createArrayGPU(std::size_t arrayLen)
{
  void * gpuArray {nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&gpuArray, arrayLen * sizeof(RealComplexType)));
  return gpuArray;
}


/** Helper function for initializing an array in GPU memory from a CPU array:
 *   RealComplexType = {float, double, complex<float>, complex<double>} */
template <typename RealComplexType>
void initializeArrayGPU(const std::vector<RealComplexType> & cpuArray,
                        void * gpuArray)
{
  const std::size_t arrayLen = cpuArray.size();
  if (arrayLen > 0) {
    HANDLE_CUDA_ERROR(cudaMemcpy(gpuArray, static_cast<const void*>(cpuArray.data()),
                                 arrayLen * sizeof(RealComplexType), cudaMemcpyHostToDevice));
  }
  return;
}


/** Helper function for creating and initializing an array in GPU memory:
 *   RealComplexType = {float, double, complex<float>, complex<double>} */
template <typename RealComplexType>
void * createInitializeArrayGPU(const std::vector<RealComplexType> & cpuArray)
{
  void * gpuArray {nullptr};
  const std::size_t arrayLen = cpuArray.size();
  if (arrayLen > 0) {
    gpuArray = createArrayGPU<RealComplexType>(arrayLen);
    if (gpuArray != nullptr) {
      initializeArrayGPU(cpuArray, gpuArray);
    }
  }
  return gpuArray;
}


/** Helper function for destroying a previously created array copy in GPU memory */
inline void destroyArrayGPU(void * gpuArray)
{
  HANDLE_CUDA_ERROR(cudaFree(gpuArray));
  return;
}


/** Helper function for printing a GPU array after GPU synchronization
 *   RealComplexType = {float, double, complex<float>, complex<double>} */
template <typename RealComplexType>
void printArrayGPU(void * gpuArray,
                   std::size_t arrayLen)
{
  std::vector<RealComplexType> cpuArray(arrayLen);
  const std::size_t arraySize = arrayLen * sizeof(RealComplexType);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaMemcpy(static_cast<void*>(cpuArray.data()), gpuArray, arraySize, cudaMemcpyDeviceToHost));
  std::cout << "Printing array " << gpuArray << "[" << arrayLen << "]:\n";
  for (std::size_t i = 0; i < arrayLen; ++i) {
    std::cout << " " << i << "   " << cpuArray[i] << std::endl;
  }
  std::cout << std::flush;
  return;
}

/** Helper function for printing a CPU array
 *   RealComplexType = {float, double, complex<float>, complex<double>} */
 template <typename RealComplexType>
 void printArrayCPU(void * cpuArray,
                    std::size_t arrayLen)
 {
   RealComplexType * realComplexCpuArray = static_cast<RealComplexType*>(cpuArray);
   for (std::size_t i = 0; i < arrayLen; ++i) {
     std::cout << " " << i << "   " << realComplexCpuArray[i] << std::endl;
   }
   std::cout << std::flush;
   return;
 }
