/* Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: Sampler #1

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <complex>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <cutensornet.h>

#define HANDLE_CUDA_ERROR(x) \
{ const auto err = x; \
  if( err != cudaSuccess ) \
  { printf("CUDA error %s in line %d\n", cudaGetErrorString(err), __LINE__); fflush(stdout); std::abort(); } \
};

#define HANDLE_CUTN_ERROR(x) \
{ const auto err = x; \
  if( err != CUTENSORNET_STATUS_SUCCESS ) \
  { printf("cuTensorNet error %s in line %d\n", cutensornetGetErrorString(err), __LINE__); fflush(stdout); std::abort(); } \
};


int main()
{
  static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

  constexpr std::size_t fp64size = sizeof(double);

  // Sphinx: Sampler #2

  // Quantum state configuration
  const int64_t numSamples = 100;
  const int32_t numQubits = 16;
  const std::vector<int64_t> qubitDims(numQubits, 2); // qubit size
  std::cout << "Quantum circuit: " << numQubits << " qubits; " << numSamples << " samples\n";

  // Sphinx: Sampler #3

  // Initialize the cuTensorNet library
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  cutensornetHandle_t cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  std::cout << "Initialized cuTensorNet library on GPU 0\n";

  // Sphinx: Sampler #4

  // Define necessary quantum gate tensors in Host memory
  const double invsq2 = 1.0 / std::sqrt(2.0);
  //  Hadamard gate
  const std::vector<std::complex<double>> h_gateH {{invsq2, 0.0},  {invsq2, 0.0},
                                                   {invsq2, 0.0}, {-invsq2, 0.0}};
  //  CX gate
  const std::vector<std::complex<double>> h_gateCX {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
                                                    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  // Copy quantum gates to Device memory
  void *d_gateH{nullptr}, *d_gateCX{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateH, 4 * (2 * fp64size)));
  std::cout << "H gate buffer allocated on GPU: " << d_gateH << std::endl; //debug
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateCX, 16 * (2 * fp64size)));
  std::cout << "CX gate buffer allocated on GPU: " << d_gateCX << std::endl; //debug
  std::cout << "Allocated quantum gate memory on GPU\n";
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateH, h_gateH.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateCX, h_gateCX.data(), 16 * (2 * fp64size), cudaMemcpyHostToDevice));
  std::cout << "Copied quantum gates to GPU memory\n";

  // Sphinx: Sampler #5

  // Create the initial quantum state
  cutensornetState_t quantumState;
  HANDLE_CUTN_ERROR(cutensornetCreateState(cutnHandle, CUTENSORNET_STATE_PURITY_PURE, numQubits, qubitDims.data(),
                    CUDA_C_64F, &quantumState));
  std::cout << "Created the initial quantum state\n";

  // Sphinx: Sampler #6

  // Construct the quantum circuit state (apply quantum gates)
  int64_t id;
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{0}}.data(),
                    d_gateH, nullptr, 1, 0, 1, &id));
  for(int32_t i = 1; i < numQubits; ++i) {
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 2, std::vector<int32_t>{{i-1,i}}.data(),
                      d_gateCX, nullptr, 1, 0, 1, &id));
  }
  std::cout << "Applied quantum gates\n";

  // Sphinx: Sampler #7

  // Create the quantum circuit sampler
  cutensornetStateSampler_t sampler;
  HANDLE_CUTN_ERROR(cutensornetCreateSampler(cutnHandle, quantumState, numQubits, nullptr, &sampler));
  std::cout << "Created the quantum circuit sampler\n";

  // Sphinx: Sampler #8

  // Configure the quantum circuit sampler
  const int32_t numHyperSamples = 8; // desired number of hyper samples used in the tensor network contraction path finder
  HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(cutnHandle, sampler,
    CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples, sizeof(numHyperSamples)));
  const int32_t rndSeed = 13; // explicit random seed to get the same results each run
  HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(cutnHandle, sampler,
    CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC, &rndSeed, sizeof(rndSeed)));

  // Sphinx: Sampler #9

  // Query the free memory on Device
  std::size_t freeSize {0}, totalSize {0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  const std::size_t workSizeAvailable = (freeSize - (freeSize % 4096)) / 2; // use half of available memory with alignment

  // Prepare the quantum circuit sampler
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  HANDLE_CUTN_ERROR(cutensornetSamplerPrepare(cutnHandle, sampler, workSizeAvailable, workDesc, 0x0));
  std::cout << "Prepared the quantum circuit state sampler\n";
  double flops {0.0};
  HANDLE_CUTN_ERROR(cutensornetSamplerGetInfo(cutnHandle, sampler,
                    CUTENSORNET_SAMPLER_INFO_FLOPS, &flops, sizeof(flops)));
  std::cout << "Total flop count per sample = " << (flops/1e9) << " GFlop\n";

  // Sphinx: Sampler #10

  // Attach the workspace buffer
  int64_t worksize {0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
                                                      workDesc,
                                                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                      CUTENSORNET_MEMSPACE_DEVICE,
                                                      CUTENSORNET_WORKSPACE_SCRATCH,
                                                      &worksize));
  assert(worksize > 0);

  void *d_scratch {nullptr};
  if(worksize <= workSizeAvailable) {
    HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, worksize));
    std::cout << "Allocated " << worksize << " bytes of scratch memory on GPU: "
              << "[" << d_scratch << ":" << (void*)(((char*)(d_scratch))  + worksize) << ")\n";

    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  }else{
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }

  int64_t reqCacheSize {0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
                                                      workDesc,
                                                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                      CUTENSORNET_MEMSPACE_DEVICE,
                                                      CUTENSORNET_WORKSPACE_CACHE,
                                                      &reqCacheSize));

  //query the free size again
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  // grab the minimum of [required size, or 90% of the free memory (to avoid oversubscribing)]
  const std::size_t cacheSizeAvailable = std::min(static_cast<size_t>(reqCacheSize), size_t(freeSize * 0.9) - (size_t(freeSize * 0.9) % 4096));
  void *d_cache {nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_cache, cacheSizeAvailable));
  std::cout << "Allocated " << cacheSizeAvailable << " bytes of cache memory on GPU: "
            << "[" << d_cache << ":" << (void*)(((char*)(d_cache))  + cacheSizeAvailable) << ")\n";
  HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle,
                                                  workDesc,
                                                  CUTENSORNET_MEMSPACE_DEVICE,
                                                  CUTENSORNET_WORKSPACE_CACHE,
                                                  d_cache,
                                                  cacheSizeAvailable));
  std::cout << "Set the workspace buffer\n";

  // Sphinx: Sampler #11

  // Sample the quantum circuit state
  std::vector<int64_t> samples(numQubits * numSamples); // samples[SampleId][QubitId] reside in Host memory
  HANDLE_CUTN_ERROR(cutensornetSamplerSample(cutnHandle, sampler, numSamples, workDesc, samples.data(), 0));
  std::cout << "Performed quantum circuit state sampling\n";
  std::cout << "Bit-string samples:\n";
  for(int64_t i = 0; i < numSamples; ++i) {
    for(int64_t j = 0; j < numQubits; ++j) std::cout << " " << samples[i * numQubits + j];
    std::cout << std::endl;
  }

  // Sphinx: Sampler #12

  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  std::cout << "Destroyed the workspace descriptor\n";

  // Destroy the quantum circuit sampler
  HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler));
  std::cout << "Destroyed the quantum circuit state sampler\n";

  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));
  std::cout << "Destroyed the quantum circuit state\n";

  HANDLE_CUDA_ERROR(cudaFree(d_scratch));
  HANDLE_CUDA_ERROR(cudaFree(d_cache));
  HANDLE_CUDA_ERROR(cudaFree(d_gateCX));
  HANDLE_CUDA_ERROR(cudaFree(d_gateH));
  std::cout << "Freed memory on GPU\n";

  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
  std::cout << "Finalized the cuTensorNet library\n";

  return 0;
}
