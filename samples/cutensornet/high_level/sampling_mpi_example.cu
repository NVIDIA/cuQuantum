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

#include <mpi.h>

#define HANDLE_CUDA_ERROR(x) \
{ const auto err = x; \
  if( err != cudaSuccess ) \
  { printf("CUDA error %s in line %d\n", cudaGetErrorString(err), __LINE__); \
    fflush(stdout); \
    std::abort(); \
  } \
};

#define HANDLE_CUTN_ERROR(x) \
{ const auto err = x; \
  if( err != CUTENSORNET_STATUS_SUCCESS ) \
  { printf("cuTensorNet error %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
    fflush(stdout); \
    std::abort(); \
  } \
};

#define HANDLE_MPI_ERROR(x) \
{ const auto err = x; \
  if( err != MPI_SUCCESS ) \
  { char error[MPI_MAX_ERROR_STRING]; \
    int len; \
    MPI_Error_string(err, error, &len); \
    printf("MPI Error: %s in line %d\n", error, __LINE__); \
    fflush(stdout); \
    MPI_Abort(MPI_COMM_WORLD, err); \
  } \
};


int main(int argc, char **argv)
{
  static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

  constexpr std::size_t fp64size = sizeof(double);

  // Sphinx: Sampler #2

  // Initialize MPI library
  HANDLE_MPI_ERROR(MPI_Init(&argc, &argv));
  int rank {-1};
  HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int numProcs {0};
  HANDLE_MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &numProcs));

  bool verbose = (rank == 0) ? true : false;
  if (verbose)
  {
    std::cout << "*** Printing is done only from the root MPI process to prevent jumbled messages ***\n";
    std::cout << "The number of MPI processes is " << numProcs << std::endl;
  }

  // Sphinx: Sampler #3

  // Quantum state configuration
  const int64_t numSamples = 100;
  const int32_t numQubits = 16;
  const std::vector<int64_t> qubitDims(numQubits, 2); // qubit size
  if (verbose)
   std::cout << "Quantum circuit: " << numQubits << " qubits; " << numSamples << " samples\n";

  // Sphinx: Sampler #4

  // Initialize the cuTensorNet library
  int numDevices {0};
  HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
  const int deviceId = rank % numDevices; // we assume that the processes are mapped to nodes in contiguous chunks
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  
  cutensornetHandle_t cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  if (verbose)
   std::cout << "Initialized cuTensorNet library on GPU 0\n";

  // Sphinx: Sampler #5

  // Activate distributed (parallel) execution
  // HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(handle, NULL, 0)); // reset back to serial execution
  MPI_Comm cutnComm;
  HANDLE_MPI_ERROR(MPI_Comm_dup(MPI_COMM_WORLD, &cutnComm)); // duplicate MPI communicator to dedicate it to cuTensorNet
  HANDLE_CUTN_ERROR(cutensornetDistributedResetConfiguration(cutnHandle, &cutnComm, sizeof(cutnComm)));
  if(verbose)
   printf("Reset cuTensorNet distributed MPI configuration\n");

  // Sphinx: Sampler #6

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
  if (verbose)
   std::cout << "H gate buffer allocated on GPU: " << d_gateH << std::endl; //debug
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateCX, 16 * (2 * fp64size)));
  if (verbose)
   std::cout << "CX gate buffer allocated on GPU: " << d_gateCX << std::endl; //debug
  if (verbose)
   std::cout << "Allocated quantum gate memory on GPU\n";
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateH, h_gateH.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateCX, h_gateCX.data(), 16 * (2 * fp64size), cudaMemcpyHostToDevice));
  if (verbose)
   std::cout << "Copied quantum gates to GPU memory\n";

  // Sphinx: Sampler #7

  // Create the initial quantum state
  cutensornetState_t quantumState;
  HANDLE_CUTN_ERROR(cutensornetCreateState(cutnHandle, CUTENSORNET_STATE_PURITY_PURE, numQubits, qubitDims.data(),
                    CUDA_C_64F, &quantumState));
  if (verbose)
   std::cout << "Created the initial quantum state\n";

  // Sphinx: Sampler #8

  // Construct the quantum circuit state (apply quantum gates)
  int64_t id;
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{0}}.data(),
                    d_gateH, nullptr, 1, 0, 1, &id));
  for(int32_t i = 1; i < numQubits; ++i) {
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 2, std::vector<int32_t>{{i-1,i}}.data(),
                      d_gateCX, nullptr, 1, 0, 1, &id));
  }
  if (verbose)
   std::cout << "Applied quantum gates\n";

  // Sphinx: Sampler #9

  // Create the quantum circuit sampler
  cutensornetStateSampler_t sampler;
  HANDLE_CUTN_ERROR(cutensornetCreateSampler(cutnHandle, quantumState, numQubits, nullptr, &sampler));
  if (verbose)
   std::cout << "Created the quantum circuit sampler\n";

  // Sphinx: Sampler #10

  // Configure the quantum circuit sampler
  const int32_t numHyperSamples = 8; // desired number of hyper samples used in the tensor network contraction path finder
  HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(cutnHandle, sampler,
                    CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples, sizeof(numHyperSamples)));

  // Sphinx: Sampler #11
  // Query the free memory on Device
  std::size_t freeSize {0}, totalSize {0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  std::size_t workSizeAvailable = size_t(double(freeSize) * 0.9) / 8; // assume max of 8 GPUs per node
  workSizeAvailable -= workSizeAvailable % 4096;

  // Prepare the quantum circuit sampler
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  HANDLE_CUTN_ERROR(cutensornetSamplerPrepare(cutnHandle, sampler, workSizeAvailable, workDesc, 0x0));
  if (verbose)
   std::cout << "Prepared the quantum circuit state sampler\n";
  double flops {0.0};
  HANDLE_CUTN_ERROR(cutensornetSamplerGetInfo(cutnHandle, sampler,
                    CUTENSORNET_SAMPLER_INFO_FLOPS, &flops, sizeof(flops)));
  if (verbose)
   std::cout << "Total flop count per sample = " << (flops/1e9) << " GFlop\n";

  // Sphinx: Sampler #12

  // Attach the workspace buffer
  int64_t worksize {0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
                                                      workDesc,
                                                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                      CUTENSORNET_MEMSPACE_DEVICE,
                                                      CUTENSORNET_WORKSPACE_SCRATCH,
                                                      &worksize));
  assert(worksize > 0);

  int64_t cacheWorksize {0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
                                                      workDesc,
                                                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                      CUTENSORNET_MEMSPACE_DEVICE,
                                                      CUTENSORNET_WORKSPACE_CACHE,
                                                      &cacheWorksize));
  void *d_scratch {nullptr};
  if(worksize <= workSizeAvailable) {
    HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, worksize));
    if (verbose)
     std::cout << "Allocated " << worksize << " bytes of scratch memory on GPU: "
               << "[" << d_scratch << ":" << (void*)(((char*)(d_scratch))  + worksize) << ")\n";
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  }else{
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  const auto cacheSizeAvailable = std::min(workSizeAvailable - static_cast<size_t>(worksize), static_cast<size_t>(cacheWorksize));
  void *d_cache {nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_cache, cacheSizeAvailable));
  if (verbose)
    std::cout << "Allocated " << cacheSizeAvailable << " bytes of cache memory on GPU: "
              << "[" << d_cache << ":" << (void*)(((char*)(d_cache))  + cacheSizeAvailable) << ")\n";
  HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle,
                                                  workDesc,
                                                  CUTENSORNET_MEMSPACE_DEVICE,
                                                  CUTENSORNET_WORKSPACE_CACHE,
                                                  d_cache,
                                                  cacheSizeAvailable));
  if (verbose)
   std::cout << "Set the workspace buffer\n";

  // Sphinx: Sampler #13

  // Sample the quantum circuit state
  std::vector<int64_t> samples(numQubits * numSamples); // samples[SampleId][QubitId] reside in Host memory
  HANDLE_CUTN_ERROR(cutensornetSamplerSample(cutnHandle, sampler, numSamples, workDesc, samples.data(), 0));
  if (verbose) {
    std::cout << "Performed quantum circuit state sampling\n";
    std::cout << "Bit-string samples:\n";
    for(int64_t i = 0; i < numSamples; ++i) {
      for(int64_t j = 0; j < numQubits; ++j) std::cout << " " << samples[i * numQubits + j];
      std::cout << std::endl;
    }
  }

  // Sphinx: Sampler #14

  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  if (verbose)
   std::cout << "Destroyed the workspace descriptor\n";

  // Destroy the quantum circuit sampler
  HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler));
  if (verbose)
   std::cout << "Destroyed the quantum circuit state sampler\n";

  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));
  if (verbose)
   std::cout << "Destroyed the quantum circuit state\n";

  HANDLE_CUDA_ERROR(cudaFree(d_scratch));
  HANDLE_CUDA_ERROR(cudaFree(d_cache));
  HANDLE_CUDA_ERROR(cudaFree(d_gateCX));
  HANDLE_CUDA_ERROR(cudaFree(d_gateH));
  if (verbose)
   std::cout << "Freed memory on GPU\n";

  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
  if (verbose)
   std::cout << "Finalized the cuTensorNet library\n";

  // Finalize the MPI library
  HANDLE_MPI_ERROR(MPI_Finalize());

  return 0;
}
