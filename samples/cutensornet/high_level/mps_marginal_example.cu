/* Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: MPS Marginal #1

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <complex>
#include <vector>
#include <bitset>
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

  // Sphinx: MPS Marginal #2

  // Quantum state configuration
  constexpr int32_t numQubits = 16;
  const std::vector<int64_t> qubitDims(numQubits,2); // qubit dimensions
  constexpr int32_t numMarginalModes = 2; // rank of the marginal (reduced density matrix)
  const std::vector<int32_t> marginalModes({0,1}); // open qubits (must be in acsending order)
  std::cout << "Quantum circuit: " << numQubits << " qubits\n";

  // Sphinx: MPS Marginal #3

  // Initialize the cuTensorNet library
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  cutensornetHandle_t cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  std::cout << "Initialized cuTensorNet library on GPU 0\n";

  // Sphinx: MPS Marginal #4

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
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateCX, 16 * (2 * fp64size)));
  std::cout << "Allocated quantum gate memory on GPU\n";
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateH, h_gateH.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateCX, h_gateCX.data(), 16 * (2 * fp64size), cudaMemcpyHostToDevice));
  std::cout << "Copied quantum gates to GPU memory\n";

  // Sphinx: MPS Marginal #5

  // Determine the MPS representation and allocate buffers for the MPS tensors
  const int64_t maxExtent = 2; // GHZ state can be exactly represented with max bond dimension of 2
  std::vector<std::vector<int64_t>> extents;
  std::vector<int64_t*> extentsPtr(numQubits); 
  std::vector<void*> d_mpsTensors(numQubits, nullptr);
  for (int32_t i = 0; i < numQubits; i++) {
    if (i == 0) { // left boundary MPS tensor
      extents.push_back({2, maxExtent});
      HANDLE_CUDA_ERROR(cudaMalloc(&d_mpsTensors[i], 2 * maxExtent * 2 * fp64size));
    } 
    else if (i == numQubits-1) { // right boundary MPS tensor
      extents.push_back({maxExtent, 2});
      HANDLE_CUDA_ERROR(cudaMalloc(&d_mpsTensors[i], 2 * maxExtent * 2 * fp64size));
    }
    else { // middle MPS tensors
      extents.push_back({maxExtent, 2, maxExtent});
      HANDLE_CUDA_ERROR(cudaMalloc(&d_mpsTensors[i], 2 * maxExtent * maxExtent * 2 * fp64size));
    }
    extentsPtr[i] = extents[i].data();
  }

  // Sphinx: MPS Marginal #6

  // Allocate the specified quantum circuit reduced density matrix (marginal) in Device memory
  void *d_rdm{nullptr};
  std::size_t rdmDim = 1;
  for(const auto & mode: marginalModes) rdmDim *= qubitDims[mode];
  const std::size_t rdmSize = rdmDim * rdmDim;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_rdm, rdmSize * (2 * fp64size)));

  // Sphinx: MPS Marginal #7

  // Query the free memory on Device
  std::size_t freeSize{0}, totalSize{0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  const std::size_t scratchSize = (freeSize - (freeSize % 4096)) / 2; // use half of available memory with alignment
  void *d_scratch{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  std::cout << "Allocated " << scratchSize << " bytes of scratch memory on GPU\n";

  // Sphinx: MPS Marginal #8

  // Create the initial quantum state
  cutensornetState_t quantumState;
  HANDLE_CUTN_ERROR(cutensornetCreateState(cutnHandle, CUTENSORNET_STATE_PURITY_PURE, numQubits, qubitDims.data(),
                    CUDA_C_64F, &quantumState));
  std::cout << "Created the initial quantum state\n";

  // Sphinx: MPS Marginal #9

  // Construct the final quantum circuit state (apply quantum gates) for the GHZ circuit
  int64_t id;
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{0}}.data(),
                    d_gateH, nullptr, 1, 0, 1, &id));
  for(int32_t i = 1; i < numQubits; ++i) {
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 2, std::vector<int32_t>{{i-1,i}}.data(),
                      d_gateCX, nullptr, 1, 0, 1, &id));
  }
  std::cout << "Applied quantum gates\n";

  // Sphinx: MPS Marginal #10

  // Specify the final target MPS representation (use default fortran strides)
  HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(cutnHandle, quantumState, 
                    CUTENSORNET_BOUNDARY_CONDITION_OPEN, extentsPtr.data(), /*strides=*/nullptr));
  std::cout << "Requested the final MPS factorization of the quantum circuit state\n";

  // Sphinx: MPS Marginal #11

  // Optional, set up the SVD method for MPS truncation.
  cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ; 
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(cutnHandle, quantumState, 
                    CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO, &algo, sizeof(algo)));
  std::cout << "Configured the MPS factorization computation\n";

  // Sphinx: MPS Marginal #12

  // Prepare the MPS computation and attach workspace
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  std::cout << "Created the workspace descriptor\n";
  HANDLE_CUTN_ERROR(cutensornetStatePrepare(cutnHandle, quantumState, scratchSize, workDesc, 0x0));
  std::cout << "Prepared the computation of the quantum circuit state\n";
  double flops {0.0};
  HANDLE_CUTN_ERROR(cutensornetStateGetInfo(cutnHandle, quantumState,
                    CUTENSORNET_STATE_INFO_FLOPS, &flops, sizeof(flops)));
  if(flops > 0.0) {
    std::cout << "Total flop count = " << (flops/1e9) << " GFlop\n";
  }else if(flops < 0.0) {
    std::cout << "ERROR: Negative Flop count!\n";
    std::abort();
  }

  int64_t worksize {0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
                                                      workDesc,
                                                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                      CUTENSORNET_MEMSPACE_DEVICE,
                                                      CUTENSORNET_WORKSPACE_SCRATCH,
                                                      &worksize));
  std::cout << "Scratch GPU workspace size (bytes) for MPS computation = " << worksize << std::endl;
  if(worksize <= scratchSize) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  }else{
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  std::cout << "Set the workspace buffer for the MPS factorization computation\n";

  // Sphinx: MPS Marginal #13

  // Execute MPS computation
  HANDLE_CUTN_ERROR(cutensornetStateCompute(cutnHandle, quantumState, 
                    workDesc, extentsPtr.data(), /*strides=*/nullptr, d_mpsTensors.data(), 0));
  std::cout << "Computed the MPS factorization\n";

  // Sphinx: MPS Marginal #14

  // Specify the desired reduced density matrix (marginal)
  cutensornetStateMarginal_t marginal;
  HANDLE_CUTN_ERROR(cutensornetCreateMarginal(cutnHandle, quantumState, numMarginalModes, marginalModes.data(),
                    0, nullptr, std::vector<int64_t>{{1,2,4,8}}.data(), &marginal)); // using explicit strides
  std::cout << "Created the specified quantum circuit reduced densitry matrix (marginal)\n";

  // Sphinx: MPS Marginal #15

  // Configure the computation of the specified quantum circuit reduced density matrix (marginal)
  const int32_t numHyperSamples = 8; // desired number of hyper samples used in the tensor network contraction path finder
  HANDLE_CUTN_ERROR(cutensornetMarginalConfigure(cutnHandle, marginal,
                    CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples, sizeof(numHyperSamples)));
  std::cout << "Configured the specified quantum circuit reduced density matrix (marginal) computation\n";

  // Sphinx: MPS Marginal #16

  // Prepare the specified quantum circuit reduced densitry matrix (marginal)
  HANDLE_CUTN_ERROR(cutensornetMarginalPrepare(cutnHandle, marginal, scratchSize, workDesc, 0x0));
  std::cout << "Prepared the specified quantum circuit reduced density matrix (marginal)\n";
  flops = 0.0;
  HANDLE_CUTN_ERROR(cutensornetMarginalGetInfo(cutnHandle, marginal,
                    CUTENSORNET_MARGINAL_INFO_FLOPS, &flops, sizeof(flops)));
  std::cout << "Total flop count = " << (flops/1e9) << " GFlop\n";
  if(flops <= 0.0) {
    std::cout << "ERROR: Invalid Flop count!\n";
    std::abort();
  }

  // Sphinx: MPS Marginal #17

  // Attach the workspace buffer
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
                                                      workDesc,
                                                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                      CUTENSORNET_MEMSPACE_DEVICE,
                                                      CUTENSORNET_WORKSPACE_SCRATCH,
                                                      &worksize));
  std::cout << "Required scratch GPU workspace size (bytes) for marginal computation = " << worksize << std::endl;
  if(worksize <= scratchSize) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  }else{
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  std::cout << "Set the workspace buffer\n";
  
  // Sphinx: MPS Marginal #18

  // Compute the specified quantum circuit reduced densitry matrix (marginal)
  HANDLE_CUTN_ERROR(cutensornetMarginalCompute(cutnHandle, marginal, nullptr, workDesc, d_rdm, 0));
  std::cout << "Computed the specified quantum circuit reduced density matrix (marginal)\n";
  std::vector<std::complex<double>> h_rdm(rdmSize);
  HANDLE_CUDA_ERROR(cudaMemcpy(h_rdm.data(), d_rdm, rdmSize * (2 * fp64size), cudaMemcpyDeviceToHost));
  std::cout << "Reduced density matrix for " << numMarginalModes << " qubits:\n";
  for(std::size_t i = 0; i < rdmDim; ++i) {
    for(std::size_t j = 0; j < rdmDim; ++j) {
      std::cout << " " << h_rdm[i + j * rdmDim];
    }
    std::cout << std::endl;
  }

  // Sphinx: MPS Marginal #19

  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  std::cout << "Destroyed the workspace descriptor\n";

  // Destroy the quantum circuit reduced density matrix
  HANDLE_CUTN_ERROR(cutensornetDestroyMarginal(marginal));
  std::cout << "Destroyed the quantum circuit state reduced density matrix (marginal)\n";

  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));
  std::cout << "Destroyed the quantum circuit state\n";

  for (int32_t i = 0; i < numQubits; i++) {
    HANDLE_CUDA_ERROR(cudaFree(d_mpsTensors[i]));
  }
  HANDLE_CUDA_ERROR(cudaFree(d_scratch));
  HANDLE_CUDA_ERROR(cudaFree(d_rdm));
  HANDLE_CUDA_ERROR(cudaFree(d_gateCX));
  HANDLE_CUDA_ERROR(cudaFree(d_gateH));
  std::cout << "Freed memory on GPU\n";

  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
  std::cout << "Finalized the cuTensorNet library\n";

  return 0;
}
