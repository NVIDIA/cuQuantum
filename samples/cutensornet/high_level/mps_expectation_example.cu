/* Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: MPS Expectation #1

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

  // Sphinx: MPS Expectation #2

  // Quantum state configuration
  constexpr int32_t numQubits = 16; // number of qubits
  const std::vector<int64_t> qubitDims(numQubits,2); // qubit dimensions
  std::cout << "Quantum circuit: " << numQubits << " qubits\n";

  // Sphinx: MPS Expectation #3

  // Initialize the cuTensorNet library
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  cutensornetHandle_t cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  std::cout << "Initialized cuTensorNet library on GPU 0\n";

  // Sphinx: MPS Expectation #4

  // Define necessary quantum gate tensors in Host memory
  const double invsq2 = 1.0 / std::sqrt(2.0);
  //  Hadamard gate
  const std::vector<std::complex<double>> h_gateH {{invsq2, 0.0},  {invsq2, 0.0},
                                                   {invsq2, 0.0}, {-invsq2, 0.0}};
  //  Pauli X gate
  const std::vector<std::complex<double>> h_gateX {{0.0, 0.0}, {1.0, 0.0},
                                                   {1.0, 0.0}, {0.0, 0.0}};
  //  Pauli Y gate
  const std::vector<std::complex<double>> h_gateY {{0.0, 0.0}, {0.0, -1.0},
                                                   {0.0, 1.0}, {0.0, 0.0}};
  //  Pauli Z gate
  const std::vector<std::complex<double>> h_gateZ {{1.0, 0.0}, {0.0, 0.0},
                                                   {0.0, 0.0}, {-1.0, 0.0}};
  //  CX gate
  const std::vector<std::complex<double>> h_gateCX {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
                                                    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  // Copy quantum gates to Device memory
  void *d_gateH{nullptr}, *d_gateX{nullptr}, *d_gateY{nullptr}, *d_gateZ{nullptr}, *d_gateCX{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateH, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateX, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateY, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateZ, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateCX, 16 * (2 * fp64size)));
  std::cout << "Allocated quantum gate memory on GPU\n";
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateH, h_gateH.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateX, h_gateX.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateY, h_gateY.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateZ, h_gateZ.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateCX, h_gateCX.data(), 16 * (2 * fp64size), cudaMemcpyHostToDevice));
  std::cout << "Copied quantum gates to GPU memory\n";

  // Sphinx: MPS Expectation #5

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

  // Sphinx: MPS Expectation #6

  // Query the free memory on Device
  std::size_t freeSize{0}, totalSize{0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  const std::size_t scratchSize = (freeSize - (freeSize % 4096)) / 2; // use half of available memory with alignment
  void *d_scratch{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  std::cout << "Allocated " << scratchSize << " bytes of scratch memory on GPU\n";

  // Sphinx: MPS Expectation #7

  // Create the initial quantum state
  cutensornetState_t quantumState;
  HANDLE_CUTN_ERROR(cutensornetCreateState(cutnHandle, CUTENSORNET_STATE_PURITY_PURE, numQubits, qubitDims.data(),
                    CUDA_C_64F, &quantumState));
  std::cout << "Created the initial quantum state\n";

  // Sphinx: MPS Expectation #8

  // Construct the final quantum circuit state (apply quantum gates) for the GHZ circuit
  int64_t id;
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{0}}.data(),
                    d_gateH, nullptr, 1, 0, 1, &id));
  for(int32_t i = 1; i < numQubits; ++i) {
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 2, std::vector<int32_t>{{i-1,i}}.data(),
                      d_gateCX, nullptr, 1, 0, 1, &id));
  }
  std::cout << "Applied quantum gates\n";

  // Sphinx: MPS Expectation #9

  // Specify the final target MPS representation (use default fortran strides)
  HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(cutnHandle, quantumState, 
                    CUTENSORNET_BOUNDARY_CONDITION_OPEN, extentsPtr.data(), /*strides=*/nullptr ));

  // Sphinx: MPS Expectation #10

  // Optional, set up the SVD method for truncation.
  cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ; 
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(cutnHandle, quantumState, 
                    CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO, &algo, sizeof(algo)));
  std::cout << "Configured the MPS computation\n";

  // Sphinx: MPS Expectation #11

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
  std::cout << "Set the workspace buffer for MPS computation\n";

  // Sphinx: MPS Expectation #12

  // Execute MPS computation
  HANDLE_CUTN_ERROR(cutensornetStateCompute(cutnHandle, quantumState, 
                    workDesc, extentsPtr.data(), /*strides=*/nullptr, d_mpsTensors.data(), 0));

  // Sphinx: MPS Expectation #13

  // Create an empty tensor network operator
  cutensornetNetworkOperator_t hamiltonian;
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkOperator(cutnHandle, numQubits, qubitDims.data(), CUDA_C_64F, &hamiltonian));
  // Append component (0.5 * Z1 * Z2) to the tensor network operator
  {
    const int32_t numModes[] = {1, 1}; // Z1 acts on 1 mode, Z2 acts on 1 mode
    const int32_t modesZ1[] = {1}; // state modes Z1 acts on
    const int32_t modesZ2[] = {2}; // state modes Z2 acts on
    const int32_t * stateModes[] = {modesZ1, modesZ2}; // state modes (Z1 * Z2) acts on
    const void * gateData[] = {d_gateZ, d_gateZ}; // GPU pointers to gate data
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(cutnHandle, hamiltonian, cuDoubleComplex{0.5,0.0},
                      2, numModes, stateModes, NULL, gateData, &id));
  }
  // Append component (0.25 * Y3) to the tensor network operator
  {
    const int32_t numModes[] = {1}; // Y3 acts on 1 mode
    const int32_t modesY3[] = {3}; // state modes Y3 acts on
    const int32_t * stateModes[] = {modesY3}; // state modes (Y3) acts on
    const void * gateData[] = {d_gateY}; // GPU pointers to gate data
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(cutnHandle, hamiltonian, cuDoubleComplex{0.25,0.0},
                      1, numModes, stateModes, NULL, gateData, &id));
  }
  // Append component (0.13 * Y0 X2 Z3) to the tensor network operator
  {
    const int32_t numModes[] = {1, 1, 1}; // Y0 acts on 1 mode, X2 acts on 1 mode, Z3 acts on 1 mode
    const int32_t modesY0[] = {0}; // state modes Y0 acts on
    const int32_t modesX2[] = {2}; // state modes X2 acts on
    const int32_t modesZ3[] = {3}; // state modes Z3 acts on
    const int32_t * stateModes[] = {modesY0, modesX2, modesZ3}; // state modes (Y0 * X2 * Z3) acts on
    const void * gateData[] = {d_gateY, d_gateX, d_gateZ}; // GPU pointers to gate data
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(cutnHandle, hamiltonian, cuDoubleComplex{0.13,0.0},
                      3, numModes, stateModes, NULL, gateData, &id));
  }
  std::cout << "Constructed a tensor network operator: (0.5 * Z1 * Z2) + (0.25 * Y3) + (0.13 * Y0 * X2 * Z3)" << std::endl;

  // Sphinx: MPS Expectation #14

  // Specify the quantum circuit expectation value
  cutensornetStateExpectation_t expectation;
  HANDLE_CUTN_ERROR(cutensornetCreateExpectation(cutnHandle, quantumState, hamiltonian, &expectation));
  std::cout << "Created the specified quantum circuit expectation value\n";

  // Sphinx: MPS Expectation #15

  // Configure the computation of the specified quantum circuit expectation value
  const int32_t numHyperSamples = 8; // desired number of hyper samples used in the tensor network contraction path finder
  HANDLE_CUTN_ERROR(cutensornetExpectationConfigure(cutnHandle, expectation,
                    CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples, sizeof(numHyperSamples)));

  // Sphinx: MPS Expectation #16

  // Prepare the specified quantum circuit expectation value for computation
  HANDLE_CUTN_ERROR(cutensornetExpectationPrepare(cutnHandle, expectation, scratchSize, workDesc, 0x0));
  std::cout << "Prepared the specified quantum circuit expectation value\n";
  flops = 0.0;
  HANDLE_CUTN_ERROR(cutensornetExpectationGetInfo(cutnHandle, expectation,
                    CUTENSORNET_EXPECTATION_INFO_FLOPS, &flops, sizeof(flops)));
  std::cout << "Total flop count = " << (flops/1e9) << " GFlop\n";
  if(flops <= 0.0) {
    std::cout << "ERROR: Invalid Flop count!\n";
    std::abort();
  }

  // Sphinx: MPS Expectation #17

  // Attach the workspace buffer
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
                                                      workDesc,
                                                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                      CUTENSORNET_MEMSPACE_DEVICE,
                                                      CUTENSORNET_WORKSPACE_SCRATCH,
                                                      &worksize));
  std::cout << "Required scratch GPU workspace size (bytes) = " << worksize << std::endl;
  if(worksize <= scratchSize) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  }else{
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  std::cout << "Set the workspace buffer\n";

  // Sphinx: MPS Expectation #18

  // Compute the specified quantum circuit expectation value
  std::complex<double> expectVal{0.0,0.0}, stateNorm2{0.0,0.0};
  HANDLE_CUTN_ERROR(cutensornetExpectationCompute(cutnHandle, expectation, workDesc,
                    static_cast<void*>(&expectVal), static_cast<void*>(&stateNorm2), 0x0));
  std::cout << "Computed the specified quantum circuit expectation value\n";
  expectVal /= stateNorm2;
  std::cout << "Expectation value = (" << expectVal.real() << ", " << expectVal.imag() << ")\n";
  std::cout << "Squared 2-norm of the state = (" << stateNorm2.real() << ", " << stateNorm2.imag() << ")\n";

  // Sphinx: MPS Expectation #19

  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  std::cout << "Destroyed the workspace descriptor\n";

  // Destroy the quantum circuit expectation value
  HANDLE_CUTN_ERROR(cutensornetDestroyExpectation(expectation));
  std::cout << "Destroyed the quantum circuit state expectation value\n";

  // Destroy the tensor network operator
  HANDLE_CUTN_ERROR(cutensornetDestroyNetworkOperator(hamiltonian));
  std::cout << "Destroyed the tensor network operator\n";

  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));
  std::cout << "Destroyed the quantum circuit state\n";

  for (int32_t i = 0; i < numQubits; i++) {
    HANDLE_CUDA_ERROR(cudaFree(d_mpsTensors[i]));
  }
  HANDLE_CUDA_ERROR(cudaFree(d_scratch));
  HANDLE_CUDA_ERROR(cudaFree(d_gateCX));
  HANDLE_CUDA_ERROR(cudaFree(d_gateZ));
  HANDLE_CUDA_ERROR(cudaFree(d_gateY));
  HANDLE_CUDA_ERROR(cudaFree(d_gateX));
  HANDLE_CUDA_ERROR(cudaFree(d_gateH));
  std::cout << "Freed memory on GPU\n";

  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
  std::cout << "Finalized the cuTensorNet library\n";

  return 0;
}
