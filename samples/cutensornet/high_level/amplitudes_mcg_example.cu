/* Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: Amplitudes #1

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

  // Sphinx: Amplitudes #2

  // Quantum state configuration
  constexpr int32_t numQubits = 6; // number of qubits
  const std::vector<int64_t> qubitDims(numQubits,2); // qubit dimensions
  const std::vector<int32_t> fixedModes({0,1}); // fixed modes in the output amplitude tensor (must be in acsending order)
  const std::vector<int64_t> fixedValues({1,1}); // values of the fixed modes in the output amplitude tensor
  const int32_t numFixedModes = fixedModes.size(); // number of fixed modes in the output amplitude tensor
  std::cout << "Quantum circuit: " << numQubits << " qubits\n";

  // Sphinx: Amplitudes #3

  // Initialize the cuTensorNet library
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  cutensornetHandle_t cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  std::cout << "Initialized cuTensorNet library on GPU 0\n";

  // Sphinx: Amplitudes #4

  // Define necessary quantum gate tensors in Host memory
  const double invsq2 = 1.0 / std::sqrt(2.0);
  //  Hadamard gate
  const std::vector<std::complex<double>> h_gateH {{invsq2, 0.0},  {invsq2, 0.0},
                                                   {invsq2, 0.0}, {-invsq2, 0.0}};
  //  X gate
  const std::vector<std::complex<double>> h_gateX {{0.0, 0.0}, {1.0, 0.0},
                                                    {1.0, 0.0}, {0.0, 0.0}};

  // Copy quantum gates to Device memory
  void *d_gateH{nullptr}, *d_gateX{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateH, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateX, 4 * (2 * fp64size)));
  std::cout << "Allocated quantum gate memory on GPU\n";
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateH, h_gateH.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateX, h_gateX.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  std::cout << "Copied quantum gates to GPU memory\n";

  // Sphinx: Amplitudes #5

  // Allocate Device memory for the specified slice of the quantum circuit amplitudes tensor
  void *d_amp{nullptr};
  std::size_t ampSize = 1;
  for(const auto & qubitDim: qubitDims) ampSize *= qubitDim; // all state modes (full size)
  for(const auto & fixedMode: fixedModes) ampSize /= qubitDims[fixedMode]; // fixed state modes reduce the slice size
  HANDLE_CUDA_ERROR(cudaMalloc(&d_amp, ampSize * (2 * fp64size)));
  std::cout << "Allocated memory for the specified slice of the quantum circuit amplitude tensor of size "
            << ampSize << " elements\n";

  // Sphinx: Amplitudes #6

  // Query the free memory on Device
  std::size_t freeSize{0}, totalSize{0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  const std::size_t scratchSize = (freeSize - (freeSize % 4096)) / 2; // use half of available memory with alignment
  void *d_scratch{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  std::cout << "Allocated " << scratchSize << " bytes of scratch memory on GPU\n";

  // Sphinx: Amplitudes #7

  // Create the initial quantum state
  cutensornetState_t quantumState;
  HANDLE_CUTN_ERROR(cutensornetCreateState(cutnHandle, CUTENSORNET_STATE_PURITY_PURE, numQubits, qubitDims.data(),
                    CUDA_C_64F, &quantumState));
  std::cout << "Created the initial quantum state\n";

  // Sphinx: Amplitudes #8

  // Construct the final quantum circuit state (apply quantum gates) for the GHZ circuit
  int64_t id;
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{0}}.data(),
                    d_gateH, nullptr, 1, 0, 1, &id));
  for(int32_t i = 1; i < numQubits; ++i) {
    // Apply Controlled-X gates
    HANDLE_CUTN_ERROR( cutensornetStateApplyControlledTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{i-1}}.data(),
    nullptr , 1, std::vector<int32_t>{{i}}.data(), d_gateX, nullptr, 1, 0, 1, &id  ) );
  }
  std::cout << "Applied quantum gates\n";

  // Sphinx: Amplitudes #9

  // Specify the quantum circuit amplitudes accessor
  cutensornetStateAccessor_t accessor;
  HANDLE_CUTN_ERROR(cutensornetCreateAccessor(cutnHandle, quantumState, numFixedModes, fixedModes.data(),
                    nullptr, &accessor)); // using default strides
  std::cout << "Created the specified quantum circuit amplitudes accessor\n";

  // Sphinx: Amplitudes #10

  // Configure the computation of the slice of the specified quantum circuit amplitudes tensor
  const int32_t numHyperSamples = 8; // desired number of hyper samples used in the tensor network contraction path finder
  HANDLE_CUTN_ERROR(cutensornetAccessorConfigure(cutnHandle, accessor,
                    CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples, sizeof(numHyperSamples)));

  // Sphinx: Amplitudes #11

  // Prepare the computation of the specified slice of the quantum circuit amplitudes tensor
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  std::cout << "Created the workspace descriptor\n";
  HANDLE_CUTN_ERROR(cutensornetAccessorPrepare(cutnHandle, accessor, scratchSize, workDesc, 0x0));
  std::cout << "Prepared the computation of the specified slice of the quantum circuit amplitudes tensor\n";
  double flops {0.0};
  HANDLE_CUTN_ERROR(cutensornetAccessorGetInfo(cutnHandle, accessor,
                    CUTENSORNET_ACCESSOR_INFO_FLOPS, &flops, sizeof(flops)));
  std::cout << "Total flop count = " << (flops/1e9) << " GFlop\n";

  // Sphinx: Amplitudes #12

  // Attach the workspace buffer
  int64_t worksize {0};
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

  // Sphinx: Amplitudes #13

  // Compute the specified slice of the quantum circuit amplitudes tensor
  std::complex<double> stateNorm2{0.0,0.0};
  HANDLE_CUTN_ERROR(cutensornetAccessorCompute(cutnHandle, accessor, fixedValues.data(),
                    workDesc, d_amp, static_cast<void*>(&stateNorm2), 0x0));
  std::cout << "Computed the specified slice of the quantum circuit amplitudes tensor\n";
  std::vector<std::complex<double>> h_amp(ampSize);
  HANDLE_CUDA_ERROR(cudaMemcpy(h_amp.data(), d_amp, ampSize * (2 * fp64size), cudaMemcpyDeviceToHost));
  std::cout << "Amplitudes slice for " << (numQubits - numFixedModes) << " qubits:\n";
  for(std::size_t i = 0; i < ampSize; ++i) {
    std::cout << " " << h_amp[i] << std::endl;
  }
  std::cout << "Squared 2-norm of the state = (" << stateNorm2.real() << ", " << stateNorm2.imag() << ")\n";

  // Sphinx: Amplitudes #14

  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  std::cout << "Destroyed the workspace descriptor\n";

  // Destroy the quantum circuit amplitudes accessor
  HANDLE_CUTN_ERROR(cutensornetDestroyAccessor(accessor));
  std::cout << "Destroyed the quantum circuit amplitudes accessor\n";

  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));
  std::cout << "Destroyed the quantum circuit state\n";

  HANDLE_CUDA_ERROR(cudaFree(d_scratch));
  HANDLE_CUDA_ERROR(cudaFree(d_amp));
  HANDLE_CUDA_ERROR(cudaFree(d_gateX));
  HANDLE_CUDA_ERROR(cudaFree(d_gateH));
  std::cout << "Freed memory on GPU\n";

  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
  std::cout << "Finalized the cuTensorNet library\n";

  return 0;
}
