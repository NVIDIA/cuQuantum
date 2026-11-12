/* Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: MixedState #1

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

  // Sphinx: MixedState #2

  // Quantum state configuration: build a mixed (density-matrix) state of a noisy circuit
  constexpr int32_t numQubits = 6; // number of qubits
  const std::vector<int64_t> qubitDims(numQubits, 2); // qubit dimensions
  const std::vector<int32_t> marginalModes({0, 1}); // open qubits defining the marginal (ascending order)
  const int32_t numMarginalModes = marginalModes.size();
  std::cout << "Mixed-state (density-matrix) simulation of a " << numQubits << "-qubit noisy circuit\n";

  // Sphinx: MixedState #3

  // Initialize the cuTensorNet library
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  cutensornetHandle_t cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  std::cout << "Initialized cuTensorNet library on GPU 0\n";

  // Sphinx: MixedState #4

  // Define the quantum gate tensors (Host memory)
  const double invsq2 = 1.0 / std::sqrt(2.0);
  //  Hadamard gate
  const std::vector<std::complex<double>> h_gateH {{invsq2, 0.0},  {invsq2, 0.0},
                                                   {invsq2, 0.0}, {-invsq2, 0.0}};
  //  CX gate
  const std::vector<std::complex<double>> h_gateCX {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
                                                    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  // Define the Kraus operators of an amplitude-damping channel (rate gamma), stored
  // column-major like the gate tensors. K0 = diag(1, sqrt(1-gamma)); K1 has a single
  // entry sqrt(gamma) in the |0><1| position. They satisfy K0^dag K0 + K1^dag K1 = I.
  const double gamma = 0.2;
  const double s = std::sqrt(1.0 - gamma);
  const double g = std::sqrt(gamma);
  const std::vector<std::complex<double>> h_kraus0 {{1.0, 0.0}, {0.0, 0.0},
                                                    {0.0, 0.0}, {s,   0.0}};
  const std::vector<std::complex<double>> h_kraus1 {{0.0, 0.0}, {0.0, 0.0},
                                                    {g,   0.0}, {0.0, 0.0}};

  // Copy the gate and Kraus tensors to Device memory
  void *d_gateH{nullptr}, *d_gateCX{nullptr}, *d_kraus0{nullptr}, *d_kraus1{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateH, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateCX, 16 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_kraus0, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_kraus1, 4 * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateH, h_gateH.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateCX, h_gateCX.data(), 16 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_kraus0, h_kraus0.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_kraus1, h_kraus1.data(), 4 * (2 * fp64size), cudaMemcpyHostToDevice));
  std::cout << "Copied quantum gates and Kraus operators to GPU memory\n";

  // Sphinx: MixedState #5

  // Allocate Device memory for the full reduced density matrix (marginal) and its diagonal
  std::size_t rdmDim = 1;
  for(const auto & mode: marginalModes) rdmDim *= qubitDims[mode];
  const std::size_t rdmSize = rdmDim * rdmDim;           // full RDM has rank 2*numMarginalModes
  void *d_rdm{nullptr}, *d_rdmDiag{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_rdm, rdmSize * (2 * fp64size)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_rdmDiag, rdmDim * (2 * fp64size))); // diagonal has rank numMarginalModes
  std::cout << "Allocated memory for the reduced density matrix and its diagonal\n";

  // Sphinx: MixedState #6

  // Query the free memory on Device and allocate scratch
  std::size_t freeSize{0}, totalSize{0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  const std::size_t scratchSize = (freeSize - (freeSize % 4096)) / 2; // use half of available memory with alignment
  void *d_scratch{nullptr};
  void *h_scratch{nullptr};  // host scratch workspace buffer (attached if the prepare call reports a need)
  int64_t hostWorksize{0};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  std::cout << "Allocated " << scratchSize << " bytes of scratch memory on GPU\n";

  // Sphinx: MixedState #7

  // Create the initial MIXED (density-matrix) quantum state
  cutensornetState_t quantumState;
  HANDLE_CUTN_ERROR(cutensornetCreateState(cutnHandle, CUTENSORNET_STATE_PURITY_MIXED, numQubits, qubitDims.data(),
                    CUDA_C_64F, &quantumState));
  std::cout << "Created the initial mixed quantum state\n";

  // Sphinx: MixedState #8

  // Construct the GHZ circuit, then apply an amplitude-damping channel to qubit 0
  int64_t id;
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{0}}.data(),
                    d_gateH, nullptr, 1, 0, 1, &id));
  for(int32_t i = 1; i < numQubits; ++i) {
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 2, std::vector<int32_t>{{i-1,i}}.data(),
                      d_gateCX, nullptr, 1, 0, 1, &id));
  }
  std::cout << "Applied quantum gates (GHZ circuit)\n";

  // Apply a general (non-unitary) quantum channel; this is only valid for a mixed state.
  void *channelKraus[] = {d_kraus0, d_kraus1};
  int64_t channelId;
  HANDLE_CUTN_ERROR(cutensornetStateApplyGeneralChannel(cutnHandle, quantumState, 1, std::vector<int32_t>{{0}}.data(),
                    2, channelKraus, nullptr, &channelId));
  std::cout << "Applied an amplitude-damping channel to qubit 0\n";

  // Sphinx: MixedState #9

  // Create both the full reduced density matrix and its diagonal (marginal probabilities).
  // The two factories share the same opaque handle type and lifecycle APIs.
  cutensornetStateMarginal_t marginal, marginalDiag;
  HANDLE_CUTN_ERROR(cutensornetCreateMarginal(cutnHandle, quantumState, numMarginalModes, marginalModes.data(),
                    0, nullptr, nullptr, &marginal));
  HANDLE_CUTN_ERROR(cutensornetCreateMarginalDiagonal(cutnHandle, quantumState, numMarginalModes, marginalModes.data(),
                    0, nullptr, nullptr, &marginalDiag));
  std::cout << "Created the reduced density matrix (full) and its diagonal\n";

  // Sphinx: MixedState #10

  // Configure both computations with the same number of hyper samples for the path finder
  const int32_t numHyperSamples = 8;
  HANDLE_CUTN_ERROR(cutensornetMarginalConfigure(cutnHandle, marginal,
                    CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples, sizeof(numHyperSamples)));
  HANDLE_CUTN_ERROR(cutensornetMarginalConfigure(cutnHandle, marginalDiag,
                    CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples, sizeof(numHyperSamples)));

  // Sphinx: MixedState #11

  // Create a workspace descriptor (reused for both computations)
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));

  // Helper lambda: prepare, attach workspace, and compute a marginal handle into d_out
  auto prepareAndCompute = [&](cutensornetStateMarginal_t handle, void *d_out, const char *label) {
    HANDLE_CUTN_ERROR(cutensornetMarginalPrepare(cutnHandle, handle, scratchSize, workDesc, 0x0));
    int64_t worksize {0};
    HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle, workDesc,
                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
    if(worksize > static_cast<int64_t>(scratchSize)) {
      std::cout << "ERROR: Insufficient workspace size on Device for " << label << "!\n";
      std::abort();
    }
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));

    // Query the host scratch workspace size and attach a host scratch buffer if one is required
    HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                        CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH, &hostWorksize));
    std::cout << "Scratch Host workspace size (bytes) = " << hostWorksize << std::endl;
    if(hostWorksize > 0) {
      h_scratch = std::realloc(h_scratch, hostWorksize);
      if(h_scratch == nullptr) { std::cout << "ERROR: Host scratch allocation failed!\n"; std::abort(); }
      HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_HOST,
                        CUTENSORNET_WORKSPACE_SCRATCH, h_scratch, hostWorksize));
    }
    HANDLE_CUTN_ERROR(cutensornetMarginalCompute(cutnHandle, handle, nullptr, workDesc, d_out, 0x0));
  };

  // Sphinx: MixedState #12

  // Compute the full reduced density matrix
  prepareAndCompute(marginal, d_rdm, "reduced density matrix");
  std::vector<std::complex<double>> h_rdm(rdmSize);
  HANDLE_CUDA_ERROR(cudaMemcpy(h_rdm.data(), d_rdm, rdmSize * (2 * fp64size), cudaMemcpyDeviceToHost));
  std::cout << "\nReduced density matrix on qubits {0,1}:\n";
  std::complex<double> trace{0.0, 0.0};
  for(std::size_t i = 0; i < rdmDim; ++i) {
    for(std::size_t j = 0; j < rdmDim; ++j) {
      std::cout << " " << h_rdm[i + j * rdmDim];
    }
    trace += h_rdm[i + i * rdmDim];
    std::cout << std::endl;
  }
  std::cout << "Tr(RDM) = " << trace.real() << " (should be ~1.0)\n";

  // Sphinx: MixedState #13

  // Compute the diagonal of the reduced density matrix (marginal probabilities)
  prepareAndCompute(marginalDiag, d_rdmDiag, "marginal diagonal");
  std::vector<std::complex<double>> h_rdmDiag(rdmDim);
  HANDLE_CUDA_ERROR(cudaMemcpy(h_rdmDiag.data(), d_rdmDiag, rdmDim * (2 * fp64size), cudaMemcpyDeviceToHost));
  std::cout << "\nMarginal probability distribution over qubits {0,1}:\n";
  double probSum = 0.0;
  for(std::size_t i = 0; i < rdmDim; ++i) {
    std::cout << " p[" << i << "] = " << h_rdmDiag[i].real() << std::endl;
    probSum += h_rdmDiag[i].real();
  }
  std::cout << "Sum of probabilities = " << probSum << " (should be ~1.0)\n";

  // Sphinx: MixedState #14

  // Clean up
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  HANDLE_CUTN_ERROR(cutensornetDestroyMarginal(marginalDiag));
  HANDLE_CUTN_ERROR(cutensornetDestroyMarginal(marginal));
  HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));
  std::cout << "\nDestroyed the cuTensorNet objects\n";

  std::free(h_scratch);
  HANDLE_CUDA_ERROR(cudaFree(d_scratch));
  HANDLE_CUDA_ERROR(cudaFree(d_rdmDiag));
  HANDLE_CUDA_ERROR(cudaFree(d_rdm));
  HANDLE_CUDA_ERROR(cudaFree(d_kraus1));
  HANDLE_CUDA_ERROR(cudaFree(d_kraus0));
  HANDLE_CUDA_ERROR(cudaFree(d_gateCX));
  HANDLE_CUDA_ERROR(cudaFree(d_gateH));
  std::cout << "Freed memory on GPU\n";

  HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
  std::cout << "Finalized the cuTensorNet library\n";

  return 0;
}
