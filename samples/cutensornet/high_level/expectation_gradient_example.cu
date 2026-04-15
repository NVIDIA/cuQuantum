/* Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * Expectation value with gradients for a 4-qubit circuit.
 *
 * Circuit:
 *   q0 --- Ry [grad] -------------------- CNOT(0,1) ---
 *   q1 --- H ---CNOT(1,2)--- Rx -----------------------
 *   q2 --------------------- CNOT(2,3) -- Rz [grad]----
 *   q3 --- H ------------------------------------------
 *
 * Hamiltonian: H = 2.0*XYZZ + 3.0*IZZI + 5.0*ZIYY.
 * Gradients are computed for Ry on q0 and Rz on q2.
 */

// Sphinx: Expectation Gradient #1

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <complex>
#include <vector>
#include <iostream>
#include <cmath>

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

static constexpr std::size_t fp64size = sizeof(double);
static constexpr std::size_t gate1Size = 4 * (2 * fp64size);
static constexpr std::size_t gate2Size = 16 * (2 * fp64size);

static void createHadamard(std::complex<double>* gate)
{
  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  gate[0] = std::complex<double>(inv_sqrt2, 0.0);
  gate[1] = std::complex<double>(inv_sqrt2, 0.0);
  gate[2] = std::complex<double>(inv_sqrt2, 0.0);
  gate[3] = std::complex<double>(-inv_sqrt2, 0.0);
}

// CNOT: control q0, target q1. |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>.
static void createCNOT(std::complex<double>* gate)
{
  for (int i = 0; i < 16; i++) gate[i] = std::complex<double>(0.0, 0.0);
  gate[0]  = std::complex<double>(1.0, 0.0);
  gate[5]  = std::complex<double>(1.0, 0.0);
  gate[11] = std::complex<double>(1.0, 0.0);
  gate[14] = std::complex<double>(1.0, 0.0);
}

static void createRXGate(double theta, std::complex<double>* gate)
{
  const double c = std::cos(theta / 2.0);
  const double s = -std::sin(theta / 2.0);
  gate[0] = std::complex<double>(c, 0.0);
  gate[1] = std::complex<double>(0.0, s);
  gate[2] = std::complex<double>(0.0, s);
  gate[3] = std::complex<double>(c, 0.0);
}

static void createRYGate(double theta, std::complex<double>* gate)
{
  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);
  gate[0] = std::complex<double>(c, 0.0);
  gate[1] = std::complex<double>(-s, 0.0);
  gate[2] = std::complex<double>(s, 0.0);
  gate[3] = std::complex<double>(c, 0.0);
}

static void createRZGate(double theta, std::complex<double>* gate)
{
  gate[0] = std::complex<double>(std::cos(theta/2), -std::sin(theta/2));
  gate[1] = std::complex<double>(0.0, 0.0);
  gate[2] = std::complex<double>(0.0, 0.0);
  gate[3] = std::complex<double>(std::cos(theta/2), std::sin(theta/2));
}

static void createPauliX(std::complex<double>* op)
{
  op[0] = std::complex<double>(0.0, 0.0);
  op[1] = std::complex<double>(1.0, 0.0);
  op[2] = std::complex<double>(1.0, 0.0);
  op[3] = std::complex<double>(0.0, 0.0);
}

static void createPauliY(std::complex<double>* op)
{
  op[0] = std::complex<double>(0.0, 0.0);
  op[1] = std::complex<double>(0.0, -1.0);
  op[2] = std::complex<double>(0.0, 1.0);
  op[3] = std::complex<double>(0.0, 0.0);
}

static void createPauliZ(std::complex<double>* op)
{
  op[0] = std::complex<double>(1.0, 0.0);
  op[1] = std::complex<double>(0.0, 0.0);
  op[2] = std::complex<double>(0.0, 0.0);
  op[3] = std::complex<double>(-1.0, 0.0);
}

static void createIdentity(std::complex<double>* op)
{
  op[0] = std::complex<double>(1.0, 0.0);
  op[1] = std::complex<double>(0.0, 0.0);
  op[2] = std::complex<double>(0.0, 0.0);
  op[3] = std::complex<double>(1.0, 0.0);
}

int main()
{
  static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

  // Sphinx: Expectation Gradient #2

  // Quantum state configuration
  constexpr int32_t numQubits = 4;
  const std::vector<int64_t> qubitDims(numQubits, 2);
  const double theta = M_PI / 4.0;
  std::cout << "Quantum circuit: " << numQubits << " qubits (expectation gradient)\n";

  // Sphinx: Expectation Gradient #3

  // Initialize the cuTensorNet library
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  cutensornetHandle_t cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  cudaStream_t stream;
  HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
  std::cout << "Initialized cuTensorNet library on GPU 0\n";

  // Sphinx: Expectation Gradient #4

  // Define necessary quantum gate tensors in Host memory
  std::complex<double> h_gateH[4], h_gateCX[16], h_gateRx[4], h_gateRy[4], h_gateRz[4];
  std::complex<double> h_gateX[4], h_gateY[4], h_gateZ[4], h_gateI[4];
  createHadamard(h_gateH);
  createCNOT(h_gateCX);
  createRXGate(theta, h_gateRx);
  createRYGate(theta, h_gateRy);
  createRZGate(theta, h_gateRz);
  createPauliX(h_gateX);
  createPauliY(h_gateY);
  createPauliZ(h_gateZ);
  createIdentity(h_gateI);

  // Copy quantum gates to Device memory
  void *d_gateH{nullptr}, *d_gateCX{nullptr}, *d_gateRx{nullptr}, *d_gateRy{nullptr}, *d_gateRz{nullptr};
  void *d_gateX{nullptr}, *d_gateY{nullptr}, *d_gateZ{nullptr}, *d_gateI{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateH, gate1Size));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateCX, gate2Size));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateRx, gate1Size));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateRy, gate1Size));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateRz, gate1Size));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateX, gate1Size));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateY, gate1Size));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateZ, gate1Size));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gateI, gate1Size));
  std::cout << "Allocated quantum gate memory on GPU\n";
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateH, h_gateH, gate1Size, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateCX, h_gateCX, gate2Size, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateRx, h_gateRx, gate1Size, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateRy, h_gateRy, gate1Size, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateRz, h_gateRz, gate1Size, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateX, h_gateX, gate1Size, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateY, h_gateY, gate1Size, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateZ, h_gateZ, gate1Size, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(d_gateI, h_gateI, gate1Size, cudaMemcpyHostToDevice));
  std::cout << "Copied quantum gates to GPU memory\n";

  // Allocate gradient output buffers (must be allocated before applying gates with gradient)
  void *d_gradRy{nullptr}, *d_gradRz{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gradRy, gate1Size));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gradRz, gate1Size));
  HANDLE_CUDA_ERROR(cudaMemset(d_gradRy, 0, gate1Size));
  HANDLE_CUDA_ERROR(cudaMemset(d_gradRz, 0, gate1Size));
  std::cout << "Allocated gradient buffers on GPU\n";

  // Sphinx: Expectation Gradient #5

  // Query the free memory on Device
  std::size_t freeSize{0}, totalSize{0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  const std::size_t scratchSize = (freeSize - (freeSize % 4096)) / 2;
  void *d_scratch{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  std::cout << "Allocated " << scratchSize << " bytes of scratch memory on GPU\n";

  // Sphinx: Expectation Gradient #6

  // Create the initial quantum state
  cutensornetState_t quantumState;
  HANDLE_CUTN_ERROR(cutensornetCreateState(cutnHandle, CUTENSORNET_STATE_PURITY_PURE, numQubits, qubitDims.data(),
                    CUDA_C_64F, &quantumState));
  std::cout << "Created the initial quantum state\n";

  // Sphinx: Expectation Gradient #7

  // Construct the final quantum circuit state (apply quantum gates); register Ry on q0 and Rz on q2 for gradient
  
  int64_t id;
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{1}}.data(),
                    d_gateH, nullptr, 1, 0, 1, &id));
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{3}}.data(),
                    d_gateH, nullptr, 1, 0, 1, &id));
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 2, std::vector<int32_t>{{1, 2}}.data(),
                    d_gateCX, nullptr, 1, 0, 1, &id));
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 2, std::vector<int32_t>{{2, 3}}.data(),
                    d_gateCX, nullptr, 1, 0, 1, &id));
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperatorWithGradient(cutnHandle, quantumState, 1, std::vector<int32_t>{{0}}.data(),
                    d_gateRy, nullptr, 0, 0, 1, d_gradRy, nullptr, &id));
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 1, std::vector<int32_t>{{1}}.data(),
                    d_gateRx, nullptr, 0, 0, 1, &id));
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperatorWithGradient(cutnHandle, quantumState, 1, std::vector<int32_t>{{2}}.data(),
                    d_gateRz, nullptr, 0, 0, 1, d_gradRz, nullptr, &id));
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(cutnHandle, quantumState, 2, std::vector<int32_t>{{0, 1}}.data(),
                    d_gateCX, nullptr, 1, 0, 1, &id));
  std::cout << "Applied quantum gates (Ry on q0 and Rz on q2 registered for gradient)\n";

  // Sphinx: Expectation Gradient #8

  // Create an empty tensor network operator and append Hamiltonian terms: 2.0*XYZZ, 3.0*IZZI, 5.0*ZIYY, 
  cutensornetNetworkOperator_t hamiltonian;
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkOperator(cutnHandle, numQubits, qubitDims.data(), CUDA_C_64F, &hamiltonian));
  {
    const int32_t numModes[] = {1, 1, 1, 1};
    const int32_t modes0[] = {0}, modes1[] = {1}, modes2[] = {2}, modes3[] = {3};
    const int32_t * stateModes[] = {modes0, modes1, modes2, modes3};
    const void * gateData[] = {d_gateX, d_gateY, d_gateZ, d_gateZ};
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(cutnHandle, hamiltonian, cuDoubleComplex{2.0, 0.0},
                      4, numModes, stateModes, NULL, gateData, &id));
  }
  {
    const int32_t numModes[] = {1, 1, 1, 1};
    const int32_t modes0[] = {0}, modes1[] = {1}, modes2[] = {2}, modes3[] = {3};
    const int32_t * stateModes[] = {modes0, modes1, modes2, modes3};
    const void * gateData[] = {d_gateI, d_gateZ, d_gateZ, d_gateI};
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(cutnHandle, hamiltonian, cuDoubleComplex{3.0, 0.0},
                      4, numModes, stateModes, NULL, gateData, &id));
  }
  {
    const int32_t numModes[] = {1, 1, 1, 1};
    const int32_t modes0[] = {0}, modes1[] = {1}, modes2[] = {2}, modes3[] = {3};
    const int32_t * stateModes[] = {modes0, modes1, modes2, modes3};
    const void * gateData[] = {d_gateZ, d_gateI, d_gateY, d_gateY};
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(cutnHandle, hamiltonian, cuDoubleComplex{5.0, 0.0},
                      4, numModes, stateModes, NULL, gateData, &id));
  }
  
  std::cout << "Constructed a tensor network operator: 2.0*XYZZ + 3.0*IZZI + 5.0*ZIYY \n";

  // Sphinx: Expectation Gradient #9

  // Specify the quantum circuit expectation value
  cutensornetStateExpectation_t expectation;
  HANDLE_CUTN_ERROR(cutensornetCreateExpectation(cutnHandle, quantumState, hamiltonian, &expectation));
  std::cout << "Created the specified quantum circuit expectation value\n";

  // Sphinx: Expectation Gradient #10

  // Configure the computation of the specified quantum circuit expectation value
  const int32_t numHyperSamples = 8; // desired number of hyper samples used in the tensor network contraction path finder
  HANDLE_CUTN_ERROR(cutensornetExpectationConfigure(cutnHandle, expectation,
                    CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples, sizeof(numHyperSamples)));

  // Sphinx: Expectation Gradient #11

  // Prepare the specified quantum circuit expectation value for computation (with gradient support)
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  std::cout << "Created the workspace descriptor\n";
  HANDLE_CUTN_ERROR(cutensornetExpectationPrepare(cutnHandle, expectation, scratchSize, workDesc, stream));
  std::cout << "Prepared the specified quantum circuit expectation value (gradient backward)\n";

  // Sphinx: Expectation Gradient #12

  // Attach the workspace buffer
  int64_t worksize{0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle, workDesc,
                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                      CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH,
                      &worksize));
  std::cout << "Required scratch GPU workspace size (bytes) = " << worksize << std::endl;
  if (worksize <= scratchSize) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  } else {
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  std::cout << "Set the workspace buffer\n";

  // Sphinx: Expectation Gradient #13

  // Compute the specified quantum circuit expectation value and gradients (backward pass)
  std::complex<double> expectVal{0.0, 0.0};
  cuDoubleComplex expectationAdjoint = {1.0, 0.0};
  HANDLE_CUTN_ERROR(cutensornetExpectationComputeWithGradientsBackward(cutnHandle, expectation,
                    0, &expectationAdjoint, nullptr, workDesc,
                    static_cast<void*>(&expectVal), nullptr, stream));
  HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
  std::cout << "Computed the specified quantum circuit expectation value and gradients\n";
  std::cout << "Expectation value = (" << expectVal.real() << ", " << expectVal.imag() << ")\n";
  

  // Copy gradients back to host and print
  std::complex<double> h_gradRy[4], h_gradRz[4];
  HANDLE_CUDA_ERROR(cudaMemcpy(h_gradRy, d_gradRy, gate1Size, cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(h_gradRz, d_gradRz, gate1Size, cudaMemcpyDeviceToHost));
  std::cout << "Gradient d<H>/d(Ry on q0):\n";
  std::cout << "  [0,0]: (" << h_gradRy[0].real() << ", " << h_gradRy[0].imag() << ")\n";
  std::cout << "  [0,1]: (" << h_gradRy[1].real() << ", " << h_gradRy[1].imag() << ")\n";
  std::cout << "  [1,0]: (" << h_gradRy[2].real() << ", " << h_gradRy[2].imag() << ")\n";
  std::cout << "  [1,1]: (" << h_gradRy[3].real() << ", " << h_gradRy[3].imag() << ")\n";
  std::cout << "Gradient d<H>/d(Rz on q2):\n";
  std::cout << "  [0,0]: (" << h_gradRz[0].real() << ", " << h_gradRz[0].imag() << ")\n";
  std::cout << "  [0,1]: (" << h_gradRz[1].real() << ", " << h_gradRz[1].imag() << ")\n";
  std::cout << "  [1,0]: (" << h_gradRz[2].real() << ", " << h_gradRz[2].imag() << ")\n";
  std::cout << "  [1,1]: (" << h_gradRz[3].real() << ", " << h_gradRz[3].imag() << ")\n";

  // Sphinx: Expectation Gradient #14

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

  HANDLE_CUDA_ERROR(cudaFree(d_scratch));
  HANDLE_CUDA_ERROR(cudaFree(d_gradRy));
  HANDLE_CUDA_ERROR(cudaFree(d_gradRz));
  HANDLE_CUDA_ERROR(cudaFree(d_gateRz));
  HANDLE_CUDA_ERROR(cudaFree(d_gateRy));
  HANDLE_CUDA_ERROR(cudaFree(d_gateRx));
  HANDLE_CUDA_ERROR(cudaFree(d_gateCX));
  HANDLE_CUDA_ERROR(cudaFree(d_gateI));
  HANDLE_CUDA_ERROR(cudaFree(d_gateZ));
  HANDLE_CUDA_ERROR(cudaFree(d_gateY));
  HANDLE_CUDA_ERROR(cudaFree(d_gateX));
  HANDLE_CUDA_ERROR(cudaFree(d_gateH));
  std::cout << "Freed memory on GPU\n";

  HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
  HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
  std::cout << "Finalized the cuTensorNet library\n";

  return 0;
}
