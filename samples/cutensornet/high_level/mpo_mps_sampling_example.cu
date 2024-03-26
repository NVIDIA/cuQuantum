/* Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */  

// Sphinx: MPO-MPS Sampler #1

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <complex>
#include <random>
#include <functional>
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
  constexpr cudaDataType_t dataType = CUDA_C_64F;

  // Sphinx: MPO-MPS Sampler #2

  // Quantum state configuration
  const int64_t numSamples = 128; // number of samples to generate
  const int32_t numQudits = 6;    // number of qudits
  const int64_t quditDim = 2;     // dimension of all qudits
  const std::vector<int64_t> quditDims(numQudits, quditDim); // qudit dimensions
  const int64_t mpsBondDim = 8;   // maximum MPS bond dimension
  const int64_t mpoBondDim = 2;   // maximum MPO bond dimension
  const int32_t mpoNumLayers = 5; // number of MPO layers
  constexpr int32_t mpoNumSites = 4;  // number of MPO sites
  std::cout << "Quantum circuit: " << numQudits << " qudits; " << numSamples << " samples\n";

  /* Action of five alternating four-site MPO gates (operators)
     on the 6-quqit quantum register (illustration):

  Q----X---------X---------X----
       |         |         |
  Q----X---------X---------X----
       |         |         |
  Q----X----X----X----X----X----
       |    |    |    |    |
  Q----X----X----X----X----X----
            |         |
  Q---------X---------X---------
            |         |
  Q---------X---------X---------

    |layer|
  */
  static_assert(mpoNumSites > 1, "Number of MPO sites must be larger than one!");

  // Random number generator
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  auto rnd = std::bind(distribution, generator);

  // Sphinx: MPO-MPS Sampler #3

  // Initialize the cuTensorNet library
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  cutensornetHandle_t cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreate(&cutnHandle));
  std::cout << "Initialized cuTensorNet library on GPU 0\n";

  // Sphinx: MPO-MPS Sampler #4

  /* MPO tensor mode numeration (open boundary condition):

     2            3                   2
     |            |                   |
     X--1------0--X--2---- ... ----0--X
     |            |                   |
     0            1                   1

  */
  // Define shapes of the MPO tensors
  using GateData = std::vector<std::complex<double>>;
  using ModeExtents = std::vector<int64_t>;
  std::vector<GateData> mpoTensors(mpoNumSites);        // MPO tensors
  std::vector<ModeExtents> mpoModeExtents(mpoNumSites); // mode extents for MPO tensors
  int64_t upperBondDim = 1;
  for(int tensId = 0; tensId < (mpoNumSites / 2); ++tensId) {
    const auto leftBondDim = std::min(mpoBondDim, upperBondDim);
    const auto rightBondDim = std::min(mpoBondDim, leftBondDim * (quditDim * quditDim));
    if(tensId == 0) {
      mpoModeExtents[tensId] = {quditDim, rightBondDim, quditDim};
    }else{
      mpoModeExtents[tensId] = {leftBondDim, quditDim, rightBondDim, quditDim};
    }
    upperBondDim = rightBondDim;
  }
  auto centralBondDim = upperBondDim;
  if(mpoNumSites % 2 != 0) {
    const int tensId = mpoNumSites / 2;
    mpoModeExtents[tensId] = {centralBondDim, quditDim, centralBondDim, quditDim};
  }
  upperBondDim = 1;
  for(int tensId = (mpoNumSites-1); tensId >= (mpoNumSites / 2) + (mpoNumSites % 2); --tensId) {
    const auto rightBondDim = std::min(mpoBondDim, upperBondDim);
    const auto leftBondDim = std::min(mpoBondDim, rightBondDim * (quditDim * quditDim));
    if(tensId == (mpoNumSites-1)) {
      mpoModeExtents[tensId] = {leftBondDim, quditDim, quditDim};
    }else{
      mpoModeExtents[tensId] = {leftBondDim, quditDim, rightBondDim, quditDim};
    }
    upperBondDim = leftBondDim;
  }
  // Fill in the MPO tensors with random data on Host
  std::vector<const int64_t*> mpoModeExtentsPtr(mpoNumSites, nullptr);
  for(int tensId = 0; tensId < mpoNumSites; ++tensId) {
    mpoModeExtentsPtr[tensId] = mpoModeExtents[tensId].data();
    const auto tensRank = mpoModeExtents[tensId].size();
    int64_t tensVol = 1;
    for(int i = 0; i < tensRank; ++i) {
      tensVol *= mpoModeExtents[tensId][i];
    }
    mpoTensors[tensId].resize(tensVol);
    for(int64_t i = 0; i < tensVol; ++i) {
      mpoTensors[tensId][i] = std::complex<double>(rnd(), rnd());
    }
  }
  // Allocate and copy MPO tensors into Device memory
  std::vector<void*> d_mpoTensors(mpoNumSites, nullptr);
  for(int tensId = 0; tensId < mpoNumSites; ++tensId) {
    const auto tensRank = mpoModeExtents[tensId].size();
    int64_t tensVol = 1;
    for(int i = 0; i < tensRank; ++i) {
      tensVol *= mpoModeExtents[tensId][i];
    }
    HANDLE_CUDA_ERROR(cudaMalloc(&(d_mpoTensors[tensId]), 
                                 std::size_t(tensVol) * (2 * fp64size)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_mpoTensors[tensId], mpoTensors[tensId].data(),
                                 std::size_t(tensVol) * (2 * fp64size), cudaMemcpyHostToDevice));
  }
  std::cout << "Allocated and defined MPO tensors in GPU memory\n";

  // Sphinx: MPO-MPS Sampler #5

  // Define the MPS representation and allocate memory buffers for the MPS tensors
  std::vector<ModeExtents> mpsModeExtents(numQudits);
  std::vector<int64_t*> mpsModeExtentsPtr(numQudits, nullptr);
  std::vector<void*> d_mpsTensors(numQudits, nullptr);
  upperBondDim = 1;
  for (int tensId = 0; tensId < (numQudits / 2); ++tensId) {
    const auto leftBondDim = std::min(mpsBondDim, upperBondDim);
    const auto rightBondDim = std::min(mpsBondDim, leftBondDim * quditDim);
    if (tensId == 0) { // left boundary MPS tensor  
      mpsModeExtents[tensId] = {quditDim, rightBondDim};
      HANDLE_CUDA_ERROR(cudaMalloc(&(d_mpsTensors[tensId]),
                                   std::size_t(quditDim * rightBondDim) * (2 * fp64size)));
    } else { // middle MPS tensors
      mpsModeExtents[tensId] = {leftBondDim, quditDim, rightBondDim};
      HANDLE_CUDA_ERROR(cudaMalloc(&(d_mpsTensors[tensId]),
                                   std::size_t(leftBondDim * quditDim * rightBondDim) * (2 * fp64size)));
    }
    upperBondDim = rightBondDim;
    mpsModeExtentsPtr[tensId] = mpsModeExtents[tensId].data();
  }
  centralBondDim = upperBondDim;
  if(numQudits % 2 != 0) {
    const int tensId = numQudits / 2;
    mpsModeExtents[tensId] = {centralBondDim, quditDim, centralBondDim};
    mpsModeExtentsPtr[tensId] = mpsModeExtents[tensId].data();
  }
  upperBondDim = 1;
  for (int tensId = (numQudits-1); tensId >= (numQudits / 2) + (numQudits % 2); --tensId) {
    const auto rightBondDim = std::min(mpsBondDim, upperBondDim);
    const auto leftBondDim = std::min(mpsBondDim, rightBondDim * quditDim);
    if (tensId == (numQudits-1)) { // right boundary MPS tensor  
      mpsModeExtents[tensId] = {leftBondDim, quditDim};
      HANDLE_CUDA_ERROR(cudaMalloc(&(d_mpsTensors[tensId]),
                                   std::size_t(leftBondDim * quditDim) * (2 * fp64size)));
    } else { // middle MPS tensors
      mpsModeExtents[tensId] = {leftBondDim, quditDim, rightBondDim};
      HANDLE_CUDA_ERROR(cudaMalloc(&(d_mpsTensors[tensId]),
                                   std::size_t(leftBondDim * quditDim * rightBondDim) * (2 * fp64size)));
    }
    upperBondDim = leftBondDim;
    mpsModeExtentsPtr[tensId] = mpsModeExtents[tensId].data();
  }
  std::cout << "Allocated MPS tensors in GPU memory\n";

  // Sphinx: MPO-MPS Sampler #6

  // Query free memory on Device and allocate a scratch buffer
  std::size_t freeSize {0}, totalSize {0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  const std::size_t scratchSize = (freeSize - (freeSize % 4096)) / 2; // use half of available memory with alignment
  void *d_scratch {nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  std::cout << "Allocated " << scratchSize << " bytes of scratch memory on GPU: "
            << "[" << d_scratch << ":" << (void*)(((char*)(d_scratch))  + scratchSize) << ")\n";

  // Sphinx: MPO-MPS Sampler #7

  // Create the initial quantum state
  cutensornetState_t quantumState;
  HANDLE_CUTN_ERROR(cutensornetCreateState(cutnHandle, CUTENSORNET_STATE_PURITY_PURE, numQudits, quditDims.data(),
                    dataType, &quantumState));
  std::cout << "Created the initial (vacuum) quantum state\n";

  // Sphinx: MPO-MPS Sampler #8

  // Construct the MPO tensor network operators
  int64_t componentId;
  cutensornetNetworkOperator_t tnOperator1;
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkOperator(cutnHandle, numQudits, quditDims.data(),
                    dataType, &tnOperator1));
  HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendMPO(cutnHandle, tnOperator1, make_cuDoubleComplex(1.0, 0.0),
                    mpoNumSites, std::vector<int32_t>({0,1,2,3}).data(),
                    mpoModeExtentsPtr.data(), /*strides=*/nullptr,
                    std::vector<const void*>(d_mpoTensors.cbegin(), d_mpoTensors.cend()).data(),
                    CUTENSORNET_BOUNDARY_CONDITION_OPEN, &componentId));
  cutensornetNetworkOperator_t tnOperator2;
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkOperator(cutnHandle, numQudits, quditDims.data(),
                    dataType, &tnOperator2));
  HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendMPO(cutnHandle, tnOperator2, make_cuDoubleComplex(1.0, 0.0),
                    mpoNumSites, std::vector<int32_t>({2,3,4,5}).data(),
                    mpoModeExtentsPtr.data(), /*strides=*/nullptr,
                    std::vector<const void*>(d_mpoTensors.cbegin(), d_mpoTensors.cend()).data(),
                    CUTENSORNET_BOUNDARY_CONDITION_OPEN, &componentId));
  std::cout << "Constructed two MPO tensor network operators\n";

  // Sphinx: MPO-MPS Sampler #9

  // Apply the MPO tensor network operators to the quantum state
  for(int layer = 0; layer < mpoNumLayers; ++layer) {
    int64_t operatorId;
    if(layer % 2 == 0) {
      HANDLE_CUTN_ERROR(cutensornetStateApplyNetworkOperator(cutnHandle, quantumState, tnOperator1,
                        1, 0, 0, &operatorId));
    }else{
      HANDLE_CUTN_ERROR(cutensornetStateApplyNetworkOperator(cutnHandle, quantumState, tnOperator2,
        1, 0, 0, &operatorId));
    }
  }
  std::cout << "Applied " << mpoNumLayers << " MPO gates to the quantum state\n";

  // Sphinx: MPO-MPS Sampler #10

  // Specify the target MPS representation (use default strides)
  HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(cutnHandle, quantumState, 
                    CUTENSORNET_BOUNDARY_CONDITION_OPEN, mpsModeExtentsPtr.data(), /*strides=*/nullptr ));
  std::cout << "Finalized the form of the desired MPS representation\n";

  // Sphinx: MPO-MPS Sampler #11

  // Set up the SVD method for bonds truncation (optional)
  cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ; 
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(cutnHandle, quantumState, 
                    CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO, &algo, sizeof(algo)));
  cutensornetStateMPOApplication_t mpsApplication = CUTENSORNET_STATE_MPO_APPLICATION_EXACT; 
  // Use exact MPS-MPO application as extent of 8 can faithfully represent a 6 qubit state (optional)
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(cutnHandle, quantumState, 
                    CUTENSORNET_STATE_CONFIG_MPS_MPO_APPLICATION, &mpsApplication, sizeof(mpsApplication)));
  std::cout << "Configured the MPS computation\n";

  // Sphinx: MPO-MPS Sampler #12

  // Prepare the MPS computation and attach a workspace buffer
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  std::cout << "Created the workspace descriptor\n";
  HANDLE_CUTN_ERROR(cutensornetStatePrepare(cutnHandle, quantumState, scratchSize, workDesc, 0x0));
  int64_t worksize {0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
                                                      workDesc,
                                                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                      CUTENSORNET_MEMSPACE_DEVICE,
                                                      CUTENSORNET_WORKSPACE_SCRATCH,
                                                      &worksize));
  std::cout << "Scratch GPU workspace size (bytes) for the MPS computation = " << worksize << std::endl;
  if(worksize <= scratchSize) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  }else{
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  std::cout << "Set the workspace buffer and prepared the MPS computation\n";

  // Sphinx: MPO-MPS Sampler #13

  // Compute the MPS factorization of the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetStateCompute(cutnHandle, quantumState, 
                    workDesc, mpsModeExtentsPtr.data(), /*strides=*/nullptr, d_mpsTensors.data(), 0));
  std::cout << "Computed the MPS factorization of the quantum circuit state\n";
  
  // Sphinx: MPO-MPS Sampler #14

  // Create the quantum circuit sampler
  cutensornetStateSampler_t sampler;
  HANDLE_CUTN_ERROR(cutensornetCreateSampler(cutnHandle, quantumState, numQudits, nullptr, &sampler));
  std::cout << "Created the quantum circuit sampler\n";

  // Sphinx: MPO-MPS Sampler #15

  // Configure the quantum circuit sampler
  const int32_t numHyperSamples = 8; // desired number of hyper samples used in the tensor network contraction path finder
  HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(cutnHandle, sampler,
                    CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES, &numHyperSamples, sizeof(numHyperSamples)));

  // Sphinx: MPO-MPS Sampler #16

  // Prepare the quantum circuit sampler
  HANDLE_CUTN_ERROR(cutensornetSamplerPrepare(cutnHandle, sampler, scratchSize, workDesc, 0x0));
  std::cout << "Prepared the quantum circuit state sampler\n";

  // Sphinx: MPO-MPS Sampler #17

  // Attach the workspace buffer
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
                                                      workDesc,
                                                      CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
                                                      CUTENSORNET_MEMSPACE_DEVICE,
                                                      CUTENSORNET_WORKSPACE_SCRATCH,
                                                      &worksize));
  std::cout << "Scratch GPU workspace size (bytes) for MPS Sampling = " << worksize << std::endl;
  assert(worksize > 0);
  if(worksize <= scratchSize) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                      CUTENSORNET_WORKSPACE_SCRATCH, d_scratch, worksize));
  }else{
    std::cout << "ERROR: Insufficient workspace size on Device!\n";
    std::abort();
  }
  std::cout << "Set the workspace buffer\n";

  // Sphinx: MPO-MPS Sampler #18

  // Sample the quantum circuit state
  std::vector<int64_t> samples(numQudits * numSamples); // samples[SampleId][QuditId] reside in Host memory
  HANDLE_CUTN_ERROR(cutensornetSamplerSample(cutnHandle, sampler, numSamples, workDesc, samples.data(), 0));
  std::cout << "Performed quantum circuit state sampling\n";
  std::cout << "Generated samples:\n";
  for(int64_t i = 0; i < numSamples; ++i) {
    for(int64_t j = 0; j < numQudits; ++j) std::cout << " " << samples[i * numQudits + j];
    std::cout << std::endl;
  }

  // Sphinx: MPO-MPS Sampler #19

  // Destroy the quantum circuit sampler
  HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler));
  std::cout << "Destroyed the quantum circuit state sampler\n";

  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  std::cout << "Destroyed the workspace descriptor\n";

  // Destroy the MPO tensor network operators
  HANDLE_CUTN_ERROR(cutensornetDestroyNetworkOperator(tnOperator2));
  HANDLE_CUTN_ERROR(cutensornetDestroyNetworkOperator(tnOperator1));
  std::cout << "Destroyed the MPO tensor network operators\n";

  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(quantumState));
  std::cout << "Destroyed the quantum circuit state\n";

  // Free GPU buffers
  HANDLE_CUDA_ERROR(cudaFree(d_scratch));
  for (int32_t i = 0; i < numQudits; ++i) {
    HANDLE_CUDA_ERROR(cudaFree(d_mpsTensors[i]));
  }
  for (int32_t i = 0; i < mpoNumSites; ++i) {
    HANDLE_CUDA_ERROR(cudaFree(d_mpoTensors[i]));
  }
  std::cout << "Freed memory on GPU\n";
  
  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(cutnHandle));
  std::cout << "Finalized the cuTensorNet library\n";

  return 0;
}
