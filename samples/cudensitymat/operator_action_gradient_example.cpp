/* Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cudensitymat.h>  // cuDensityMat library header
#include "helpers.h"       // helper functions


// Transverse Ising Hamiltonian with double summation ordering
// and spin-operator fusion, plus fused dissipation terms
#include "transverse_ising_full_fused_noisy_grad.h" // user-defined Liouvillian operator example

#include <cmath>
#include <complex>
#include <vector>
#include <chrono>
#include <iostream>
#include <cassert>


// Number of times to perform operator action on a quantum state
constexpr int NUM_REPEATS = 2;

// Logging verbosity
bool verbose = true;


// Example workflow
void exampleWorkflow(cudensitymatHandle_t handle)
{
  // Define the composite Hilbert space shape and
  // quantum state batch size (number of individual quantum states in a batched simulation)
  const std::vector<int64_t> spaceShape({2,2,2,2}); // dimensions of quantum degrees of freedom
  const int64_t batchSize = 1;                      // number of quantum states per batch

  if (verbose) {
    std::cout << "Hilbert space rank = " << spaceShape.size() << "; Shape = (";
    for (const auto & dimsn: spaceShape)
      std::cout << dimsn << ",";
    std::cout << ")" << std::endl;
    std::cout << "Quantum state batch size = " << batchSize << std::endl;
  }

  // Construct a user-defined Liouvillian operator using a convenience C++ class
  // Note that the constructed Liouvillian operator has some batched coefficients
  UserDefinedLiouvillian liouvillian(handle, spaceShape, batchSize);
  if (verbose)
    std::cout << "Constructed the Liouvillian operator\n";

  // Set and place external user-provided Hamiltonian parameters in GPU memory
  const int32_t numParams = liouvillian.getNumParameters(); // number of external user-provided Hamiltonian parameters
  if (verbose)
    std::cout << "Number of external user-provided Hamiltonian parameters = " << numParams << std::endl;
  std::vector<double> cpuHamParams(numParams * batchSize);
  for (int64_t j = 0; j < batchSize; ++j) {
    for (int32_t i = 0; i < numParams; ++i) {
      cpuHamParams[j * numParams + i] = double(i+1) / double(j+1); // just setting some parameter values for each instance of the batch
    }
  }
  auto * hamiltonianParams = static_cast<double *>(createInitializeArrayGPU(cpuHamParams));
  if (verbose)
    std::cout << "Created an array of external user-provided Hamiltonian parameters in GPU memory\n";

  // Create an array of gradients for the user-provided Hamiltonian parameters in GPU memory
  std::vector<double> cpuHamParamsGrad(numParams * batchSize, 0.0);
  auto * hamiltonianParamsGrad = static_cast<double *>(createInitializeArrayGPU(cpuHamParamsGrad));
  if (verbose)
    std::cout << "Created an array of gradients for the external user-provided Hamiltonian parameters in GPU memory\n";

  // Declare the input quantum state
  cudensitymatState_t inputState;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(handle,
                      CUDENSITYMAT_STATE_PURITY_MIXED,  // pure (state vector) or mixed (density matrix) state
                      spaceShape.size(),
                      spaceShape.data(),
                      batchSize,
                      dataType,
                      &inputState));

  // Query the size of the quantum state storage
  std::size_t storageSize {0}; // only one storage component (tensor) is needed (no tensor factorization)
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(handle,
                      inputState,
                      1,               // only one storage component (tensor)
                      &storageSize));  // storage size in bytes
  const std::size_t stateVolume = storageSize / sizeof(NumericalType);  // quantum state tensor volume (number of elements)
  if (verbose)
    std::cout << "Quantum state storage size (bytes) = " << storageSize << std::endl;

  // Prepare some initial value for the input quantum state batch
  std::vector<NumericalType> inputStateValue(stateVolume);
  if constexpr (std::is_same_v<NumericalType, float>) {
    for (std::size_t i = 0; i < stateVolume; ++i) {
      inputStateValue[i] = 1.0f / float(i+1); // just some value
    }
  } else if constexpr (std::is_same_v<NumericalType, double>) {
    for (std::size_t i = 0; i < stateVolume; ++i) {
      inputStateValue[i] = 1.0 / double(i+1); // just some value
    }
  } else if constexpr (std::is_same_v<NumericalType, std::complex<float>>) {
    for (std::size_t i = 0; i < stateVolume; ++i) {
      inputStateValue[i] = NumericalType{1.0f / float(i+1), -1.0f / float(i+2)}; // just some value
    }
  } else if constexpr (std::is_same_v<NumericalType, std::complex<double>>) {
    for (std::size_t i = 0; i < stateVolume; ++i) {
      inputStateValue[i] = NumericalType{1.0 / double(i+1), -1.0 / double(i+2)}; // just some value
    }
  } else {
    std::cerr << "Error: Unsupported data type!\n";
    std::exit(1);
  }
  // Allocate initialized GPU storage for the input quantum state with prepared values
  auto * inputStateElems = createInitializeArrayGPU(inputStateValue);
  if (verbose)
    std::cout << "Allocated input quantum state storage and initialized it to some value\n";

  // Attach initialized GPU storage to the input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      inputState,
                      1,                                                 // only one storage component (tensor)
                      std::vector<void*>({inputStateElems}).data(),      // pointer to the GPU storage for the quantum state
                      std::vector<std::size_t>({storageSize}).data()));  // size of the GPU storage for the quantum state
  if (verbose)
    std::cout << "Constructed input quantum state\n";

  // Declare the output quantum state of the same shape
  cudensitymatState_t outputState;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(handle,
                      CUDENSITYMAT_STATE_PURITY_MIXED,  // pure (state vector) or mixed (density matrix) state
                      spaceShape.size(),
                      spaceShape.data(),
                      batchSize,
                      dataType,
                      &outputState));

  // Allocate GPU storage for the output quantum state
  auto * outputStateElems = createArrayGPU<NumericalType>(stateVolume);
  if (verbose)
    std::cout << "Allocated output quantum state storage\n";

  // Attach GPU storage to the output quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      outputState,
                      1,                                                 // only one storage component (tensor)
                      std::vector<void*>({outputStateElems}).data(),     // pointer to the GPU storage for the quantum state
                      std::vector<std::size_t>({storageSize}).data()));  // size of the GPU storage for the quantum state
  if (verbose)
    std::cout << "Constructed output quantum state\n";

  // Declare the adjoint input quantum state of the same shape
  cudensitymatState_t inputStateAdj;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(handle,
                      CUDENSITYMAT_STATE_PURITY_MIXED,  // pure (state vector) or mixed (density matrix) state
                      spaceShape.size(),
                      spaceShape.data(),
                      batchSize,
                      dataType,  // data type must match that of the operators created above
                      &inputStateAdj));

  // Allocate GPU storage for the adjoint input quantum state
  auto * inputStateAdjElems = createArrayGPU<NumericalType>(stateVolume);
  if (verbose)
    std::cout << "Allocated adjoint input quantum state storage\n";

  // Attach GPU storage to the adjoint input quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      inputStateAdj,
                      1,                                                 // only one storage component (tensor)
                      std::vector<void*>({inputStateAdjElems}).data(),   // pointer to the GPU storage for the quantum state
                      std::vector<std::size_t>({storageSize}).data()));  // size of the GPU storage for the quantum state
  if (verbose)
    std::cout << "Constructed adjoint input quantum state\n";

  // Declare a workspace descriptor
  cudensitymatWorkspaceDescriptor_t workspaceDescr;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle, &workspaceDescr));

  // Query free GPU memory
  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.45); // take 45% of the free memory for the workspace buffer
  if (verbose)
    std::cout << "Max workspace buffer size (bytes) = " << freeMem << std::endl;

  // Allocate GPU storage for the workspace buffer
  const std::size_t bufferVolume = freeMem / sizeof(NumericalType);
  auto * workspaceBuffer = createArrayGPU<NumericalType>(bufferVolume);
  if (verbose)
    std::cout << "Allocated workspace buffer of size (bytes) = " << freeMem << std::endl;

  // Prepare the Liouvillian operator action on a quantum state (needs to be done only once)
  auto startTime = std::chrono::high_resolution_clock::now();
  HANDLE_CUDM_ERROR(cudensitymatOperatorPrepareAction(handle,
                      liouvillian.get(),
                      inputState,
                      outputState,
                      CUDENSITYMAT_COMPUTE_64F,  // GPU compute type
                      freeMem,                   // max available GPU free memory for the workspace
                      workspaceDescr,            // workspace descriptor
                      0x0));                     // default CUDA stream
  auto finishTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> timeSec = finishTime - startTime;
  if (verbose)
    std::cout << "Operator action preparation time (sec) = " << timeSec.count() << std::endl;

  // Query the required workspace buffer size (bytes)
  std::size_t requiredBufferSize {0};
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(handle,
                      workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      &requiredBufferSize));
  if (verbose)
    std::cout << "Required workspace buffer size (bytes) = " << requiredBufferSize << std::endl;

  if (requiredBufferSize > freeMem) {
    std::cerr << "Error: Required workspace buffer size is greater than the available GPU free memory!\n";
    std::exit(1);
  }

  // Attach the workspace buffer to the workspace descriptor
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(handle,
                      workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      workspaceBuffer,
                      requiredBufferSize));
  if (verbose)
    std::cout << "Attached workspace buffer of size (bytes) = " << requiredBufferSize << std::endl;

  // Apply the Liouvillian operator to the input quatum state
  // and accumulate its action into the output quantum state (note the accumulative += semantics)
  for (int32_t repeat = 0; repeat < NUM_REPEATS; ++repeat) { // repeat multiple times for accurate timing
    // Zero out the output quantum state
    HANDLE_CUDM_ERROR(cudensitymatStateInitializeZero(handle,
                        outputState,
                        0x0));
    if (verbose)
      std::cout << "Initialized the output quantum state to zero\n";
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    startTime = std::chrono::high_resolution_clock::now();
    HANDLE_CUDM_ERROR(cudensitymatOperatorComputeAction(handle,
                        liouvillian.get(),
                        0.3,                // time point (some value)
                        batchSize,          // user-defined batch size
                        numParams,          // number of external user-defined Hamiltonian parameters
                        hamiltonianParams,  // external Hamiltonian parameters in GPU memory
                        inputState,         // input quantum state
                        outputState,        // output quantum state
                        workspaceDescr,     // workspace descriptor
                        0x0));              // default CUDA stream
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    finishTime = std::chrono::high_resolution_clock::now();
    timeSec = finishTime - startTime;
    if (verbose)
      std::cout << "Operator action computation time (sec) = " << timeSec.count() << std::endl;
  }

  // Compute the squared norm of the output quantum state
  void * norm2 = createInitializeArrayGPU(std::vector<double>(batchSize, 0.0));
  HANDLE_CUDM_ERROR(cudensitymatStateComputeNorm(handle,
                      outputState,
                      norm2,
                      0x0));
  if (verbose) {
    std::cout << "Computed the output quantum state norm:\n";
    printArrayGPU<double>(norm2, batchSize);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Prepare the Liouvillian operator action backward differentiation (needs to be done only once)
  startTime = std::chrono::high_resolution_clock::now();
  HANDLE_CUDM_ERROR(cudensitymatOperatorPrepareActionBackwardDiff(handle,
                      liouvillian.get(),
                      inputState,
                      outputState,               // adjoint output quantum state is always congruent to the output quantum state
                      CUDENSITYMAT_COMPUTE_64F,  // GPU compute type
                      freeMem,                   // max available GPU free memory for the workspace buffer
                      workspaceDescr,            // workspace descriptor
                      0x0));                     // default CUDA stream
  finishTime = std::chrono::high_resolution_clock::now();
  timeSec = finishTime - startTime;
  if (verbose)
    std::cout << "Operator action backward differentiation preparation time (sec) = " << timeSec.count() << std::endl;

  // Query the required workspace buffer size (bytes)
  requiredBufferSize = 0;
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(handle,
                      workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      &requiredBufferSize));
  if (verbose)
    std::cout << "Required workspace buffer size (bytes) = " << requiredBufferSize << std::endl;

  if (requiredBufferSize > freeMem) {
    std::cerr << "Error: Required workspace buffer size is greater than the available GPU free memory!\n";
    std::exit(1);
  }

  // Attach the workspace buffer to the workspace descriptor
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(handle,
                      workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      workspaceBuffer,
                      requiredBufferSize));
  if (verbose)
    std::cout << "Attached workspace buffer of size (bytes) = " << requiredBufferSize << std::endl;

  // Liouvillian operator action backward differentiation:
  // The adjoint output quantum state, which is always congruent to the output quantum state,
  // depends on the user-defined cost function, so here we simply pass the previously computed output quantum state.
  // In real-life applications, the user will pass their adjoint output quantum state, computed for their cost function.
  for (int32_t repeat = 0; repeat < NUM_REPEATS; ++repeat) { // repeat multiple times for accurate timing
    // Zero out the adjoint input quantum state and gradients
    HANDLE_CUDM_ERROR(cudensitymatStateInitializeZero(handle,
                        inputStateAdj,
                        0x0));
    initializeArrayGPU(std::vector<double>(numParams * batchSize, 0.0), hamiltonianParamsGrad);
    if (verbose)
      std::cout << "Initialized the adjoint input quantum state and gradients to zero\n";
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    startTime = std::chrono::high_resolution_clock::now();
    HANDLE_CUDM_ERROR(cudensitymatOperatorComputeActionBackwardDiff(handle,
                        liouvillian.get(),
                        0.3,                    // time point (some value)
                        batchSize,              // user-defined batch size
                        numParams,              // number of external user-defined Hamiltonian parameters
                        hamiltonianParams,      // external Hamiltonian parameters in GPU memory
                        inputState,             // input quantum state
                        outputState,            // adjoint output quantum state (here we just pass the previously computed output quantum state for simplicity)
                        inputStateAdj,          // adjoint input quantum state
                        hamiltonianParamsGrad,  // partial gradients with respect to the user-defined real parameters
                        workspaceDescr,         // workspace descriptor
                        0x0));                  // default CUDA stream
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    finishTime = std::chrono::high_resolution_clock::now();
    timeSec = finishTime - startTime;
    if (verbose)
      std::cout << "Operator action backward differentiation computation time (sec) = " << timeSec.count() << std::endl;
  }

  // Compute the squared norm of the adjoint input quantum state
  initializeArrayGPU(std::vector<double>(batchSize, 0.0), norm2);
  HANDLE_CUDM_ERROR(cudensitymatStateComputeNorm(handle,
                      inputStateAdj,
                      norm2,
                      0x0));
  if (verbose) {
    std::cout << "Computed the adjoint input quantum state norm:\n";
    printArrayGPU<double>(norm2, batchSize);
    std::cout << "Hamiltonian parameters gradients:\n";
    printArrayGPU<double>(hamiltonianParamsGrad, numParams * batchSize);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Destroy the norm2 array
  destroyArrayGPU(norm2);

  // Destroy workspace descriptor
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspaceDescr));

  // Destroy workspace buffer storage
  destroyArrayGPU(workspaceBuffer);

  // Destroy quantum states
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(inputStateAdj));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(outputState));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(inputState));

  // Destroy quantum state storage
  destroyArrayGPU(inputStateAdjElems);
  destroyArrayGPU(outputStateElems);
  destroyArrayGPU(inputStateElems);

  // Destroy external Hamiltonian parameters
  destroyArrayGPU(static_cast<void *>(hamiltonianParamsGrad));
  destroyArrayGPU(static_cast<void *>(hamiltonianParams));

  if (verbose)
    std::cout << "Destroyed resources\n" << std::flush;
}


int main(int argc, char ** argv)
{
  // Assign a GPU to the process
  HANDLE_CUDA_ERROR(cudaSetDevice(0));
  if (verbose)
    std::cout << "Set active device\n";

  // Create a library handle
  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
  if (verbose)
    std::cout << "Created a library handle\n";

  // Run the example
  exampleWorkflow(handle);

  // Destroy the library handle
  HANDLE_CUDM_ERROR(cudensitymatDestroy(handle));
  if (verbose)
    std::cout << "Destroyed the library handle\n";

  HANDLE_CUDA_ERROR(cudaDeviceReset());

  // Done
  return 0;
}
