/* Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cudensitymat.h>  // cuDensityMat library header
#include "helpers.h"       // helper functions


// Transverse Ising Hamiltonian with double summation ordering and spin-operator fusion
#include "transverse_ising_full_fused.h"  // user-defined Liouvillian operator example

#include <cmath>
#include <complex>
#include <vector>
#include <chrono>
#include <iostream>
#include <cassert>


// Logging verbosity
bool verbose = true;


// Example workflow
void exampleWorkflow(cudensitymatHandle_t handle)
{
  // Define the composite Hilbert space shape and
  // quantum state batch size (number of individual quantum states in a batched simulation)
  const std::vector<int64_t> spaceShape({2,2,2,2,2,2,2,2,2,2}); // dimensions of quantum degrees of freedom
  const int64_t batchSize = 1;        // number of quantum states per batch (currently only 1 state per batch)
  const int32_t numEigenStates = 4;   // number of eigenstates to compute

  if (verbose) {
    std::cout << "Hilbert space rank = " << spaceShape.size() << "; Shape = (";
    for (const auto & dimsn: spaceShape)
      std::cout << dimsn << ",";
    std::cout << ")" << std::endl;
    std::cout << "Quantum state batch size = " << batchSize << std::endl;
  }

  // Construct a user-defined Liouvillian operator using a convenience C++ class
  UserDefinedLiouvillian liouvillian(handle, spaceShape, batchSize);
  if (verbose)
    std::cout << "Constructed the Liouvillian operator\n";

  // Create quantum states to store the eigenstates
  std::size_t stateVolume {0};
  std::vector<cudensitymatState_t> eigenStates(numEigenStates);
  std::vector<void *> eigenStatesElems(numEigenStates);
  for (int32_t id = 0; id < numEigenStates; ++id) {

    // Declare the quantum state
    HANDLE_CUDM_ERROR(cudensitymatCreateState(handle,
                        CUDENSITYMAT_STATE_PURITY_PURE,  // pure (state vector)
                        spaceShape.size(),
                        spaceShape.data(),
                        batchSize,
                        dataType,
                        &eigenStates[id]));

    // Query the size of the quantum state storage
    std::size_t storageSize {0}; // only one storage component (tensor) is needed (no tensor factorization)
    HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(handle,
                        eigenStates[id],
                        1,               // only one storage component (tensor)
                        &storageSize));  // storage size in bytes
    stateVolume = storageSize / sizeof(NumericalType);  // quantum state tensor volume (number of elements)
    if (verbose)
      std::cout << "Quantum state storage size (bytes) = " << storageSize << std::endl;

    // Prepare some initial value for the quantum state
    std::vector<NumericalType> stateValue(stateVolume);
    if constexpr (std::is_same_v<NumericalType, double>) {
      for (std::size_t i = 0; i < stateVolume; ++i) {
        stateValue[i] = 1.0 / double(id*5 + i+1); // just some value
      }
    } else if constexpr (std::is_same_v<NumericalType, std::complex<double>>) {
      for (std::size_t i = 0; i < stateVolume; ++i) {
        stateValue[i] = NumericalType{1.0 / double(id*5 + i+1), -1.0 / double(id*3 + i+2)}; // just some value
      }
    } else {
      std::cerr << "Error: Unsupported data type!\n";
      std::exit(1);
    }
    // Allocate initialized GPU storage for the quantum state with prepared values
    eigenStatesElems[id] = createInitializeArrayGPU(stateValue);
    if (verbose)
      std::cout << "Allocated quantum state storage and initialized it to some value\n";

    // Attach initialized GPU storage to the quantum state
    HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                        eigenStates[id],
                        1,                                                 // only one storage component (tensor)
                        std::vector<void*>({eigenStatesElems[id]}).data(), // pointer to the GPU storage for the quantum state
                        std::vector<std::size_t>({storageSize}).data()));  // size of the GPU storage for the quantum state
    if (verbose)
      std::cout << "Constructed quantum state\n";
  }

  // Allocate storage for the eigenvalues and convergence tolerances
  void * eigenvalues = createArrayGPU<NumericalType>(numEigenStates * batchSize);
  std::vector<double> tolerances(numEigenStates * batchSize, 1e-6);

  // Declare a workspace descriptor
  cudensitymatWorkspaceDescriptor_t workspaceDescr;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle, &workspaceDescr));

  // Query free GPU memory
  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.95); // take 95% of the free memory for the workspace buffer
  if (verbose)
    std::cout << "Max workspace buffer size (bytes) = " << freeMem << std::endl;

  // Create the operator eigenspectrum computation object
  cudensitymatOperatorSpectrum_t spectrum;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorSpectrum(handle,
                      liouvillian.get(),
                      1,  // Hermitian operator
                      CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL,
                      &spectrum));

  // Prepare the operator eigenspectrum computation (needs to be done only once)
  auto startTime = std::chrono::high_resolution_clock::now();
  HANDLE_CUDM_ERROR(cudensitymatOperatorSpectrumPrepare(handle,
                      spectrum,
                      numEigenStates,
                      eigenStates[0],
                      CUDENSITYMAT_COMPUTE_64F,
                      freeMem,
                      workspaceDescr,
                      0x0));
  auto finishTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> timeSec = finishTime - startTime;
  if (verbose)
    std::cout << "Operator eigenspectrum preparation time (sec) = " << timeSec.count() << std::endl;

  // Query the required workspace buffer size (bytes)
  std::size_t requiredBufferSize {0};
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(handle,
                      workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      &requiredBufferSize));
  if (verbose)
    std::cout << "Required workspace buffer size (bytes) = " << requiredBufferSize << std::endl;

  // Allocate GPU storage for the workspace buffer
  const std::size_t bufferVolume = requiredBufferSize / sizeof(NumericalType);
  auto * workspaceBuffer = createArrayGPU<NumericalType>(bufferVolume);
  if (verbose)
    std::cout << "Allocated workspace buffer of size (bytes) = " << requiredBufferSize << std::endl;

  // Attach the workspace buffer to the workspace descriptor
  HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(handle,
                      workspaceDescr,
                      CUDENSITYMAT_MEMSPACE_DEVICE,
                      CUDENSITYMAT_WORKSPACE_SCRATCH,
                      workspaceBuffer,
                      requiredBufferSize));
  if (verbose)
    std::cout << "Attached workspace buffer of size (bytes) = " << requiredBufferSize << std::endl;

  // Compute the operator eigenspectrum
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  startTime = std::chrono::high_resolution_clock::now();
  HANDLE_CUDM_ERROR(cudensitymatOperatorSpectrumCompute(handle,
                      spectrum,
                      0.0,
                      batchSize,
                      0,
                      nullptr,
                      numEigenStates,
                      eigenStates.data(),
                      eigenvalues,
                      tolerances.data(),
                      workspaceDescr,
                      0x0));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  finishTime = std::chrono::high_resolution_clock::now();
  timeSec = finishTime - startTime;
  if (verbose)
    std::cout << "Operator eigenspectrum computation time (sec) = " << timeSec.count() << std::endl;

  // Print the eigenvalues
  if (verbose) {
    std::cout << "Eigenvalues:\n";
    printArrayGPU<NumericalType>(eigenvalues, numEigenStates);
  }

  // Print the residual norms
  if (verbose) {
    std::cout << "Residual norms:\n";
    printArrayCPU<double>(tolerances.data(), numEigenStates);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Destroy workspace descriptor
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspaceDescr));

  // Destroy workspace buffer storage
  destroyArrayGPU(workspaceBuffer);

  // Destroy operator eigenspectrum computation object
  HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorSpectrum(spectrum));

  // Destroy eigenvalues storage
  destroyArrayGPU(eigenvalues);

  // Destroy quantum states
  for (int32_t id = 0; id < numEigenStates; ++id)
    HANDLE_CUDM_ERROR(cudensitymatDestroyState(eigenStates[id]));

  // Destroy quantum state storage
  for (int32_t id = 0; id < numEigenStates; ++id)
    destroyArrayGPU(eigenStatesElems[id]);

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
