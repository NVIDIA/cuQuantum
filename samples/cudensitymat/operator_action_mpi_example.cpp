/* Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cudensitymat.h>  // cuDensityMat library header
#include "helpers.h"       // helper functions


// Transverse Ising Hamiltonian with double summation ordering
// and spin-operator fusion, plus fused dissipation terms
#include "transverse_ising_full_fused_noisy.h"  // user-defined Liouvillian operator example


// MPI library (optional)
#ifdef MPI_ENABLED
#include <mpi.h>
#endif

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
  // quantum state batch size (number of individual quantum states)
  const std::vector<int64_t> spaceShape({2,2,2,2,2,2,2,2}); // dimensions of quantum degrees of freedom
  const int64_t batchSize = 1;                              // number of quantum states per batch (default is 1)

  if (verbose) {
    std::cout << "Hilbert space rank = " << spaceShape.size() << "; Shape = (";
    for (const auto & dimsn: spaceShape)
      std::cout << dimsn << ",";
    std::cout << ")" << std::endl;
    std::cout << "Quantum state batch size = " << batchSize << std::endl;
  }

  // Construct a user-defined Liouvillian operator using a convenience C++ class
  UserDefinedLiouvillian liouvillian(handle, spaceShape);
  if (verbose)
    std::cout << "Constructed the Liouvillian operator\n";

  // Declare the input quantum state
  cudensitymatState_t inputState;
  HANDLE_CUDM_ERROR(cudensitymatCreateState(handle,
                      CUDENSITYMAT_STATE_PURITY_MIXED,  // pure (state vector) or mixed (density matrix) state
                      spaceShape.size(),
                      spaceShape.data(),
                      batchSize,
                      CUDA_C_64F,  // data type must match that of the operators created above
                      &inputState));

  // Query the size of the quantum state storage
  std::size_t storageSize {0}; // only one storage component (tensor) is needed
  HANDLE_CUDM_ERROR(cudensitymatStateGetComponentStorageSize(handle,
                      inputState,
                      1,               // only one storage component
                      &storageSize));  // storage size in bytes
  const std::size_t stateVolume = storageSize / sizeof(std::complex<double>);  // quantum state tensor volume (number of elements)
  if (verbose)
    std::cout << "Quantum state storage size (bytes) = " << storageSize << std::endl;

  // Prepare some initial value for the input quantum state
  std::vector<std::complex<double>> inputStateValue(stateVolume);
  for (std::size_t i = 0; i < stateVolume; ++i) {
    inputStateValue[i] = std::complex<double>{double(i+1), double(-(i+2))}; // just some value
  }

  // Allocate initialized GPU storage for the input quantum state with prepared values
  auto * inputStateElems = createArrayGPU(inputStateValue);

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
                      CUDA_C_64F,  // data type must match that of the operators created above
                      &outputState));

  // Allocate initialized GPU storage for the output quantum state
  auto * outputStateElems = createArrayGPU(std::vector<std::complex<double>>(stateVolume, {0.0, 0.0}));

  // Attach initialized GPU storage to the output quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateAttachComponentStorage(handle,
                      outputState,
                      1,                                                 // only one storage component (no tensor factorization)
                      std::vector<void*>({outputStateElems}).data(),     // pointer to the GPU storage for the quantum state
                      std::vector<std::size_t>({storageSize}).data()));  // size of the GPU storage for the quantum state
  if (verbose)
    std::cout << "Constructed output quantum state\n";

  // Declare a workspace descriptor
  cudensitymatWorkspaceDescriptor_t workspaceDescr;
  HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(handle, &workspaceDescr));

  // Query free GPU memory
  std::size_t freeMem = 0, totalMem = 0;
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
  freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.95); // take 95% of the free memory for the workspace buffer
  if (verbose)
    std::cout << "Max workspace buffer size (bytes) = " << freeMem << std::endl;

  // Prepare the Liouvillian operator action on a quantum state (needs to be done only once)
  const auto startTime = std::chrono::high_resolution_clock::now();
  HANDLE_CUDM_ERROR(cudensitymatOperatorPrepareAction(handle,
                      liouvillian.get(),
                      inputState,
                      outputState,
                      CUDENSITYMAT_COMPUTE_64F,  // GPU compute type
                      freeMem,                   // max available GPU free memory for the workspace
                      workspaceDescr,            // workspace descriptor
                      0x0));                     // default CUDA stream
  const auto finishTime = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> timeSec = finishTime - startTime;
  if (verbose)
    std::cout << "Operator action prepation time (sec) = " << timeSec.count() << std::endl;

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
  const std::size_t bufferVolume = requiredBufferSize / sizeof(std::complex<double>);
  auto * workspaceBuffer = createArrayGPU(std::vector<std::complex<double>>(bufferVolume, {0.0, 0.0}));
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

  // Zero out the output quantum state
  HANDLE_CUDM_ERROR(cudensitymatStateInitializeZero(handle,
                      outputState,
                      0x0));
  if (verbose)
    std::cout << "Initialized the output state to zero\n";

  // Apply the Liouvillian operator to the input quatum state
  // and accumulate its action into the output quantum state (note += semantics)
  for (int32_t repeat = 0; repeat < NUM_REPEATS; ++repeat) { // repeat multiple times for accurate timing
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    const auto startTime = std::chrono::high_resolution_clock::now();
    HANDLE_CUDM_ERROR(cudensitymatOperatorComputeAction(handle,
                        liouvillian.get(),
                        0.01,                                  // time point
                        1,                                     // number of external user-defined Hamiltonian parameters
                        std::vector<double>({13.42}).data(),   // Hamiltonian parameter(s)
                        inputState,                            // input quantum state
                        outputState,                           // output quantum state
                        workspaceDescr,                        // workspace descriptor
                        0x0));                                 // default CUDA stream
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    const auto finishTime = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> timeSec = finishTime - startTime;
    if (verbose)
      std::cout << "Operator action computation time (sec) = " << timeSec.count() << std::endl;
  }

  // Compute the squared norm of the output quantum state
  void * norm2 = createArrayGPU(std::vector<double>(batchSize, 0.0));
  HANDLE_CUDM_ERROR(cudensitymatStateComputeNorm(handle,
                      outputState,
                      norm2,
                      0x0));
  if (verbose)
    std::cout << "Computed the output state norm\n";
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  destroyArrayGPU(norm2);

  // Destroy workspace descriptor
  HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspaceDescr));

  // Destroy workspace buffer storage
  destroyArrayGPU(workspaceBuffer);

  // Destroy quantum states
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(outputState));
  HANDLE_CUDM_ERROR(cudensitymatDestroyState(inputState));

  // Destroy quantum state storage
  destroyArrayGPU(outputStateElems);
  destroyArrayGPU(inputStateElems);

  if (verbose)
    std::cout << "Destroyed resources\n" << std::flush;
}


int main(int argc, char ** argv)
{
  // Initialize MPI library (if needed)
#ifdef MPI_ENABLED
  HANDLE_MPI_ERROR(MPI_Init(&argc, &argv));
  int procRank {-1};
  HANDLE_MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &procRank));
  int numProcs {0};
  HANDLE_MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &numProcs));
  if (procRank != 0) verbose = false;
  if (verbose)
    std::cout << "Initialized MPI library\n";
#else
  const int procRank {0};
  const int numProcs {1};
#endif

  // Assign a GPU to the process
  int numDevices {0};
  HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
  const int deviceId = procRank % numDevices;
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  if (verbose)
    std::cout << "Set active device\n";

  // Create a library handle
  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
  if (verbose)
    std::cout << "Created a library handle\n";

  // Reset distributed configuration (once)
#ifdef MPI_ENABLED
  MPI_Comm comm;
  HANDLE_MPI_ERROR(MPI_Comm_dup(MPI_COMM_WORLD, &comm));
  HANDLE_CUDM_ERROR(cudensitymatResetDistributedConfiguration(handle,
                      CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI,
                      &comm, sizeof(comm)));
#endif

  // Run the example
  exampleWorkflow(handle);

  // Synchronize MPI processes
#ifdef MPI_ENABLED
  HANDLE_MPI_ERROR(MPI_Barrier(MPI_COMM_WORLD));
#endif

  // Destroy the library handle
  HANDLE_CUDM_ERROR(cudensitymatDestroy(handle));
  if (verbose)
    std::cout << "Destroyed the library handle\n";

  // Finalize the MPI library
#ifdef MPI_ENABLED
  HANDLE_MPI_ERROR(MPI_Finalize());
  if (verbose)
    std::cout << "Finalized MPI library\n";
#endif

  // Done
  return 0;
}
