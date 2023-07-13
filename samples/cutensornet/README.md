# cuTensorNet - Samples

* [Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html)

## Install

### Linux

You can use `make` to compile the cuTensorNet samples. The environment variables `CUDA_PATH`, `CUTENSOR_ROOT` and `CUTENSORNET_ROOT` need to be defined to point to the CUDA Toolkit, cuTENSOR and cuTensorNet location, respectively. Additionally, the environment variable `MPI_ROOT` needs to point to your MPI installation if you use `make`.

Using `make`:
```
export CUDA_PATH=<path_to_cuda_root>
export CUTENSOR_ROOT=<path_to_cutensor_root>
export CUTENSORNET_ROOT=<path_to_cutensornet_root>
export MPI_ROOT=<path_to_mpi_root>
make -j
```
or `cmake`:
```
export CUDA_PATH=<path_to_cuda_root>
export CUTENSOR_ROOT=<path_to_cutensor_root>
export CUTENSORNET_ROOT=<path_to_cutensornet_root>
cmake . && make -j
```

## Run

To execute the serial sample in a command shell, simply use:
```
./tensornet_example
```
To execute the parallel MPI sample with automatic MPI parallelization, run:
```
mpiexec -n N ./tensornet_example_mpi_auto
```
where `N` is the desired number of processes. You will need to define
the environment variable CUTENSORNET_COMM_LIB as described in the Getting Started
section of the cuTensorNet library documentation (Installation and Compilation).

To execute the parallel MPI sample with explicit MPI parallelization, run:
```
mpiexec -n N ./tensornet_example_mpi
```
where `N` is the desired number of processes. In this example, `N` can be larger than the number of GPUs in your system.

The tensor SVD sample can be easily executed in a command shell using:
```
./tensor_svd_example
```
The sample for tensor QR, gate split and MPS can also be executed in the same fashion.

**Note**: Depending on how CUDA Toolkit and cuTENSOR are installed, you might need to add them to `LD_LIBRARY_PATH` like this:
```
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUTENSOR_ROOT/lib/11:$LD_LIBRARY_PATH
```
The cuTENSOR library path would depend on the CUDA major version. Please refer to the [Getting Started](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/getting_started.html) page for further detail.

## Support

* **Supported SM Architectures:** SM 7.0, SM 7.5, SM 8.0, SM 8.6, SM 9.0
* **Supported OSes:** Linux
* **Supported CPU Architectures**: x86_64, aarch64-sbsa, ppc64le
* **Language**: C++11 or above

## Prerequisites

* [CUDA Toolkit 11.x](https://developer.nvidia.com/cuda-downloads) and compatible driver r450+ (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* cuTENSOR 1.5.0+.
* GNU OpenMP (GOMP) runtime.
* CMake 3.17+ if using `cmake`.

## Description

### 1. Serial execution (`tensornet_example.cu`)

The serial sample helps users get familiar with cuTensorNet. It provides an example of calling cuTensorNet to find a contraction path as well as performing the contraction.

This sample consists of:
* Defining a tensor network using `cutensornetCreateNetworkDescriptor`.
* Finding a close-to-optimal order of contraction (i.e., a contraction path) via `cutensornetContractionOptimize`. Users can control the parameters of `cutensornetContractionOptimize` (e.g., the pathfinder) using the `cutensornetContractionOptimizerConfigSetAttribute` function. Users can also provide their own path and use the `cutensornetContractionOptimizerInfoSetAttribute` API to set their own path to the `cutensornetContractionOptimizerInfo_t` object.
* Creating a contraction plan for performing the contraction using `cutensornetCreateContractionPlan`. This step will prepare a plan for the execution of a list of pairwise contractions provided by the path.
* Optionally, calling `cutensornetContractionAutotune` to perform autotuning, which chooses the best contraction kernels for the corresponding path. These kernels will be used by all subsequent `cutensornetContractSlices` calls to contract the tensor network. Autotuning is usually beneficial when `cutensornetContractSlices` is called multiple times on the same plan/network.
* Performing the computation of the contraction using `cutensornetContractSlices` for a group of slices (in this case, all of the slices) created (destroyed) using the `cutensornetCreateSliceGroupFromIDRange` (`cutensornetDestroySliceGroup`) API.
* Freeing the cuTensorNet resources.

### 2. Parallel execution (`tensornet_example_mpi_auto.cu`)

This parallel MPI sample enables automatic distributed parallelization across multiple/many GPUs.
Specifically, it demonstrates how to activate an automatic distributed parallelization inside
the cuTensorNet library such that it will find a contraction path and subsequently contract
the tensor network in parallel using exactly the same source code as in a serial (single-GPU) run.
Currently one will need a CUDA-aware MPI library implementation to run this sample. Please refer
to the Getting Started section of the cuTensorNet library documenation for full details.

This sample consists of:
* A basic skeleton setting up a simple MPI+CUDA computation using a one GPU per MPI process model.
* Activation call that enables automatic distributed parallelization inside the cuTensorNet library.
* Parallel execution of the tensor network contraction path finder (`cutensornetContractionOptimize`).
* Parallel execution of the tensor network contraction (`cutensornetContractSlices`).

### 3. Parallel execution via explicit MPI calls (`tensornet_example_mpi.cu`)

This parallel MPI sample illustrates advanced usage of cuTensorNet. Specifically, it demonstrates
how to find a contraction path in parallel and how to exploit slice-based parallelism by contracting
a subset of slices on each process using manual MPI instrumentation. Note that the previous parallel
sample will do all these for you automatically without any chages to the original (serial) source code.

This sample consists of:
* A basic skeleton setting up a simple MPI+CUDA computation using a one GPU per process model.
* Setting a common workspace limit in all participating processes.
* Finding an optimal path with `cutensornetContractionOptimize` in parallel, and using global reduction (`MPI_MINLOC`) to find the best path and the owning process's identity. Note that the contraction optimizer on each process sets a different random seed, so each process typically computes a different optimal path for sufficiently large tensor networks.
* Broadcasting the winner's `optimizerInfo` object by serializing it using the `cutensornetContractionOptimizerInfoGetPackedSize` and `cutensornetContractionOptimizerInfoPackData` APIs, and deserializing it into an existing `optimizerInfo` object using the `cutensornetUpdateContractionOptimizerInfoFromPackedData` API.
* Computing the subset of slice IDs (in a relatively load-balanced fashion) for which each process is responsible, contracting them, and performing a global reduction (sum) to get the final result on the root process.

### 4. Tensor QR (`approxTN/tensor_qr_example.cu`)

This sample demonstrates how to use cuTensorNet to perform tensor QR operation. 

This sample consists of:
* Defining input and output tensors using `cutensornetCreateTensorDescriptor`.
* Querying the required workspace for the computation using `cutensornetWorkspaceComputeQRSizes`. 
* Performing the computation of tensor QR using `cutensornetTensorQR`. 
* Freeing the cuTensorNet resources.

### 5. Tensor SVD (`approxTN/tensor_svd_example.cu`)

This sample demonstrates how to use cuTensorNet to perform tensor SVD operation. 

This sample consists of:
* Defining input and output tensors using `cutensornetCreateTensorDescriptor`. Fixed extent truncation can be directly specified by modifying the corresponding extent in the output tensor descriptor.
* Setting up the SVD truncation options using the `cutensornetTensorSVDConfigSetAttribute` function of the `svdConfig` object created by `cutensornetCreateTensorSVDConfig`.
* Optionally, calling `cutensornetCreateTensorSVDInfo` and `cutensornetTensorSVDInfoGetAttribute` to store and retrieve runtime SVD truncation information.
* Querying the required workspace for the computation using `cutensornetWorkspaceComputeSVDSizes`. 
* Performing the computation of tensor SVD using `cutensornetTensorSVD`. 
* Freeing the cuTensorNet resources.

### 6. Gate Split (`approxTN/gate_split_example.cu`)

This sample demonstrates how to use cuTensorNet to perform a single gate split operation. 

This sample consists of:
* Defining input and output tensors using `cutensornetCreateTensorDescriptor`. Fixed extent truncation can be directly specified by modifying the corresponding extent in the output tensor descriptor.
* Setting up the SVD truncation options using the `cutensornetTensorSVDConfigSetAttribute` function of the `svdConfig` object created by `cutensornetCreateTensorSVDConfig`.
* Optionally, calling `cutensornetCreateTensorSVDInfo` and `cutensornetTensorSVDInfoGetAttribute` to store and retrieve runtime SVD truncation information.
* Querying the required workspace for the computation using `cutensornetWorkspaceComputeGateSplitSizes`. The gate split algorithm is specified in `cutensornetGateSplitAlgo_t`. 
* Performing the computation of tensor SVD using `cutensornetTensorGateSplit`. 
* Freeing the cuTensorNet resources.

### 7. MPS (`approxTN/mps_example.cu`)

This sample demonstrates how to integrate cuTensorNet into matrix product states (MPS) simulator. 

This sample is based on an ``MPSHelper`` that can systematically manage the MPS metadata and cuTensorNet library objects. 
Following functionalities are encapsulated in this class:
* Dynamically updating the `cutensornetTensorDescriptor_t` for all MPS tensors by calling `cutensornetCreateTensorDescriptor` and `cutensornetDestroyTensorDescriptor`.
* Querying the maximal data size needed for each MPS tensor.
* Setting up the SVD truncation options using the `cutensornetTensorSVDConfigSetAttribute` function of the `svdConfig` object created by `cutensornetCreateTensorSVDConfig`.
* Querying the required workspace size for all gate split operations by calling `cutensornetWorkspaceComputeGateSplitSizes` on the largest problem.
* Optionally, calling `cutensornetCreateTensorSVDInfo` and `cutensornetTensorSVDInfoGetAttribute` to store and retrieve runtime SVD truncation information.
* Performing gate split operations for all gates using `cutensornetTensorGateSplit`. 
* Freeing the cuTensorNet resources.

### 8. Intermediate tensor(s) reuse (`tensornet_example_reuse.cu`)

This sample demonstrates how to use the "intermediate tensor reuse" feature to accelerate the contractions
of a network with constant intput tensors, where repeated contractions would change some of the input tensor's data only. This sample largely builds on first sample provided above.

This sample demonstrates how to:
* Mark input tensors as "constant" when creating a tensor network using `cutensornetCreateNetworkDescriptor`, by setting the corresponding `cutensornetTensorQualifiers_t` field.
* Provide a cache workspace to the contraction plan which will be used to accelerate the subsequent contractions of the same network. It shows how to query the required cache memory size using `cutensornetWorkspaceGetMemorySize` with a `CUTENSORNET_WORKSPACE_CACHE` workspace-kind, and how to the provide the workspace memory using `cutensornetWorkspaceSetMemory`.
* Provide a predefined contraction path to the contraction optimizer by calling `cutensornetContractionOptimizerInfoSetAttribute` with `CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH` attribute.

### 9. Gradient via back-propagation (`tensornet_example_gradients.cu`)

This sample demonstrates how to perform back-propagation to compute gradients of a tensor-network w.r.t. select input tensors.
This sample largely builds on first sample provided above.

This sample demonstrates how to:
* Mark input tensors for gradient computation when creating a tensor network by calling `cutensornetNetworkSetAttribute`.
* Provide a cache workspace to the contraction plan which will be used to hold intermediate data needed for gradient computation. It shows how to query the required cache memory size using `cutensornetWorkspaceGetMemorySize` with a `CUTENSORNET_WORKSPACE_CACHE` workspace-kind, and how to the provide the workspace memory using `cutensornetWorkspaceSetMemory`.
* Call `cutensornetComputeGradientsBackward` to perform the gradient computation.
* Call `cutensornetWorkspacePurgeCache` to clean up the cache and prepare for the next gradient calculation.
