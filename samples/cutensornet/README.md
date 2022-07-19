# cuTensorNet - Samples

* [Documentation](https://docs.nvidia.com/cuda/cuquantum/cutensornet/index.html)

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
To execute the parallel MPI sample, run:
```
mpiexec -n N ./tensornet_example_mpi
```
where `N` is the desired number of processes. In this example, `N` can be larger than the number of GPUs in your system.

## Support

* **Supported SM Architectures:** SM 7.0, SM 7.5, SM 8.0, SM 8.6
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

### 2. Parallel execution (`tensornet_example_mpi.cu`)

The parallel MPI sample illustrates advanced usage of cuTensorNet. Specifically, it demonstrates how to find a contraction path in parallel and how to exploit slice-based parallelism by contracting a subset of slices on each process.

This sample consists of:
* A basic skeleton setting up a simple MPI+CUDA computation using a one GPU per process model.
* Setting a common workspace limit in all participating processes.
* Finding an optimal path with `cutensornetContractionOptimize` in parallel, and using global reduction (`MPI_MINLOC`) to find the best path and the owning process's identity. Note that the contraction optimizer on each process sets a different random seed, so each process typically computes a different optimal path for sufficiently large tensor networks.
* Broadcasting the winner's `optimizerInfo` object by serializing it using the `cutensornetContractionOptimizerInfoGetPackedSize` and `cutensornetContractionOptimizerInfoPackData` APIs, and deserializing it into an existing `optimizerInfo` object using the `cutensornetUpdateContractionOptimizerInfoFromPackedData` API.
* Computing the subset of slice IDs (in a relatively load-balanced fashion) for which each process is responsible, contracting them, and performing a global reduction (sum) to get the final result on the root process.
