# cuTensorNet - Samples

* [Documentation](https://docs.nvidia.com/cuda/cutensornet/index.html)

## Install

### Linux

You can use make to compile the cuTensorNet samples. The environment variables `CUTENSOR_ROOT` and `CUTENSORNET_ROOT` need to be defined if cuTENSOR and cuTensorNet is not the CUDA installation folder.

Using `make`:
```
export CUTENSOR_ROOT=<path_to_cutensor_root>
export CUTENSORNET_ROOT=<path_to_cutensornet_root>
make -j8
```
or `cmake`:
```
export CUTENSOR_ROOT=<path_to_cutensor_root>
export CUTENSORNET_ROOT=<path_to_cutensornet_root>
cmake . && make
```

## Run

To execute the sample, simply run:
```
./tensornet_example
```

## Support

* **Supported SM Architectures:** SM 7.0, SM 7.5, SM 8.0, SM 8.6
* **Supported OSes:** Linux
* **Supported CPU Architectures**: x86_64, aarch64-sbsa, ppc64le
* **Language**: C++11 or above

## Prerequisites

* [CUDA Toolkit 11.x](https://developer.nvidia.com/cuda-downloads) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).

## Description
This sample helps users get familiar with cuTensorNet. It provides an example of calling cuTensorNet to find a contraction path and as well as performing the contraction.

The sample consists of:
* Define a tensor network using `cutensornetCreateNetworkDescriptor`.
* Find a close-to-optimal order of contraction (i.e., a contraction path) via `cutensornetContractionOptimize`. Users can control the parameters of `cutensornetContractionOptimize` (e.g., the pathfinder) using the `cutensornetContractionOptimizerConfigSetAttribute` function. Users also can provide their own path and use the `cutensornetContractionOptimizerInfoSetAttribute` API to set the `cutensornetContractionOptimizerInfo_t` structure to their own path.
* Create a contraction plan for performing the contraction using `cutensornetCreateContractionPlan`. This step will prepare a plan for the execution of a list of the pairwise contractions provided by the path.
* Optionally, call `cutensornetContractionAutotune` to perform autotuning, which chooses the best performant kernels for the corresponding path such that the winner kernels will be called for all subsequent calls of `cutensornetContraction` to perform the contraction of the tensor network. The autotuning could bring improvement in particular when `cutensornetContraction` is called multiple times with the same plan/network.
* Perform the computation of the contraction using `cutensornetContraction`.
* Free the cuTensorNet resources.
