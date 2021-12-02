# cuTensorNet - Samples

* [Documentation](https://docs.nvidia.com/cuda/cutensornet/index.html)

# Install

## Linux 

You can use make to compile the cutensornet samples. The option CUTENSORNET_ROOT need to be defined if cuTensorNet is not the CUDA installation folder. 

With make

```
export CUTENSORNET_ROOT=<path_to_custatevec_root>
make -j8
```

# Support

* **Supported SM Architectures:** SM 7.0, SM 7.5, SM 8.0, SM 8.6
* **Supported OSes:** Linux
* **Supported CPU Architectures**: x86_64, arm64
* **Language**: `C++11`

# Prerequisites

* [CUDA 1X.X toolkit](https://developer.nvidia.com/cuda-downloads) (or above) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).

# Description
This sample help users get familiar with cuTensorNet. 
It provide an example of calling cuTensorNet to find a contraction path as well as to performs the contraction.
The sample consists of:
* Defining a Tensor Network (Create Contraction Descriptor using "cutensornetCreateNetworkDescriptor")
* Find a close to "optimal" order of contraction (here user call "cutensornetContractionOptimize" to find the contraction order. User can control some parameters of the cutensornetContractionOptimize (e.g., path finder) using the "cutensornetContractionOptimizerConfigSetAttribute" function. User also can provide their own path and use the SetAttribute tool to set the Info structure to their own path.
* Create a planning to performs the contraction using "cutensornetCreateContractionPlan". This step will prepare a planning for the execution of list of the pairwise contractions provided by the path.
* User can optionally call "cutensornetContractionAutotune" to performs an autotuning and choose the best performing kernel for the corresponding path such as the winner kernels will be called for all subsequent calls to performs the contraction "cutensornetContraction". The autotuning could bring improvement in particular when "cutensornetContraction" is called more than numAutotuningIterations times.
* Performs the computation of the contraction using "cutensornetContraction"
