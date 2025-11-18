# cuDensityMat - Samples

* [Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html)

## Install

### Linux

You can use `make` or `cmake` to compile the cuDensityMat samples. The environment variables `CUDA_PATH`, `CUTENSOR_ROOT` and `CUDENSITYMAT_ROOT` need to be defined to point to the CUDA Toolkit, cuTensor and cuDensityMat locations, respectively. Additionally, the environment variable `MPI_ROOT` needs to point to your MPI installation if you use `make`.

Using `make`:
```
export CUDA_PATH=<path_to_cuda_root>
export CUTENSOR_ROOT=<path_to_cutensor_root>
export CUDENSITYMAT_ROOT=<path_to_cudensitymat_root>
export MPI_ROOT=<path_to_mpi_root>
make
```
or `cmake`:
```
export CUDA_PATH=<path_to_cuda_root>
export CUTENSOR_ROOT=<path_to_cutensor_root>
export CUDENSITYMAT_ROOT=<path_to_cudensitymat_root>
cmake . && make
```

## Run

To execute the single-GPU sample in a command shell, simply use:
```
./operator_action_example
```
To execute the multi-GPU MPI sample with automatic MPI parallelization, run:
```
mpiexec -n N ./operator_action_mpi_example
```
where `N` is the desired number of MPI processes, which must be a power of two.
You will need to define the environment variable CUDENSITYMAT_COMM_LIB as described
in the Getting Started section of the cuDensityMat library documentation.

**Note**: Depending on how CUDA Toolkit and cuTensor are installed,
you might need to add them to `LD_LIBRARY_PATH` like this:
```
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUTENSOR_ROOT/lib/12:$LD_LIBRARY_PATH
```
The cuTENSOR library path would depend on the CUDA major version. Please refer
to the [Getting Started](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/getting_started.html)
page for further detail.

## Support

* **Supported SM Architectures:** SM 7.0, SM 7.5, SM 8.0, SM 8.6, SM 9.0, SM 10.0, SM 12.0
* **Supported OSes:** Linux
* **Supported CPU Architectures**: x86_64, aarch64-sbsa
* **Language**: C++11 or above

## Prerequisites

* [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) or higher and compatible driver
(see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* cuTENSOR 2.3.1+.
* CMake 3.22+ if using `cmake`.
