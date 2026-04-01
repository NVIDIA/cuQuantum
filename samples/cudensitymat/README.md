# cuDensityMat - Samples

* [Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html)

## Samples

| Sample | Description |
|--------|-------------|
| `operator_action_example` | Single-GPU operator action on a mixed quantum state |
| `operator_action_mpi_example` | Multi-GPU operator action using MPI |
| `operator_action_nccl_example` | Multi-GPU operator action using NCCL |
| `operator_action_gradient_example` | Single-GPU operator action with backward differentiation |
| `operator_action_batched_gradient_example` | Single-GPU batched operator action with backward differentiation |
| `operator_eigenspectrum_example` | Single-GPU operator eigen-spectrum computation |
| `mps_tdvp_example` | Single-GPU MPS TDVP time propagation |

## Install

### Linux

You can use `make` or `cmake` to compile the cuDensityMat samples.

#### Environment variables

The following environment variables must be set:

| Variable | Description | Required |
|----------|-------------|----------|
| `CUDA_PATH` | Path to the CUDA Toolkit | Always |
| `CUTENSOR_ROOT` | Path to the cuTENSOR installation | Always |
| `CUTENSORNET_ROOT` | Path to the cuTensorNet installation (or `CUQUANTUM_ROOT`) | Always |
| `CUDENSITYMAT_ROOT` | Path to the cuDensityMat installation (or `CUQUANTUM_ROOT`) | Always |
| `MPI_ROOT` | Path to a CUDA-aware MPI installation | For MPI/NCCL examples |
| `NCCL_ROOT` | Path to the NCCL installation | For NCCL example |

Set the environment variables to match your installation:

```
export CUDA_PATH=<path_to_cuda_root>
export CUTENSOR_ROOT=<path_to_cutensor_root>
export CUTENSORNET_ROOT=<path_to_cutensornet_root>
export CUDENSITYMAT_ROOT=<path_to_cudensitymat_root>
export MPI_ROOT=<path_to_mpi_root>       # optional, for MPI/NCCL examples
export NCCL_ROOT=<path_to_nccl_root>     # optional, for NCCL example
```

#### Using `make`

```
make
```

#### Using `cmake`

Replace `<arch>` below with your target GPU architecture (e.g., `80` for A100, `90` for H100, `100` for B100/B200).

Single-GPU examples only:
```
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=<arch>
make
```

With MPI support (builds MPI example):
```
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=<arch> -DENABLE_MPI=TRUE
make
```

With NCCL support (builds NCCL example; requires MPI for bootstrapping):
```
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=<arch> -DENABLE_MPI=TRUE -DENABLE_NCCL=TRUE
make
```

## Run

To execute the single-GPU samples, simply use:
```
./operator_action_example
./operator_action_gradient_example
./operator_action_batched_gradient_example
./operator_eigenspectrum_example
./mps_tdvp_example
```

To execute the multi-GPU MPI sample, run:
```
mpiexec -n N ./operator_action_mpi_example
```
where `N` is the desired number of MPI processes, which must be a power of two.
You will need to define the environment variable `CUDENSITYMAT_COMM_LIB` to point to
`libcudensitymat_distributed_interface_mpi.so`, as described in the
[Multi-GPU multi-node execution](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/overview.html#multi-gpu-multi-node-execution)
section of the cuDensityMat library documentation.

To execute the multi-GPU NCCL sample, run:
```
mpiexec -n N ./operator_action_nccl_example
```
where `N` is the desired number of MPI processes, which must be a power of two.
You will need to define the environment variable `CUDENSITYMAT_COMM_LIB` to point to
`libcudensitymat_distributed_interface_nccl.so`, as described in the
[Multi-GPU multi-node execution](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/overview.html#multi-gpu-multi-node-execution)
section of the cuDensityMat library documentation.

**Note**: Depending on how CUDA Toolkit and cuTENSOR are installed,
you might need to add them to `LD_LIBRARY_PATH`, for example:
```
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUTENSOR_ROOT/lib:$LD_LIBRARY_PATH
```
The exact library paths depends on how cuTENSOR and the CUDA toolkit were installed.

## Support

* **Supported SM Architectures:** SM 7.0, SM 7.5, SM 8.0, SM 8.6, SM 9.0, SM 10.0, SM 12.0
* **Supported OSes:** Linux
* **Supported CPU Architectures**: x86_64, aarch64-sbsa
* **Language**: C++17 or above

## Prerequisites

* [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) or higher and compatible driver
(see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* cuTENSOR 2.5.0+.
* CMake 3.22+ if using `cmake`.
* A CUDA-aware MPI library (e.g., OpenMPI, MPICH, MVAPICH) for the MPI and NCCL examples.
* NCCL for the NCCL example.
