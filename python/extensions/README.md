# cuQuantum Python JAX

cuQuantum Python JAX provides a JAX extension for cuQuantum Python. It exposes selected functionalities of cuQuantum SDK in a JAX-compatible way that enables JAX frameworks to directly interface with the exposed cuQuantum API. In the current release, cuQuantum JAX exposes JAX interfaces to the Operator Action API from the cuDensityMat library and the GF(2) sparse-dense matrix multiplication from the cuStabilizer library.

## Documentation

Please visit the [NVIDIA cuQuantum Python documentation](https://docs.nvidia.com/cuda/cuquantum/latest/python).

## Building and installing cuQuantum Python JAX

### Requirements

The build-time dependencies of the cuQuantum Python JAX package include:

* Python 3.11+
* CUDA Toolkit 12.x or 13.x
* cmake>=3.22
* jax[cuda12-local]>=0.8 for CUDA 12 or jax[cuda13-local]>=0.8 for CUDA 13
* pybind11
* wheel
* setuptools>=77.0.3

Note:
- cuQuantum Python JAX is only supported with CUDA 12 and CUDA 13 and the distributions are CUDA-versioned: `cuquantum-python-jax-cu12` for CUDA 12 and `cuquantum-python-jax-cu13` for CUDA 13.
- **Build isolation note:** When installing with build isolation (`pip`'s default mode), the JAX version used at build time is determined by the build dependency specification in `pyproject.toml` and may differ from the JAX version pre-installed in the environment. Mismatched build-time and runtime JAX versions can lead to errors when importing cuQuantum's JAX extension. We recommend passing `--no-build-isolation` to `pip install` and pre-installing all build dependencies manually.

#### Installation using `jax[cudaXX-local]`

`cuquantum-python-jax-cu12` (or `cuquantum-python-jax-cu13`) depends explicitly on `jax[cudaXX-local]`. Installing the package will also install `jax[cudaXX-local]`.

Using `jax[cudaXX-local]` assumes the user provides both cuDNN and the CUDA Toolkit. cuDNN is not a part of the CUDA Toolkit and requires an additional installation. The user must also specify `LD_LIBRARY_PATH`, including the library folders containing `libcudnn.so` and `libcupti.so`.

`libcupti.so` is provided by the CUDA Toolkit. If the CUDA Toolkit is installed under `/usr/local/cuda`, `libcupti.so` is located under `/usr/local/cuda/extras/CUPTI/lib64` and `LD_LIBRARY_PATH` should contain this path.

`libcudnn.so` is installed separately from the CUDA Toolkit. The default installation location is `/usr/local/cuda/lib64`, and `LD_LIBRARY_PATH` should contain this path.

Both `libcudnn.so` and `libcupti.so` are installable with pip:

```
pip install nvidia-cudnn-cu12
pip install nvidia-cuda-cupti-cu12
```

After installing cuDNN and cuPTI, the user may install cuQuantum Python JAX with `pip` using either:

```
pip install cuquantum-python-jax-cu12   # for CUDA 12
pip install cuquantum-python-jax-cu13   # for CUDA 13
```

or one of

```
pip install cuquantum-python-cu12[jax]
pip install cuquantum-python-cu13[jax]
```

where the CUDA version is explicitly specified on cuquantum-python.

Note:
- If cuDNN and cuPTI are installed with `pip`, the user does not need to specify library folders in `LD_LIBRARY_PATH`.

#### Installing from source

To install cuQuantum Python JAX from source, first compile cuQuantum Python from source using the [instructions on GitHub](https://github.com/NVIDIA/cuQuantum/blob/main/python/README.md). Once complete, navigate to `python/extensions`, run `./configure.sh` to generate a CUDA version-specific `pyproject.toml` from the template, and then:

```
pip install .
```

The CUDA version is detected automatically from `$CUDA_PATH` and the wheel will be named accordingly (`cuquantum-python-jax-cu12` or `cuquantum-python-jax-cu13`).

## Running

### Requirements

Runtime dependencies of the cuQuantum Python JAX package include:

* Python 3.11+
* An NVIDIA GPU with compute capability 7.5+
* CUDA Toolkit 12.x or 13.x
* cuquantum-python-cu12~=26.11.0 for CUDA 12 or cuquantum-python-cu13~=26.11.0 for CUDA 13
* jax[cuda12-local]>=0.8 for CUDA 12 or jax[cuda13-local]>=0.8 for CUDA 13

## Developer Notes

* cuQuantum Python JAX does not support editable installation.
* Both cuQuantum Python and cuQuantum Python JAX need to be installed into `site-packages` for proper import of the library.
* cuQuantum Python JAX assumes cuQuantum Python will be available under the current `site-packages` directory.
