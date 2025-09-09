# cuQuantum Python JAX

cuQuantum Python JAX provides a JAX extension for cuQuantum Python. It exposes selected functionality of cuQuantum SDK in a JAX-compatible way that enables JAX frameworks to directly interface with the exposed cuQuantum API. In the current release, cuQuantum JAX exposes a JAX interface to the Operator Action API from the cuDensityMat library.

## Documentation

Please visit the [NVIDIA cuQuantum Python documentation](https://docs.nvidia.com/cuda/cuquantum/latest/python).

## Building and installing cuQuantum Python JAX

### Requirements

The install-time dependencies of the cuQuantum Python package include:

* cuquantum-python-cu12~=25.09
* setuptools>=77.0.3
* jax[cuda12]>=0.5,<0.7 or jax[cuda12-local]>=0.5,<0.7
* pybind11

Note: cuQuantum Python JAX is only supported with CUDA 12.

#### Installation using `jax[cuda12-local]`

`cuquantum-python-jax` depends explicitly on `jax[cuda12-local]`. `pip install cuquantum-python-jax` will install `jax[cuda12-local]`.

Using `jax[cuda12-local]` assumes the user provides both cuDNN and the CUDA Toolkit. cuDNN is not a part of the CUDA Toolkit and requires an additional installation. The user must also specify `LD_LIBRARY_PATH`, including the library folders containing `libcudnn.so` and `libcupti.so`.

`libcupti.so` is provided by the CUDA Toolkit. If the CUDA Toolkit is installed under `/usr/local/cuda`, `libcupti.so` is located under `/usr/local/cuda/extras/CUPTI/lib64` and `LD_LIBRARY_PATH` should contain this path.

`libcudnn.so` is installed separately from the CUDA Toolkit. The default installation location is `/usr/local/cuda/lib64`, and `LD_LIBRARY_PATH` should contain this path.

Both `libcudnn.so` and `libcupti.so` are installable with pip:

```
pip install nvidia-cudnn-cu12
pip install nvidia-cuda-cupti-cu12
```

After installing cuDNN and cuPTI, the user may install `cuquantum-python-jax` using `pip` using either:

```
pip install cuquantum-python-jax
```

or

```
pip install cuquantum-python-cu12[jax]
```

Note: if cuDNN and cuPTI are installed with `pip`, the user does not need to specify library folders in `LD_LIBRARY_PATH`.

#### Installation using `jax[cuda12]`

Alternatively, the user may 

```
pip install jax[cuda12]  # install cuPTI and cuDNN together with CUDA-enabled JAX
```

and either 

```
pip install cuquantum-python-jax
```

or

```
pip install cuquantum-python-cu12[jax]
```

Warning: if the user has an installation of CUDA outside of `pip`, this may create conflicts and undefined behavior.

#### Installing from source

To install cuQuantum Python JAX from source, first compile cuQuantum Python from source using the [instructions on GitHub](https://github.com/NVIDIA/cuQuantum/blob/main/python/README.md). Once complete, navigate to `python/extensions`, then:

```
export CUDENSITYMAT_ROOT=...
pip install .
```

Where `CUDENSITYMAT_ROOT` is the path to the libraries parent directory. For example, if `CUDENSITYMAT_ROOT=/usr/local`, `libcudensitymat.so` would be found under `/usr/local/lib` or `/usr/local/lib64`.

## Running

### Requirements

Runtime dependencies of the cuQuantum Python package include:

* An NVIDIA GPU with compute capability 7.5+
* cuquantum-python-cu12~=25.09
* jax[cuda12]>=0.5,<0.7 or jax[cuda12-local]>=0.5,<0.7
* pybind11

## Developer Notes

* cuQuantum Python JAX does not support editable installation.
* Both cuQuantum Python and cuQuantum Python JAX need to be installed into `site-packages` for proper import of the library.
* cuQuantum Python JAX assumes cuQuantum Python will be available under the current `site-packages` directory.
