# cuQuantum Python

## Documentation

Please visit the [NVIDIA cuQuantum Python documentation](https://docs.nvidia.com/cuda/cuquantum/latest/python).

For instructions on installing *cuQuantum Python*, refer to our 
[getting started section](../getting-started/index.rst)

## Building and installing cuQuantum Python from source

### Requirements

The build-time dependencies of the cuQuantum Python package include:

* CUDA Toolkit 11.x or 12.x
* cuStateVec 1.4.0+
* cuTensorNet 2.5.0+
* Python 3.10+
* Cython >=0.29.22,<3
* pip 21.3.1+
* [packaging](https://packaging.pypa.io/en/latest/)
* setuptools 61.0.0+
* wheel 0.34.0+

Except for CUDA and Python, the rest of the build-time dependencies are handled by the new PEP-517-based build system (see Step 7 below).

To compile and install cuQuantum Python from source, please follow the steps below:

1. Clone the [NVIDIA/cuQuantum](https://github.com/NVIDIA/cuQuantum) repository: `git clone https://github.com/NVIDIA/cuQuantum`
2. Set `CUDA_PATH` to point to your CUDA installation
3. [optional] Set `CUQUANTUM_ROOT` to point to your cuQuantum installation
4. [optional] Set `CUTENSOR_ROOT` to point to your cuTENSOR installation
5. [optional] Make sure cuQuantum and cuTENSOR are visible in your `LD_LIBRARY_PATH`
6. Switch to the directory containing the Python implementation: `cd cuQuantum/python`
7. Build and install:
   - Run `pip install .` if you skip Step 3-5 above
   - Run `pip install -v --no-deps --no-build-isolation .` otherwise (advanced)

Notes:
- For Step 7, if you are building from source for testing/developing purposes you'd likely want to insert a `-e` flag before the last period (so `pip ... .` becomes `pip ... -e .`):
  * `-e`: use the "editable" (in-place) mode
  * `-v`: enable more verbose output
  * `--no-deps`: avoid installing the *run-time* dependencies
  * `--no-build-isolation`: reuse the current Python environment instead of creating a new one for building the package (this avoids installing any *build-time* dependencies)
- As an alternative to setting `CUQUANTUM_ROOT`, `CUSTATEVEC_ROOT` and `CUTENSORNET_ROOT` can be set to point to the cuStateVec and the cuTensorNet libraries, respectively. The latter two environment variables take precedence if defined.
- Please ensure that you use consistent binaries and packages for either CUDA 11 or 12. Mixing-and-matching will result in undefined behavior.


## Running

### Requirements

Runtime dependencies of the cuQuantum Python package include:

* An NVIDIA GPU with compute capability 7.0+
* Driver: Linux (450.80.02+ for CUDA 11, 525.60.13+ for CUDA 12)
* CUDA Toolkit 11.x or 12.x
* cuStateVec 1.4.0+
* cuTensorNet 2.5.0+
* Python 3.10+
* NumPy v1.21+
* CuPy v13.0.0+ (see [installation guide](https://docs.cupy.dev/en/stable/install.html))
* PyTorch v1.10+ (optional, see [installation guide](https://pytorch.org/get-started/locally/))
* Qiskit v0.24.0+ (optional, see [installation guide](https://qiskit.org/documentation/getting_started.html))
* Cirq v0.6.0+ (optional, see [installation guide](https://quantumai.google/cirq/install))
* mpi4py v3.1.0+ (optional, see [installation guide](https://mpi4py.readthedocs.io/en/stable/install.html))

If you install everything from conda-forge, all the required dependencies are taken care for you (except for the driver).

If you install the pip wheels, CuPy, cuTENSOR and cuQuantum (but not CUDA Toolkit or the driver,
please make sure the CUDA libraries are visible through your `LD_LIBRARY_PATH`) are installed for you.

If you build cuQuantum Python from source, please make sure that the paths to the CUDA, cuQuantum, and cuTENSOR libraries are added
to your `LD_LIBRARY_PATH` environment variable, and that a compatible CuPy is installed.

Known issues:
- If a system has multiple copies of cuTENSOR, one of which is installed in a default system path, the Python runtime could pick it up despite cuQuantum Python is linked to another copy installed elsewhere, potentially causing a version-mismatch error. The proper fix is to remove cuTENSOR from the system paths to ensure the visibility of the proper copy. **DO NOT ATTEMPT** to use `LD_PRELOAD` to overwrite it --- it could cause hard to debug behaviors!
- In certain environments, if PyTorch is installed `import cuquantum` could fail (with a segmentation fault). It is currently under investigation and a temporary workaround is to import `torch` before importing `cuquantum`.
- Please ensure that you use consistent binaries and packages for either CUDA 11 or 12. Mixing-and-matching will result in undefined behavior.

### Samples

Samples for demonstrating the usage of both low-level and high-level Python APIs are
available in the `samples` directory. The low-level API samples are 1:1 translations of the corresponding
samples written in C. The high-level API samples demonstrate pythonic usages of the cuTensorNet
library in Python.


## Testing

If pytest is installed, typing `pytest tests` at the command prompt in the Python source root directory will
run all tests. Some tests would be skipped if `cffi` is not installed or if the environment
variable `CUDA_PATH` is not set.


## Citing cuQuantum

Please click this Zenodo badge to see the citation format: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6385574.svg)](https://doi.org/10.5281/zenodo.6385574)
