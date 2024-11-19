# Welcome to the cuQuantum repository!

<img align="right" width="200"
src="https://developer.nvidia.com/sites/default/files/akamai/nvidia-cuquantum-icon.svg"
/>

This public repository contains a few sets of files related to the [NVIDIA cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk):

- `benchmarks`: NVIDIA cuQuantum Performance Benchmark Suite (v0.3.0), see [README](./benchmarks/README.md) for detail.
- `extra`: Files to help utilize the cuQuantum SDK and the cuQuantum Appliance container.
- `python`: The open-sourced cuQuantum Python project.
  - Available for download on
    - conda-forge:
      - `cuquantum` [![Conda Version](https://img.shields.io/conda/vn/conda-forge/cuquantum.svg)](https://anaconda.org/conda-forge/cuquantum)
        - `cudensitymat` [![Conda Version](https://img.shields.io/conda/vn/conda-forge/cudensitymat.svg)](https://anaconda.org/conda-forge/cudensitymat)
        - `custatevec` [![Conda Version](https://img.shields.io/conda/vn/conda-forge/custatevec.svg)](https://anaconda.org/conda-forge/custatevec)
        - `cutensornet` [![Conda Version](https://img.shields.io/conda/vn/conda-forge/cutensornet.svg)](https://anaconda.org/conda-forge/cutensornet)
      - `cuquantum-python` [![Conda Version](https://img.shields.io/conda/vn/conda-forge/cuquantum-python.svg)](https://anaconda.org/conda-forge/cuquantum-python)
    - PyPI:
      - `cuquantum` [![pypi](https://img.shields.io/pypi/v/cuquantum.svg)](https://pypi.python.org/pypi/cuquantum)
        - `cuquantum-cu11` [![pypi](https://img.shields.io/pypi/v/cuquantum-cu11.svg)](https://pypi.python.org/pypi/cuquantum-cu11)
          - `cudensitymat-cu11` [![pypi](https://img.shields.io/pypi/v/cudensitymat-cu11.svg)](https://pypi.python.org/pypi/cudensitymat-cu11)
          - `custatevec-cu11` [![pypi](https://img.shields.io/pypi/v/custatevec-cu11.svg)](https://pypi.python.org/pypi/custatevec-cu11)
          - `cutensornet-cu11` [![pypi](https://img.shields.io/pypi/v/cutensornet-cu11.svg)](https://pypi.python.org/pypi/cutensornet-cu11)
        - `cuquantum-cu12` [![pypi](https://img.shields.io/pypi/v/cuquantum-cu12.svg)](https://pypi.python.org/pypi/cuquantum-cu12)
          - `cudensitymat-cu12` [![pypi](https://img.shields.io/pypi/v/cudensitymat-cu12.svg)](https://pypi.python.org/pypi/cudensitymat-cu12)
          - `custatevec-cu12` [![pypi](https://img.shields.io/pypi/v/custatevec-cu12.svg)](https://pypi.python.org/pypi/custatevec-cu12)
          - `cutensornet-cu12` [![pypi](https://img.shields.io/pypi/v/cutensornet-cu12.svg)](https://pypi.python.org/pypi/cutensornet-cu12)
      - `cuquantum-python` [![pypi](https://img.shields.io/pypi/v/cuquantum-python.svg)](https://pypi.python.org/pypi/cuquantum-python)
        - `cuquantum-python-cu11` [![pypi](https://img.shields.io/pypi/v/cuquantum-python-cu11.svg)](https://pypi.python.org/pypi/cuquantum-python-cu11)
        - `cuquantum-python-cu12` [![pypi](https://img.shields.io/pypi/v/cuquantum-python-cu12.svg)](https://pypi.python.org/pypi/cuquantum-python-cu12)
- `samples`: All C/C++ sample codes for the cuQuantum SDK.

## Installation

The instructions for how to build and install these files are given in both the subfolders and
the [cuQuantum documentation](https://docs.nvidia.com/cuda/cuquantum/latest/index.html).

## License

All files hosted in this repository are subject to the [BSD-3-Clause](./LICENSE) license.

## Citing cuQuantum

H. Bayraktar et al., "cuQuantum SDK: A High-Performance Library for Accelerating Quantum Science", 2023 IEEE International Conference on Quantum Computing and Engineering (QCE), Bellevue, WA, USA, 2023, pp. 1050-1061, doi: [10.1109/QCE57702.2023.00119](https://doi.org/10.1109/QCE57702.2023.00119)
