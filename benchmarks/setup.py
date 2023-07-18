# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import importlib
import os
import shutil
import site
import subprocess
import sys

from setuptools import find_packages, setup


source_root = os.path.abspath(os.path.dirname(__file__))

# Use README for the project long description
with open(os.path.join(source_root, "README.md")) as f:
    long_description = f.read()

# Get project version
with open(os.path.join(source_root, "cuquantum_benchmarks", "__init__.py")) as f:
    exec(f.read())
    version = __version__
    del __version__


# A user could have cuquantum-python-cuXX installed but not cuquantum-python,
# so before asking pip to install it we need to confirm
install_requires = [
    "psutil",
    "scipy",
    "networkx",
    "nvtx",
]
if importlib.util.find_spec('cuquantum') is None:
    install_requires.append("cuquantum-python>=23.3")


setup(
    name="cuquantum-benchmarks",
    version=version,
    description="NVIDIA cuQuantum Performance Benchmark Suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/cuQuantum",
    author="NVIDIA Corporation",
    author_email="cuquantum-python@nvidia.com",
    license="BSD-3-Clause",
    license_files = ('LICENSE',),
    keywords=["cuda", "nvidia", "state vector", "tensor network", "high-performance computing", "quantum computing",
              "quantum circuit simulation"],
    packages=find_packages(include=['cuquantum_benchmarks', 'cuquantum_benchmarks.*']),
    package_data={"": ["*.py"],},
    entry_points = {
        'console_scripts': [
            'cuquantum-benchmarks = cuquantum_benchmarks.run:run',
        ]
    },
    zip_safe=False,
    setup_requires=[
        "setuptools",
    ],
    install_requires=install_requires,
    extras_require={
        "all": ["cirq", "qsimcirq", "qiskit", "pennylane", "pennylane-lightning", "pennylane-lightning[gpu]"],
    },
    classifiers=[
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
