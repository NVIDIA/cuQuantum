# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# Note: cuQuantum Python follows the cuQuantum SDK version, which is now
# switched to YY.MM and is different from individual libraries' (semantic)
# versioning scheme.

import os
import re
import sys
import shutil
import pathlib
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel


try:
    cuda_path = os.environ['CUDA_PATH']
except KeyError as e:
    raise RuntimeError('CUDA is not found, please set $CUDA_PATH') from e


def check_cuda_version():
    try:
        # We cannot do a dlopen and call cudaRuntimeGetVersion, because it
        # requires GPUs. We also do not want to rely on the compiler utility
        # provided in distutils (deprecated) or setuptools, as this is a very
        # simple string parsing task.
        # TODO: switch to cudaRuntimeGetVersion once it's fixed (nvbugs 3624208)
        cuda_h = os.path.join(cuda_path, 'include', 'cuda.h')
        with open(cuda_h, 'r') as f:
            cuda_h = f.read()
        m = re.search('#define CUDA_VERSION ([0-9]*)', cuda_h)
        if m:
            ver = int(m.group(1))
        else:
            raise RuntimeError("cannot parse CUDA_VERSION")
    except:
        raise
    else:
        # 12020 -> "12.2"
        return str(ver // 1000) + '.' + str((ver % 100) // 10)


# We support CUDA 12/13 starting 25.09
cuda_ver = check_cuda_version()

if '12.0' <= cuda_ver < '13.0':
    cuda_major_ver = '12'
elif '13.0' <= cuda_ver < '14.0':
    cuda_major_ver = '13'
else:
    raise RuntimeError(f"Unsupported CUDA version: {cuda_ver}")


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str = ""):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = []

        additional_cmake_args = os.environ.get("CMAKE_ARGS", "")
        if additional_cmake_args:
            cmake_args += additional_cmake_args.split()

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.source_dir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--verbose"] + build_args, cwd=self.build_temp
        )
        dso_dir = os.path.join(extdir, "cuquantum/lib")
        os.makedirs(dso_dir, exist_ok=True)
        for dso in pathlib.Path(self.build_temp).rglob("*.so"):
            shutil.copy2(dso, dso_dir)

class CustomBdistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

description = "NVIDIA cuQuantum Python JAX"
# Use README for the project long description
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.3"

classifiers=[
    "Development Status :: 5 - Production/Stable",
    "Operating System :: POSIX :: Linux",
    "Topic :: Education",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: Implementation :: CPython",
    "Environment :: GPU :: NVIDIA CUDA"
]

install_requires = ['pybind11']
if cuda_major_ver == '12':
    classifiers.append("Environment :: GPU :: NVIDIA CUDA :: 12")
    install_requires.append("cuquantum-python-cu12~=25.11")
    install_requires.append("jax[cuda12-local]>=0.5,<0.7")
else:
    classifiers.append("Environment :: GPU :: NVIDIA CUDA :: 13")
    install_requires.append("cuquantum-python-cu13~=25.11")
    install_requires.append("jax[cuda13-local]>=0.8,<0.9")

setup(
    name="cuquantum-python-jax",
    version=__version__,
    url="https://developer.nvidia.com/cuquantum-sdk",
    author="NVIDIA Corporation",
    author_email="cuquantum-python@nvidia.com",
    python_requires=">=3.11.0",
    install_requires=install_requires,
    license="BSD-3-Clause",
    license_files = ('LICENSE',),
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=classifiers,
    ext_modules=[
        CMakeExtension("cppsrc", source_dir="cuquantum/densitymat/jax/cppsrc"),
    ],
    cmdclass=dict(
        build_ext=CMakeBuild,
        bdist_wheel=CustomBdistWheel
    ),
    packages=[
        "cuquantum.densitymat.jax",
        "cuquantum.densitymat.jax.pysrc"
    ],
    package_data={
        "cuquantum.densitymat.jax": ["**/*.so"]
    },
    exclude_package_data={
        "cuquantum.densitymat.jax": ["cppsrc/*"],
        "": ["tests/*"]
    },
    include_package_data=True,
    zip_safe=False
)
