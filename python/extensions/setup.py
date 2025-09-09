# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# Note: cuQuantum Python follows the cuQuantum SDK version, which is now
# switched to YY.MM and is different from individual libraries' (semantic)
# versioning scheme.

import os
import shutil
import pathlib
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel


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

__version__ = "0.0.1"

setup(
    name="cuquantum-python-jax",
    version=__version__,
    url="https://developer.nvidia.com/cuquantum-sdk",
    author="NVIDIA Corporation",
    author_email="cuquantum-python@nvidia.com",
    python_requires=">=3.11.0",
    install_requires=[
        "cuquantum-python-cu12~=25.09",
        "jax[cuda12-local]>=0.5,<0.7",
        "pybind11"
    ],
    license="BSD-3-Clause",
    license_files = ('LICENSE',),
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
    ],
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
