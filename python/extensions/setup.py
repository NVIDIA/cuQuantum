# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# Note: Project metadata (name, version, dependencies, classifiers) is
# declared in pyproject.toml, which is generated from pyproject.toml.template
# by configure.sh.  This file only contains build mechanics (CMake extension
# compilation, wheel packaging, and package layout).

import os
import sys
import shutil
import pathlib
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel


# ---------------------------------------------------------------------------
# Guard: verify that configure.sh has been run before building.
# ---------------------------------------------------------------------------
_pyproject = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml")
if os.path.exists(_pyproject):
    with open(_pyproject) as _f:
        if "[project]" not in _f.read():
            raise RuntimeError(
                "pyproject.toml does not contain [project] metadata.\n"
                "Run configure.sh first:\n"
                "  CUDA_PATH=/usr/local/cuda bash configure.sh"
            )
else:
    raise RuntimeError(
        "pyproject.toml not found.\n"
        "Run configure.sh first:\n"
        "  CUDA_PATH=/usr/local/cuda bash configure.sh"
    )


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
        cmake_args += [f"-DPython3_EXECUTABLE={sys.executable}"]
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

setup(
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
