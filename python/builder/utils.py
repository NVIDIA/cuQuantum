# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import site
import sys

from packaging.version import Version
from setuptools.command.build_ext import build_ext as _build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


# Get __version__ variable
source_root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(source_root, '..', 'cuquantum', '_version.py')) as f:
    exec(f.read())
cuqnt_py_ver = __version__
cuqnt_py_ver_obj = Version(cuqnt_py_ver)
cuqnt_ver_major_minor = f"{cuqnt_py_ver_obj.major}.{cuqnt_py_ver_obj.minor}"

del __version__, cuqnt_py_ver_obj, source_root


# We can't assume users to have CTK installed via pip, so we really need this...
# TODO(leofang): try /usr/local/cuda?
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
print("\n"+"*"*80)
print("CUDA version:", cuda_ver)
print("CUDA path:", cuda_path)
print("*"*80+"\n")

if '12.0' <= cuda_ver < '13.0':
    cuda_major_ver = '12'
elif '13.0' <= cuda_ver < '14.0':
    cuda_major_ver = '13'
else:
    raise RuntimeError(f"Unsupported CUDA version: {cuda_ver}")


building_wheel = False


class bdist_wheel(_bdist_wheel):

    def run(self):
        global building_wheel
        building_wheel = True
        super().run()


class build_ext(_build_ext):

    def _prep_includes_libs_rpaths(self, ext_name):
        """
        Set global vars extra_linker_flags.

        With the new bindings, we no longer need to link to cuQuantum DSOs.
        """

        if not building_wheel:
            # Note: with PEP-517 the editable mode would not build a wheel for installation
            # (and we purposely do not support PEP-660).
            extra_linker_flags = []
        else:
            # Note: soname = library major version
            # We don't need to link to cuBLAS/cuSOLVER/cuTensor at build time
            # The rpaths must be adjusted given the following full-wheel installation:
            # - cuquantum-python: site-packages/cuquantum/bindings/_internal/  [=$ORIGIN]
            # - cusv, cutn & cudm:      site-packages/cuquantum/lib/
            # - cutensor:         site-packages/cutensor/lib/
            # - cublas:           site-packages/nvidia/cublas/lib/
            # - cusolver:         site-packages/nvidia/cusolver/lib/
            # (Note that starting v22.11 we use the new wheel format, so all lib wheels have suffix -cuXX,
            #  and cuBLAS/cuSOLVER additionally have prefix nvidia-.)
            ldflag = "-Wl,--disable-new-dtags"
            ldflag += ",-rpath,$ORIGIN/../../lib"
            ldflag += ",-rpath,$ORIGIN/../../../nvidia/cublas/lib"
            if "cutensornet" in ext_name or "cudensitymat" in ext_name:
                ldflag += ",-rpath,$ORIGIN/../../../cutensor/lib"
                ldflag += ",-rpath,$ORIGIN/../../../nvidia/cusolver/lib"
                #TODO: curand is only a cudensitymat dependency, not cutensornet
                ldflag += ",-rpath,$ORIGIN/../../../nvidia/curand/lib"
            extra_linker_flags = [ldflag]

        return extra_linker_flags

    def build_extension(self, ext):
        ext.include_dirs = (os.path.join(cuda_path, 'include'),)
        ext.extra_link_args = self._prep_includes_libs_rpaths(ext.name)
        super().build_extension(ext)

    def build_extensions(self):
        self.parallel = 4  # use 4 threads
        super().build_extensions()
