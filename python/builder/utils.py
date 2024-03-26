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
        # 11020 -> "11.2"
        return str(ver // 1000) + '.' + str((ver % 100) // 10)


# We support CUDA 11/12 starting 23.03
cuda_ver = check_cuda_version()
if cuda_ver == '11.0':
    cuda_major_ver = '11'
elif '11.0' < cuda_ver < '12.0':
    cuda_major_ver = '11'
elif '12.0' <= cuda_ver < '13.0':
    cuda_major_ver = '12'
else:
    raise RuntimeError(f"Unsupported CUDA version: {cuda_ver}")


building_wheel = False


class bdist_wheel(_bdist_wheel):

    def run(self):
        global building_wheel
        building_wheel = True
        super().run()


class build_ext(_build_ext):

    def _set_library_roots(self):
        custatevec_root = cutensornet_root = cuquantum_root = None
        # Note that we need sys.path because of build isolation (since PEP 517)
        py_paths = sys.path + [site.getusersitepackages()] + site.getsitepackages()

        # search order:
        # 1. installed "cuquantum" package
        # 2. env var
        for path in py_paths:
            path = os.path.join(path, 'cuquantum')
            if os.path.isdir(os.path.join(path, 'include')):
                custatevec_root = cutensornet_root = path
                break
        else:
            # We allow setting CUSTATEVEC_ROOT and CUTENSORNET_ROOT separately for the ease
            # of development, but users are encouraged to either install cuquantum from PyPI
            # or conda, or set CUQUANTUM_ROOT to the existing installation.
            cuquantum_root = os.environ.get('CUQUANTUM_ROOT')
            try:
                custatevec_root = os.environ['CUSTATEVEC_ROOT']
            except KeyError as e:
                if cuquantum_root is None:
                    raise RuntimeError('cuStateVec is not found, please set $CUQUANTUM_ROOT '
                                       'or $CUSTATEVEC_ROOT') from e
            try:
                cutensornet_root = os.environ['CUTENSORNET_ROOT']
            except KeyError as e:
                if cuquantum_root is None:
                    raise RuntimeError('cuTensorNet is not found, please set $CUQUANTUM_ROOT '
                                       'or $CUTENSORNET_ROOT') from e

        return custatevec_root, cutensornet_root, cuquantum_root

    def _prep_includes_libs_rpaths(self):
        """
        Set global vars cusv_incl_dir, cutn_incl_dir, and extra_linker_flags.

        With the new bindings, we no longer need to link to cuQuantum DSOs.
        """
        custatevec_root, cutensornet_root, cuquantum_root = self._set_library_roots()

        global cusv_incl_dir, cutn_incl_dir, cuqnt_incl_dir
        cusv_incl_dir = cutn_incl_dir = cuqnt_incl_dir = None
        base_incl_dir = (os.path.join(cuda_path, 'include'),)
        if cuquantum_root is not None:
            cuqnt_incl_dir = base_incl_dir + (os.path.join(cuquantum_root, 'include'),)
        if custatevec_root is not None:
            cusv_incl_dir = base_incl_dir + (os.path.join(custatevec_root, 'include'),)
        if cutensornet_root is not None:
            cutn_incl_dir = base_incl_dir + (os.path.join(cutensornet_root, 'include'),)

        global extra_linker_flags
        if not building_wheel:
            # Note: with PEP-517 the editable mode would not build a wheel for installation
            # (and we purposely do not support PEP-660).
            extra_linker_flags = []
        else:
            # Note: soname = library major version
            # We don't need to link to cuBLAS/cuSOLVER/cuTensor at build time
            # The rpaths must be adjusted given the following full-wheel installation:
            # - cuquantum-python: site-packages/cuquantum/{custatevec, cutensornet}/_internal/  [=$ORIGIN]
            # - cusv & cutn:      site-packages/cuquantum/lib/
            # - cutensor:         site-packages/cutensor/lib/
            # - cublas:           site-packages/nvidia/cublas/lib/
            # - cusolver:         site-packages/nvidia/cusolver/lib/
            # (Note that starting v22.11 we use the new wheel format, so all lib wheels have suffix -cuXX,
            #  and cuBLAS/cuSOLVER additionally have prefix nvidia-.)
            ldflag = "-Wl,--disable-new-dtags,"
            ldflag += "-rpath,$ORIGIN/../../lib,"
            ldflag += "-rpath,$ORIGIN/../../../cutensor/lib,"
            ldflag += "-rpath,$ORIGIN/../../../nvidia/cublas/lib,"
            ldflag += "-rpath,$ORIGIN/../../../nvidia/cusolver/lib"
            extra_linker_flags = [ldflag]

        print("\n"+"*"*80)
        print("CUDA version:", cuda_ver)
        print("CUDA path:", cuda_path)
        print("cuStateVec path:", custatevec_root if custatevec_root else cuquantum_root)
        print("cuTensorNet path:", cutensornet_root if cutensornet_root else cuquantum_root)
        print("*"*80+"\n")

    def build_extension(self, ext):
        ext.include_dirs = ()
        for include_dir in (cusv_incl_dir, cutn_incl_dir, cuqnt_incl_dir):
            if include_dir is not None:
                ext.include_dirs += include_dir
        if ext.name.endswith("custatevec"):
            ext.extra_link_args = extra_linker_flags
        elif ext.name.endswith("cutensornet"):
            ext.extra_link_args = extra_linker_flags

        super().build_extension(ext)

    def build_extensions(self):
        self._prep_includes_libs_rpaths()
        self.parallel = 4  # use 4 threads
        super().build_extensions()
