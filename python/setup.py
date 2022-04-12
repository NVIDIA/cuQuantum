# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import site
import subprocess
import sys

from packaging.version import Version
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


# Get __version__ variable
source_root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(source_root, 'cuquantum', '_version.py')) as f:
    exec(f.read())


# set up version constraints: note that CalVer like 22.03 is normalized to
# 22.3 by setuptools, so we must follow the same practice in the constraints;
# also, we don't need the Python patch number here
cuqnt_py_ver = Version(__version__)
cuqnt_ver_major_minor = f"{cuqnt_py_ver.major}.{cuqnt_py_ver.minor}"


# search order:
# 1. installed "cuquantum" package
# 2. env var
for path in site.getsitepackages():
    path = os.path.join(path, 'cuquantum')
    if os.path.isdir(path):
        cuquantum_root = path
        using_cuquantum_wheel = True
        break
else:
    cuquantum_root = os.environ.get('CUQUANTUM_ROOT')
    using_cuquantum_wheel = False


# We allow setting CUSTATEVEC_ROOT and CUTENSORNET_ROOT separately for the ease
# of development, but users are encouraged to either install cuquantum from PyPI
# or conda, or set CUQUANTUM_ROOT to the existing installation.
try:
    custatevec_root = os.environ['CUSTATEVEC_ROOT']
    using_cuquantum_wheel = False
except KeyError as e:
    if cuquantum_root is None:
        raise RuntimeError('cuStateVec is not found, please install "cuquantum" '
                           'or set $CUQUANTUM_ROOT') from e
    else:
        custatevec_root = cuquantum_root
try:
    cutensornet_root = os.environ['CUTENSORNET_ROOT']
    using_cuquantum_wheel = False
except KeyError as e:
    if cuquantum_root is None:
        raise RuntimeError('cuTensorNet is not found, please install "cuquantum" '
                           'or set $CUQUANTUM_ROOT') from e
    else:
        cutensornet_root = cuquantum_root


# search order:
# 1. installed "cutensor" package
# 2. env var
for path in site.getsitepackages():
    path = os.path.join(path, 'cutensor')
    if os.path.isdir(path):
        cutensor_root = path
        assert using_cuquantum_wheel  # if this raises, the env is corrupted
        break
else:
    cutensor_root = os.environ.get('CUTENSOR_ROOT')
    assert not using_cuquantum_wheel
if cutensor_root is None:
    raise RuntimeError('cuTENSOR is not found, please install "cutensor" '
                       'or set $CUTENSOR_ROOT')


# We can't assume users to have CTK installed via pip, so we really need this...
# TODO(leofang): try /usr/local/cuda?
try:
    cuda_path = os.environ['CUDA_PATH']
except KeyError as e:
    raise RuntimeError('CUDA is not found, please set $CUDA_PATH') from e


# TODO: use setup.cfg and/or pyproject.toml
setup_requires = [
    'Cython>=0.29.22,<3',
    'packaging',
    ]
install_requires = [
    'numpy',
    # 'cupy', # <-- can't be listed here as on PyPI this is the name for source build, not for wheel
    # 'torch', # <-- PyTorch is optional; also, it does not live on PyPI...
    ]
ignore_cuquantum_dep = bool(os.environ.get('CUQUANTUM_IGNORE_SOLVER', False))
if not ignore_cuquantum_dep:
    assert using_cuquantum_wheel  # if this raises, the env is corrupted
    # cuTENSOR version is constrained in the cuquantum package, so we don't
    # need to list it
    setup_requires.append(f'cuquantum=={cuqnt_ver_major_minor}.*')
    install_requires.append(f'cuquantum=={cuqnt_ver_major_minor}.*')


def check_cuda_version():
    try:
        # We cannot do a dlopen and call cudaRuntimeGetVersion, because it
        # requires GPUs. We also do not want to rely on the compiler utility
        # provided in distutils (deprecated) or setuptools, as this is a very
        # simple string parsing task.
        cuda_h = os.path.join(cuda_path, 'include', 'cuda.h')
        with open(cuda_h, 'r') as f:
            cuda_h = f.read().split('\n')
        for line in cuda_h:
            if "#define CUDA_VERSION" in line:
                ver = int(line.split()[-1])
                break
        else:
            raise RuntimeError("cannot parse CUDA_VERSION")
    except:
        raise
    else:
        # 11020 -> "11.2"
        return str(ver // 1000) + '.' + str((ver % 100) // 10)


cuda_ver = check_cuda_version()
if cuda_ver in ('10.2', '11.0'):
    cutensor_ver = cuda_ver
elif '11.0' < cuda_ver < '12.0':
    cutensor_ver = '11'
else:
    raise RuntimeError(f"Unsupported CUDA version: {cuda_ver}")


def prepare_libs_and_rpaths():
    global cusv_lib_dir, cutn_lib_dir
    # we include both lib64 and lib to accommodate all possible sources
    cusv_lib_dir = [os.path.join(custatevec_root, 'lib'),
                    os.path.join(custatevec_root, 'lib64')]
    cutn_lib_dir = [os.path.join(cutensornet_root, 'lib'),
                    os.path.join(cutensornet_root, 'lib64'),
                    os.path.join(cutensor_root, 'lib', cutensor_ver)]

    global cusv_lib, cutn_lib, extra_linker_flags
    if using_cuquantum_wheel:
        cusv_lib = [':libcustatevec.so.1']
        cutn_lib = [':libcutensornet.so.1', ':libcutensor.so.1']
        # The rpaths must be adjusted given the following full-wheel installation:
        #   cuquantum-python: site-packages/cuquantum/{custatevec, cutensornet}/  [=$ORIGIN]
        #   cusv & cutn:      site-packages/cuquantum/lib/
        #   cutensor:         site-packages/cutensor/lib/CUDA_VER/
        ldflag = "-Wl,--disable-new-dtags,"
        ldflag += "-rpath,$ORIGIN/../lib,"
        ldflag += f"-rpath,$ORIGIN/../../cutensor/lib/{cutensor_ver}"
        extra_linker_flags = [ldflag]
    else:
        cusv_lib = ['custatevec']
        cutn_lib = ['cutensornet', 'cutensor']
        extra_linker_flags = []


prepare_libs_and_rpaths()
print("\n****************************************************************")
print("CUDA version:", cuda_ver)
print("CUDA path:", cuda_path)
print("cuStateVec path:", custatevec_root)
print("cuTensorNet path:", cutensornet_root)
print("cuTENSOR path:", cutensor_root)
print("****************************************************************\n")


custatevec = Extension(
    "cuquantum.custatevec.custatevec",
    sources=["cuquantum/custatevec/custatevec.pyx"],
    include_dirs=[os.path.join(cuda_path, 'include'),
                  os.path.join(custatevec_root, 'include')],
    library_dirs=cusv_lib_dir,
    libraries=cusv_lib,
    extra_link_args=extra_linker_flags,
)


cutensornet = Extension(
    "cuquantum.cutensornet.cutensornet",
    sources=["cuquantum/cutensornet/cutensornet.pyx"],
    include_dirs=[os.path.join(cuda_path, 'include'),
                  os.path.join(cutensornet_root, 'include')],
    library_dirs=cutn_lib_dir,
    libraries=cutn_lib,
    extra_link_args=extra_linker_flags,
)


setup(
    name="cuquantum-python",
    version=__version__,
    description="Python APIs for cuQuantum",
    url="https://github.com/NVIDIA/cuQuantum",
    author="NVIDIA Corporation",
    author_email="cuquantum-python@nvidia.com",
    license="BSD-3-Clause",
    license_files = ('LICENSE',),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
        "Environment :: GPU :: NVIDIA CUDA :: 11.1",
        "Environment :: GPU :: NVIDIA CUDA :: 11.2",
        "Environment :: GPU :: NVIDIA CUDA :: 11.3",
        "Environment :: GPU :: NVIDIA CUDA :: 11.4",
        "Environment :: GPU :: NVIDIA CUDA :: 11.5",
        #"Environment :: GPU :: NVIDIA CUDA :: 11.6",  # PyPI has not added it yet
    ],
    ext_modules=cythonize([
        custatevec,
        cutensornet,
        ], verbose=True, language_level=3,
        compiler_directives={'embedsignature': True}),
    packages=find_packages(include=['cuquantum', 'cuquantum.*']),
    package_data={"": ["*.pxd", "*.pyx", "*.py"],},
    zip_safe=False,
    python_requires='>=3.7',
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=install_requires + [
        # pytest < 6.2 is slow in collecting tests
        'pytest>=6.2',
        #'cffi>=1.0.0',  # optional
    ]
)
