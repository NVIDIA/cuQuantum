import os
import site
import subprocess
import sys

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


# search order:
# 1. installed "cuquantum" package
# 2. env var
for path in site.getsitepackages():
    path = os.path.join(path, 'cuquantum')
    if os.path.isdir(path):
        cuquantum_root = path
        break
else:
    cuquantum_root = os.environ.get('CUQUANTUM_ROOT')


# We allow setting CUSTATEVEC_ROOT and CUTENSORNET_ROOT separately for the ease
# of development, but users are encouraged to either install cuquantum from PyPI
# or conda, or set CUQUANTUM_ROOT to the existing installation.
try:
    custatevec_root = os.environ['CUSTATEVEC_ROOT']
except KeyError as e:
    if cuquantum_root is None:
        raise RuntimeError('cuStateVec is not found, please install "cuquantum" '
                           'or set $CUQUANTUM_ROOT') from e
    else:
        custatevec_root = cuquantum_root
try:
    cutensornet_root = os.environ['CUTENSORNET_ROOT']
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
        break
else:
    cutensor_root = os.environ.get('CUTENSOR_ROOT')
if cutensor_root is None:
    raise RuntimeError('cuTENSOR is not found, please install "cutensor" '
                       'or set $CUTENSOR_ROOT')


# We can't assume users to have CTK installed via pip, so we really need this...
# TODO(leofang): try /usr/local/cuda?
try:
    cuda_path = os.environ['CUDA_PATH']
except KeyError as e:
    raise RuntimeError('CUDA is not found, please set $CUDA_PATH') from e


setup_requires = [
    'Cython>=0.29.22,<3',
    ]
install_requires = [
    'numpy',
    # 'cupy', # <-- can't be listed here as on PyPI this is the name for source build, not for wheel
    # 'torch', # <-- PyTorch is optional; also, it does not live on PyPI...
    ]
ignore_cuquantum_dep = bool(os.environ.get('CUQUANTUM_IGNORE_SOLVER', False))
if not ignore_cuquantum_dep:
    setup_requires.append('cuquantum==0.0.1.*')
    setup_requires.append('cutensor>=1.4.*')
    install_requires.append('cuquantum==0.0.1.*')
    install_requires.append('cutensor>=1.4.*')


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


print()
print("****************************************************************")
print("CUDA version:", cuda_ver)
print("CUDA path:", cuda_path)
print("cuStateVec path:", custatevec_root)
print("cuTensorNet path:", cutensornet_root)
print("****************************************************************\n")


custatevec = Extension(
    "cuquantum.custatevec.custatevec",
    sources=["cuquantum/custatevec/custatevec.pyx"],
    include_dirs=[os.path.join(cuda_path, 'include'),
                  os.path.join(custatevec_root, 'include')],
    library_dirs=[os.path.join(custatevec_root, 'lib64')],
    libraries=['custatevec'],
)


cutensornet = Extension(
    "cuquantum.cutensornet.cutensornet",
    sources=["cuquantum/cutensornet/cutensornet.pyx"],
    include_dirs=[os.path.join(cuda_path, 'include'),
                  os.path.join(cutensornet_root, 'include')],
    library_dirs=[os.path.join(cutensornet_root, 'lib64'),
                  os.path.join(cutensor_root, 'lib', cutensor_ver)],
    libraries=['cutensornet', 'cutensor'],
)


setup(
    name="cuquantum-python",
    version='0.1.0.0',  # the last digit is dedicated to cuQuantum Python
    description="Python APIs for cuQuantum",
    url="https://github.com/NVIDIA/cuQuantum",
    author="NVIDIA Corporation",
    author_email="cuquantum-python@nvidia.com",
    license="BSD-3-Clause",
    license_files = ('LICENSE',),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: NVIDIA CUDA :: 11.2",
        "Environment :: GPU :: NVIDIA CUDA :: 11.3",
        "Environment :: GPU :: NVIDIA CUDA :: 11.4",
        "Environment :: GPU :: NVIDIA CUDA :: 11.5",
    ],
    ext_modules=cythonize([
        custatevec,
        cutensornet,
        ], verbose=True, language_level=3,
        compiler_directives={'embedsignature': True}),
    packages=find_packages(include=['cuquantum', 'cuquantum.*']),
    package_data={"": ["*.pxd", "*.pyx", "*.py"],},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=install_requires + [
        # pytest < 6.2 is slow in collecting tests
        'pytest>=6.2',
    ]
)
