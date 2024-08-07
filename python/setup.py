# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import atexit
import glob
import os
import shutil
import sys
import tempfile

from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

# this is tricky: sys.path gets overwritten at different stages of the build
# flow, so we need to hack sys.path ourselves...
source_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(source_root, 'builder'))
import utils  # this is builder.utils


# Use README for the project long description
with open(os.path.join(source_root, "README.md")) as f:
    long_description = f.read()


# Get test requirements
with open(os.path.join(source_root, "tests/requirements.txt")) as f:
    tests_require = f.read().split('\n')


# Runtime dependencies
# - cuTENSOR version is constrained in the cutensornet-cuXX package, so we don't
#   need to list it
install_requires = [
    'numpy>=1.21, <3.0',  # ">=1.21,<3"
    # 'torch', # <-- PyTorch is optional; also, the PyPI version does not support GPU...
    f'custatevec-cu{utils.cuda_major_ver}~=1.6',   # ">=1.6.0,<2"
    f'cutensornet-cu{utils.cuda_major_ver}>=2.5.0,<3',
]
if utils.cuda_major_ver == '11':
    install_requires.append('cupy-cuda11x>=13.0')  # no ambiguity
elif utils.cuda_major_ver == '12':
    install_requires.append('cupy-cuda12x>=13.0')  # no ambiguity


# WAR: Check if this is still valid
# TODO: can this support cross-compilation?
if sys.platform == 'linux':
    src_files = glob.glob('**/**/_internal/*_linux.pyx')
elif sys.platform == 'win32':
    src_files = glob.glob('**/**/_internal/*_windows.pyx')
else:
    raise RuntimeError(f'platform is unrecognized: {sys.platform}')
dst_files = []
for src in src_files:
    # Set up a temporary file; it must be under the cache directory so
    # that atomic moves within the same filesystem can be guaranteed
    with tempfile.NamedTemporaryFile(delete=False, dir='.') as f:
        shutil.copy2(src, f.name)
        f_name = f.name
    dst = src.replace('_linux', '').replace('_windows', '')
    # atomic move with the destination guaranteed to be overwritten
    os.replace(f_name, f"./{dst}")
    dst_files.append(dst)


@atexit.register
def cleanup_dst_files():
    for dst in dst_files:
        try:
            os.remove(dst)
        except FileNotFoundError:
            pass


# Note: the extension attributes are overwritten in build_extension()
ext_modules = [
    Extension(
        "cuquantum.custatevec.custatevec",
        sources=["cuquantum/custatevec/custatevec.pyx"],
        language="c++",
    ),
    Extension(
        "cuquantum.custatevec.cycustatevec",
        sources=["cuquantum/custatevec/cycustatevec.pyx"],
        language="c++",
    ),
    Extension(
        "cuquantum.custatevec._internal.custatevec",
        sources=["cuquantum/custatevec/_internal/custatevec.pyx"],
        language="c++",
    ),
    Extension(
        "cuquantum.cutensornet.cutensornet",
        sources=["cuquantum/cutensornet/cutensornet.pyx"],
        language="c++",
    ),
    Extension(
        "cuquantum.cutensornet.cycutensornet",
        sources=["cuquantum/cutensornet/cycutensornet.pyx"],
        language="c++",
    ),
    Extension(
        "cuquantum.cutensornet._internal.cutensornet",
        sources=["cuquantum/cutensornet/_internal/cutensornet.pyx"],
        language="c++",
    ),
    Extension(
        "cuquantum._utils",
        sources=["cuquantum/_utils.pyx"],
        include_dirs=[os.path.join(utils.cuda_path, 'include')],
        language="c++",
    ),
]


cmdclass = {
    'build_ext': utils.build_ext,
    'bdist_wheel': utils.bdist_wheel,
}

cuda_classifier = []
if utils.cuda_major_ver == '11':
    cuda_classifier.append("Environment :: GPU :: NVIDIA CUDA :: 11")
elif utils.cuda_major_ver == '12':
    cuda_classifier.append("Environment :: GPU :: NVIDIA CUDA :: 12")

# TODO: move static metadata to pyproject.toml
setup(
    name=f"cuquantum-python-cu{utils.cuda_major_ver}",
    version=utils.cuqnt_py_ver,
    description="NVIDIA cuQuantum Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://developer.nvidia.com/cuquantum-sdk",
    project_urls={
        "Bug Tracker": "https://github.com/NVIDIA/cuQuantum/issues",
        "User Forum": "https://github.com/NVIDIA/cuQuantum/discussions",
        "Documentation": "https://docs.nvidia.com/cuda/cuquantum/latest/python/",
        "Source Code": "https://github.com/NVIDIA/cuQuantum",
    },
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Environment :: GPU :: NVIDIA CUDA",
    ] + cuda_classifier,
    ext_modules=cythonize(ext_modules,
        verbose=True, language_level=3,
        compiler_directives={'embedsignature': True}),
    packages=find_packages(include=['cuquantum', 'cuquantum.*']),
    package_data=dict.fromkeys(
        find_packages(include=["cuquantum.*"]),
        ["*.pxd", "*.pyx", "*.py"],
    ),
    zip_safe=False,
    python_requires='>=3.10',
    install_requires=install_requires,
    tests_require=install_requires+tests_require,
    cmdclass=cmdclass,
)
