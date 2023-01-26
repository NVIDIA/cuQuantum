# Demo: Dynamically link to CUDA & cuQuantum wheels

## Introduction

For a long time, downstream developers who wish to build their GPU libraries, frameworks, and products
against CUDA Toolkit (CTK) and any of NVIDIA's GPU libraries suffer from wheel generation
and distribution. Due to a number of conflicting constraints, such as
* the 100 MB default & 1 GB hard limits of PyPI.org uploads,
* [`auditwheel`](https://github.com/pypa/auditwheel) compliance (that wheels must be "self-contained", see below),
* CUDA offering no ABI compatibility,

the community has to build "mega" wheels per CUDA *minor* version (e.g., `cupy-cuda112` for CUDA 11.2,
`torch-1.13.0+cu117` for CUDA 11.7, etc) and upload them to self-hosted (non-PyPI) package repositories, which is a
very laborious process, well-explained in this ["pypackaging"](https://pypackaging-native.github.io/key-issues/gpus/)
community project.

These constraints are conflicting, if not self-contradictory, because
* If a wheel must be self-contained, one either has to use `auditwheel repair` to copy the dependent
shared libraries into the wheel (with the rpath hacked and the symbols obfuscated), or perform *static*
linking. In any case this causes the wheel size to inflate significantly.
* If the wheel is too large, the wheel generation wastes too much (usually limited) CI time and resource,
and, worst of all, PyPI would block the wheel upload after jumping through all the hoops.
* If there is no ABI compatibility, GPU library providers are forced to upload multiple such mega wheels
at once for all supported CUDA versions, which has been causing friction between the PyData/scientific
computing community and the rest of the Python community.

Fortunately, driven in part by NVIDIA, two welcome pieces of news help improve this situation:
1. CUDA minor version compatibility (since CUDA 11) ensures the ABI compatibility across minor versions
(though with [caveats](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility)),
2. `auditwheel --exclude` for allowlisting certain shared libraries for *dynamic* linking ([pypa/auditwheel#368](https://github.com/pypa/auditwheel/pull/368)).

The rest of this demo is to show users and developers how to build their projects to take advantage of
dynamic linking to NVIDIA-distributed wheels, thereby dramatically reducing the wheel build time & size,
while offering better compatibility, faster download/install time, and a nicer user experience.


## Goals

For Python GPU library *developers*:
* to reduce the file size of your Python wheels
* to reduce the build time of generating your Python wheels
* to avoid generating a swarm of Python wheels for each CUDA `major.minor` version

For Python GPU library *users*:
* to enjoy a pure `pip`-based solution, potentially allowing quickly spinning up a working CUDA environment
  (without any pre-installed CTK!)
* to reduce the demands for network bandwidth and storage required by installing GPU packages


## Prerequisite

This demo can be followed in a local environment (either bare metal or container), or in a
CI/CD automation. In particular, `cibuildwheel` users should be able to integrate
key ingredients easily (though for each specific build system, the compiler/linker flags
might need to be passed differently, which is out of scope for this demo).

The only *optional*, external dependency is to have a working CTK installation (CUDA 11+),
if you will compile any CUDA source code (with the `.cu` file extension) using `nvcc`. Otherwise,
a working host compiler (`gcc`, `g++`, ...) supported by CUDA, plus your already working
build system (setuptools, CMake, scikit-build, Meson, ...) is enough.

The utility script `search_package_path.py` requires Python 3.8+, but the same location-probing
logic can be implemented in any scripting languange.

Below we assume you have cloned this repository and `cd`'d to the directory where this README resides.


## Steps

1. Create & activate a Conda (or virtual) environment. Here we use Conda to illustrate.
```
$ conda create -n demo -y -c conda-forge python; conda activate demo
```

2. Specify the target CUDA major version. As of cuQuantum 22.11 we only support CUDA 11.
```
$ export CUDA_MAJOR=11
```
CUDA 12 will be supported in a future release.

3. Install CUDA & cuQuantum wheels from PyPI.org:
```
$ pip install nvidia-cuda-runtime-cu$CUDA_MAJOR \
              nvidia-cublas-cu$CUDA_MAJOR       \
              nvidia-cusolver-cu$CUDA_MAJOR     \
              nvidia-cusparse-cu$CUDA_MAJOR     \
              cuquantum-cu$CUDA_MAJOR
```
This works because
* currently all wheels distributed by NVIDIA have the `-cuXX` suffix, and
* the minor version is (largely) irrelevant since CUDA 11, thanks to CUDA minor version compatibility.

4. Use the utility scripts `search_package_path.py` & `setup.sh` to set up environment variables:
```
$ source setup.sh
```

5. Build your project. For simplicity and illustration purposes, we call `nvcc` from the command
line as an example in this demo, but the basic principle applies to any build system.
```
$ cd ../../samples/cutensornet
$ nvcc -I$CUTENSORNET_INCLUDE \
       -Xlinker "--disable-new-dtags" \
       -Xlinker "--rpath=$CUSOLVER_LIB" \
       -Xlinker "--rpath=$CUSPARSE_LIB" \
       -Xlinker "--rpath=$CUTENSORNET_LIB" \
       -Xlinker "--rpath=$CUDART_LIB" \
       -Xlinker "--no-as-needed,-l:libcusolver.so.$CUDA_MAJOR,-l:libcusparse.so.$CUDA_MAJOR,--as-needed" \
       -L$CUTENSORNET_LIB -l:libcutensornet.so.2 \
       -L$CUDART_LIB -l:libcudart.so.$CUDA_MAJOR.0 \
       ./tensornet_example.cu \
       -o tensornet_example
```

If your project only requires a host compiler (such as `gcc`), you can pass the linker flags
via `-Wl` as opposed to `-Xlinker` that is used by `nvcc`.

6. Confirm all linked CUDA & cuQuantum libraries are coming from the wheels (installed in `site-packages`):
```
$ ldd tensornet_example
	linux-vdso.so.1 (0x00007fff4152a000)
	libcusolver.so.11 => /home/user/miniforge3/envs/demo/lib/python3.9/site-packages/nvidia/cusolver/lib/libcusolver.so.11 (0x00007fab28dcf000)
	libcusparse.so.11 => /home/user/miniforge3/envs/demo/lib/python3.9/site-packages/nvidia/cusparse/lib/libcusparse.so.11 (0x00007fab180d7000)
	libcutensornet.so.2 => /home/user/miniforge3/envs/demo/lib/python3.9/site-packages/cuquantum/lib/libcutensornet.so.2 (0x00007fab17b41000)
	libcudart.so.11.0 => /home/user/miniforge3/envs/demo/lib/python3.9/site-packages/nvidia/cuda_runtime/lib/libcudart.so.11.0 (0x00007fab1789a000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fab17694000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fab17679000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fab17487000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fab17481000)
	libcublas.so.11 => /home/user/miniforge3/envs/demo/lib/python3.9/site-packages/nvidia/cusolver/lib/../../cublas/lib/libcublas.so.11 (0x00007fab11823000)
	libcublasLt.so.11 => /home/user/miniforge3/envs/demo/lib/python3.9/site-packages/nvidia/cusolver/lib/../../cublas/lib/libcublasLt.so.11 (0x00007faaed29b000)
	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007faaed291000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007faaed26e000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007faaed11f000)
	/lib64/ld-linux-x86-64.so.2 (0x00007fab3b0a3000)
	libcutensor.so.1 => /home/user/miniforge3/envs/demo/lib/python3.9/site-packages/cuquantum/lib/../../cutensor/lib/libcutensor.so.1 (0x00007faadfe76000)
```

7. Happy testing!
```
$ ./tensornet_example  # this example requires an actual GPU to run!
```

## How this works?

NVIDIA CTK and cuQuantum SDK are offered as proprietary binaries. To build a project on top of them, users
need to include the headers and link to the binaries using either the host or `nvcc` compiler. The distributed
wheels are basically a binary repackaging of the CTK/cuQuantum SDK, adhering to the Linux path convention:
```
root/
├── include
│   ├── <my_library>.h
|   ├── ...
|
├── lib
|   ├── lib<my_library>.so.X
|   ├── ...
```
with the noteworthy exception that the symlinks (e.g. `lib<my_library>.so` and `lib<my_library>.so.X.Y.Z`)
are absent. So the remaining tasks are
1. to set up the appropriate compiler/linker flags pointing to the right location and targeting the right SONAME, and
2. to modify the rpaths of the generated executables or shared libraries, such as Python extension modules written in C/C++.


## Discussions

All these challenges are unique to Python wheels. For Conda packages, developers (or package maintainers)
do not face these challenges due to Conda's excellent support of mixing Python with non-Python packages.

In Step 3, we installed all wheels from PyPI via plain `pip install`. For CUDA wheels this support is
currently provisional while NVIDIA is exploring better solutions to serve the community.

Note that we had to install the cuSPARSE wheel, and later explicitly link against both cuSPARSE/cuSOLVER
wheels. This is a workaround for cuTensorNet v2.0.0 that will no longer be needed in a future release.

In Step 4, we used the script `setup.sh` (which imports `search_package_path.py`) to probe the wheels'
install locations, so that we can set up the correct compiler/linker flags (for passing the header include
paths and linker search paths, respectively).

In Step 5, we performed the rpath hack manually (instead of letting `auditwheel` repair it) by setting
`--disable-new-dtags` first (otherwise it would be *runpath* being hacked) and then setting `--rpath=...`
so that the executable can inform the dynamic loader where to find the needed shared library at run time.

In practice, it is better to compute the rpath relative to self (`$ORIGIN`), see, e.g., how [cuQuantum
Python approaches this](https://github.com/NVIDIA/cuQuantum/blob/b7f847c16413e114ddc91d415f18d44d24dd3261/python/builder/utils.py#L162-L179).
This way, as long as the relative paths inside `site-packages` are fixed, the packages are relocatable.

Observe the unusual linker flags (e.g. `-l:libcutensornet.so.2` as opposed to `-lcutensornet`, and
`-l:libcudart.so.11.0` as opposed to `-lcudart`).
This is due to a defect in the wheel format (not supporting symlinks) which is better explained by
[the pypackaging authors](https://pypackaging-native.github.io/other_issues/#lack-of-support-for-symlinks-in-wheels)
(with our contribution). The result is that the developer should understand which `SONAME` needs to
be specified at build time. Unfortunately, we are unaware of any better solution.

This approach has been adopted by cuQuantum Python to keep the wheel size well below the 100 MB limit.
Had we performed `auditwheel repair` (without `--exclude`) or static linking, our wheel size would have
far exceeded the limit, as we would have absorbed a substantial portion of the CTK and cuQuantum SDK
into our wheel.


## Next steps, tips, and known issues

In the above steps we built a single executable `tensornet_example` and made it link to the installed
Python wheels. For actual wheel generation, you may wish to declare *run-time* dependencies on the
CTK/cuQuantum wheels. Depending on your build system it can be easily achieved, for example,
* set `insteall_requires` in the `setup()` function from `setup.py`,
* set `dependencies` in the `[project]` table from `pyproject.toml` (PEP 621),
* set `insteall_requires` in the `[options]` string from `setup.cfg`,
* ...

and we won't dive into such build system-dependent details (though see below).

You may also wish to generate different wheels for different CUDA *major* version (e.g. `cupy-cuda11x`
for CUDA 11, `cupy-cuda12x` for CUDA 12, etc) to take advantage of CUDA minor version compatibility.
This can be done by populating the wheel name at build time, though it might not be compatible with
static metadata-based approaches (such as PEP 621).

Other tips include:

* Remember, if you are an `auditwheel` user, you need to use the `--exclude` flag.
* In addition to link-time hacks, the rpaths can also be modified via
  [`patchelf --set-rpath`](https://github.com/NixOS/patchelf) (which is what `auditwheel repair` does
  under the hood).
* Using the environment variable `LD_LIBRARY_PATH` should be avoided, otherwise the linker may become
  confused regarding where to look for the shared libraries (since it takes precedence in the search
  order).
* You can hack the rpath further to support a user-supplied CTK installation (not from PyPI). This could
  be handy if the CTK is installed via a Linux distro, CUDA runfile, or by cluster admins. For serving
  your wheel users better, though, we recommend that you document the expectation of the CTK source
  and communicate it with your users.
  * This is important because as of today there is no way in the wheel world to honor a single source
    of truth for the CTK installation.

* Recall that the standard practice of building a wheel is to use the [manylinux container](https://github.com/pypa/manylinux).
  It would ensure the linked shared libraries are controllable, without accidental pollution.
* For developers who wish to only access individual libraries from the cuQuantum SDK, we also offer
  separate wheels, such as `custatevec-cu11` (for cuStateVec) and `cutensornet-cu11` (for cuTensorNet).
  This allows a clean dependency declaration if a pacakge does not need everything from `cuquantum-cu11`.
