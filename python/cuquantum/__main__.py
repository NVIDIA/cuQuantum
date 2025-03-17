# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import site
import sys


def get_lib_path(name):
    """Get the loaded shared library path."""
    # Ideally we should call dl_iterate_phdr or dladdr to do the job, but this
    # is simpler and not bad; the former two are not strictly portable anyway
    # (not part of POSIX). Obviously this only works on Linux!

    # We have switched to use dlopen, force library loading via internal API
    if "custatevec" in name:
        from cuquantum import bindings
        bindings._internal.custatevec._inspect_function_pointers()
    elif "cutensor" in name:  # cutensor or cutensornet
        from cuquantum import bindings
        bindings._internal.cutensornet._inspect_function_pointers()
    elif "cudensitymat" in name:
        from cuquantum import bindings
        bindings._internal.cudensitymat._inspect_function_pointers()

    try:
        with open('/proc/self/maps') as f:
            lib_map = f.read()
    except FileNotFoundError as e:
        raise NotImplementedError("This utility is available only on Linux.") from e
    lib = set()
    for line in lib_map.split('\n'):
        if name in line:
            fields = line.split()
            lib.add(fields[-1])  # pathname is the last field, check "man proc"
    if len(lib) == 0:
        raise ValueError(f"library {name} is not loaded")
    elif len(lib) > 1:
        # This could happen when, e.g., a library exists in both the user env
        # and LD_LIBRARY_PATH, and somehow both copies get loaded. This is a
        # messy problem, but let's work around it by assuming the one in the
        # user env is preferred.
        lib2 = set()
        for s in [site.getusersitepackages()] + site.getsitepackages():
            for path in lib:
                if path.startswith(s):
                    lib2.add(path)
        if len(lib2) != 1:
            raise RuntimeError(f"cannot find the unique copy of {name}: {lib}")
        else:
            lib = lib2
    return lib.pop()


def _get_cuquantum_libs():
    paths = set()
    for lib in ('custatevec', 'cutensornet', 'cutensor', 'cudensitymat'):
        path = os.path.normpath(get_lib_path(f"lib{lib}.so"))
        paths.add(path)
    return tuple(paths)


def _get_cuquantum_includes():
    paths = set()
    for path in _get_cuquantum_libs():
        path = os.path.normpath(os.path.join(os.path.dirname(path), '..'))
        if not os.path.isdir(os.path.join(path, 'include')):
            path = os.path.normpath(os.path.join(path, '../include'))
        else:
            path = os.path.join(path, 'include')
        assert os.path.isdir(path), f"path={path} is invalid"
        paths.add(path)
    return tuple(paths)


def _get_cuquantum_target(target):
    target = f"lib{target}.so"
    libs = [os.path.basename(lib) for lib in _get_cuquantum_libs()]
    for lib in libs:
        if target in lib:
            lib = '.'.join(lib.split('.')[:3])  # keep SONAME
            flag = f"-l:{lib} "
            break
    else:
        assert False
    return flag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--includes', action='store_true',
                        help='get cuQuantum include flags')
    parser.add_argument('--libs', action='store_true',
                        help='get cuQuantum linker flags')
    parser.add_argument('--target', action='append', default=[],
                        choices=('custatevec', 'cutensornet', 'cudensitymat'),
                        help='get the linker flag for the target cuQuantum component')
    args = parser.parse_args()

    if not sys.argv[1:]:
        parser.print_help()
        sys.exit(1)
    if args.includes:
        out = ' '.join(f"-I{path}" for path in _get_cuquantum_includes())
        print(out, end=' ')
    if args.libs:
        paths = set([os.path.dirname(path) for path in _get_cuquantum_libs()])
        out = ' '.join(f"-L{path}" for path in paths)
        print(out, end=' ')
    flag = ''
    for target in args.target:
        flag += _get_cuquantum_target(target)
        if target in ('cutensornet', 'cudensitymat') :
            flag += _get_cuquantum_target('cutensor')
    print(flag)
