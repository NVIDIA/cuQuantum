# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib as _importlib
import os as _os
from typing import TYPE_CHECKING as _TYPE_CHECKING

from cuquantum._version import __version__

# Lazy loading of submodules and top-level attributes.
_PUBLIC_MODULES = ["bindings", "densitymat", "memory", "pauliprop", "stabilizer", "tensornet"]
_LAZY_MODULES = frozenset(_PUBLIC_MODULES)
_LAZY_ATTRS = {
    "BaseCUDAMemoryManager": "cuquantum.memory",
    "MemoryLimitExceeded": "cuquantum.memory",
    "MemoryPointer": "cuquantum.memory",
    "ComputeType": "cuquantum.bindings._utils",
    "cudaDataType": "cuquantum.bindings._utils",
    "libraryPropertyType": "cuquantum.bindings._utils",
}

__all__ = _PUBLIC_MODULES + [*_LAZY_ATTRS.keys()]

if _TYPE_CHECKING:
    from cuquantum import bindings, densitymat, memory, pauliprop, stabilizer, tensornet
    from cuquantum.bindings._utils import ComputeType, cudaDataType, libraryPropertyType
    from cuquantum.memory import BaseCUDAMemoryManager, MemoryLimitExceeded, MemoryPointer


def _eager_import_enabled() -> bool:
    return _os.environ.get("CUQUANTUM_EAGER_IMPORT", "").strip().lower() in ("y", "yes", "t", "true", "on", "1")


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        mod = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    origin = _LAZY_ATTRS.get(name)
    if origin is not None:
        value = getattr(_importlib.import_module(origin), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODULES | _LAZY_ATTRS.keys())


if _eager_import_enabled():
    for _name in sorted(_LAZY_MODULES | _LAZY_ATTRS.keys()):
        __getattr__(_name)
    del _name
