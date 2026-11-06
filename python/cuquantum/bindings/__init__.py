# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING

from cuquantum import _eager_import_enabled

# Lazy loading of submodules.
__all__ = [
    "cudensitymat",
    "custatevec",
    "custatevecEx",
    "cutensornet",
    "cupauliprop",
    "custabilizer",
]
# Preserve access to Cython glue modules without adding them to wildcard exports.
_LAZY_MODULES = frozenset(__all__) | frozenset(f"cy{name}" for name in __all__)

if _TYPE_CHECKING:
    from cuquantum.bindings import cudensitymat, cupauliprop, custabilizer, custatevec, custatevecEx, cutensornet


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        mod = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODULES)


if _eager_import_enabled():
    for _name in sorted(_LAZY_MODULES):
        __getattr__(_name)
    del _name
