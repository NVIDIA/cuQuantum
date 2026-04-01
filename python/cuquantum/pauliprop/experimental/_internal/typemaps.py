# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cuquantum.bindings.cupauliprop as cupp 
from nvmath.internal.typemaps import *

__all__ = ["PAULI_MAP", "PAULI_MAP_INV", "CLIFFORD_MAP", "COMPUTE_TYPE_TO_NAME", "DATA_TYPE_TO_NAME", "NAME_TO_DATA_TYPE", "NAME_TO_COMPUTE_TYPE", "NAME_TO_DATA_WIDTH"]

class CaseInsensitiveDict(dict):
    """A dictionary that normalizes string keys to uppercase for case-insensitive lookups.

    Keys are stored in uppercase internally, so ``.keys()`` returns uppercase strings.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__()
        if data:
            items = data.items() if isinstance(data, dict) else data
            for k, v in items:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = key.upper()
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.upper()
        return super().__getitem__(key)

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.upper()
        return super().__contains__(key)

    def get(self, key, default=None):
        if isinstance(key, str):
            key = key.upper()
        return super().get(key, default)

    def __delitem__(self, key):
        if isinstance(key, str):
            key = key.upper()
        super().__delitem__(key)


PAULI_MAP = CaseInsensitiveDict({
    "X": cupp.PauliKind.PAULI_X,
    "Y": cupp.PauliKind.PAULI_Y,
    "Z": cupp.PauliKind.PAULI_Z,
    "I": cupp.PauliKind.PAULI_I
})

PAULI_MAP_INV = {
    cupp.PauliKind.PAULI_X: "X",
    cupp.PauliKind.PAULI_Y: "Y",
    cupp.PauliKind.PAULI_Z: "Z",
    cupp.PauliKind.PAULI_I: "I"
}
             
CLIFFORD_MAP = CaseInsensitiveDict({
    "I": cupp.CliffordGateKind.CLIFFORD_GATE_I,
    "X": cupp.CliffordGateKind.CLIFFORD_GATE_X,
    "Y": cupp.CliffordGateKind.CLIFFORD_GATE_Y,
    "Z": cupp.CliffordGateKind.CLIFFORD_GATE_Z,
    "H": cupp.CliffordGateKind.CLIFFORD_GATE_H,
    "S": cupp.CliffordGateKind.CLIFFORD_GATE_S,
    "CX": cupp.CliffordGateKind.CLIFFORD_GATE_CX,
    "CY": cupp.CliffordGateKind.CLIFFORD_GATE_CY,
    "CZ": cupp.CliffordGateKind.CLIFFORD_GATE_CZ,
    "SqrtX": cupp.CliffordGateKind.CLIFFORD_GATE_SQRTX,
    "SqrtY": cupp.CliffordGateKind.CLIFFORD_GATE_SQRTY,
    "SqrtZ": cupp.CliffordGateKind.CLIFFORD_GATE_SQRTZ,
    "SWAP" : cupp.CliffordGateKind.CLIFFORD_GATE_SWAP,
    "iSWAP": cupp.CliffordGateKind.CLIFFORD_GATE_ISWAP
})


# TODO[OPTIONAL]: move this map elsewhere
WORK_SPACE_KIND_MAP = CaseInsensitiveDict({})
WORK_SPACE_KIND_MAP["SCRATCH"] = cupp.WorkspaceKind.WORKSPACE_SCRATCH
# WORK_SPACE_KIND_MAP["CACHE"] = cudm.WorkspaceKind.WORKSPACE_CACHE #Not yet implemented

# TODO[OPTIONAL]: move this map elsewhere
MEM_SPACE_MAP = CaseInsensitiveDict({})
MEM_SPACE_MAP["DEVICE"] = cupp.Memspace.DEVICE
MEM_SPACE_MAP["HOST"] = cupp.Memspace.HOST

MEM_SPACE_MAP_INV = {
    cupp.Memspace.DEVICE: "DEVICE",
    cupp.Memspace.HOST: "HOST"
}