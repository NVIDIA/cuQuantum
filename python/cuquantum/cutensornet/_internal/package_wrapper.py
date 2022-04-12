# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Create map from package names to package interface objects.
"""

__all__ = ['PACKAGE']

from .package_ifc_cupy import CupyPackage

PACKAGE = {'cupy': CupyPackage}
try:
    import torch
    from .package_ifc_torch import TorchPackage
    PACKAGE['torch'] = TorchPackage
except ImportError as e:
    pass

