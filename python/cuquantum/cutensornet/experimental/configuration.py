# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for tensor network contraction and decomposition.
"""

__all__ = ['ContractDecomposeAlgorithm', 'ContractDecomposeInfo']

import dataclasses
import re
from typing import Dict, Optional, Union, Literal

from .. import configuration
from ..tensor import QRMethod, SVDMethod, SVDInfo
from .._internal import utils


@dataclasses.dataclass
class ContractDecomposeAlgorithm:

    """A data class for specifying the algorithm to use for the contract and decompose operations. 
    Three choices are supported, as listed below:

        #. When ``svd_method`` is `False` while ``qr_method`` is not `False` (default), 
           this amounts to direct contraction of the tensor network followed by a QR decomposition.
        #. When ``qr_method`` is `False` while ``svd_method`` is not `False`, 
           this amounts to direct contraction of the tensor network followed by a singular value decomposition.
        #. When neither ``qr_method`` and ``svd_method`` is `False`, 
           this amounts to QR-assisted contraction with singular value decomposition. 
           QR decomposition will first be applied onto certain input tensors to reduce the network size. 
           The resulting R tensors along with the remaining tensors form a new network that will be contracted and decomposed using SVD. 
           The Q tensors from the first QR operations along with the SVD outputs are then subject to two more contractions to yield the final output.

    .. note::
        The third choice above (QR-assisted contract and SVD) is currently only supported for ternary operands that are fully connected to each other with un-contracted modes on each tensor.
        The results from the third choice is expected to be equivalent to that from the second choice but typically at lower computational cost.


    Attributes:
        qr_method: The QR method used for the decomposition. See :class:`~cuquantum.cutensornet.tensor.QRMethod`.
        svd_method: The SVD method used for the decomposition. See :class:`~cuquantum.cutensornet.tensor.SVDMethod`.
        svd_info: The SVD information during runtime. See :class:`~cuquantum.cutensornet.tensor.SVDInfo`.
    """

    qr_method: Optional[Union[QRMethod, Literal[False, None],Dict]] = dataclasses.field(default_factory=QRMethod)
    svd_method: Optional[Union[SVDMethod,Literal[False, None],Dict]] = False
    
    def __post_init__(self):
        if self.qr_method is False and self.svd_method is False:
            raise ValueError("Must specify at least one of the qr_method or svd_method")
        
        if self.qr_method is not False:
            self.qr_method = utils.check_or_create_options(QRMethod, self.qr_method, "QR Method")
        
        if self.svd_method is not False:
            self.svd_method = utils.check_or_create_options(SVDMethod, self.svd_method, "SVD Method")


@dataclasses.dataclass
class ContractDecomposeInfo:

    """A data class for capturing contract-decompose information.

    Attributes:
        qr_method: The QR method used for the decomposition. See :class:`~cuquantum.cutensornet.tensor.QRMethod`.
        svd_method: The SVD method used for the decomposition. See :class:`~cuquantum.cutensornet.tensor.SVDMethod`.
        svd_info: The SVD information during runtime. See :class:`~cuquantum.cutensornet.tensor.SVDInfo`.
        optimizer_info: The information for the contraction path to form the intermediate tensor. See :class:`~OptimizerInfo`
    """

    qr_method: Union[QRMethod, Literal[False, None],Dict]
    svd_method: Union[SVDMethod,Literal[False, None],Dict]
    svd_info: Optional[SVDInfo] = None
    optimizer_info: Optional[configuration.OptimizerInfo] = None

    def __str__(self):
        core_method = 'QR' if self.svd_method is False else 'SVD'
        indent = 4
        repr = f"""Contract-Decompose Information:
    Summary of Operations:
        Contraction followed by {core_method} decomposition."""
        if self.svd_method is not False and self.qr_method is not False: # QR-assisted 
            repr += f"""
        Before contraction, QR is applied to reduce the size of the tensor network. Post-decomposition contractions are performed to construct the final outputs."""
        # optimizer info, hack below to match string indentation
        if self.optimizer_info is not None:
            optimizer_info = re.sub(r"\n", fr"\n{' ' * indent}", str(self.optimizer_info))
            repr += f"""
    {optimizer_info}"""
        # svd_info, hack below to match string indentation
        if self.svd_info is not None:
            svd_info = re.sub(r"\n", fr"\n{' ' * indent}", str(self.svd_info))
            repr += f"""
    {svd_info}"""
        return repr
