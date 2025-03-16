# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for tensor network contraction and decomposition.
"""

__all__ = ['ContractDecomposeAlgorithm', 'ContractDecomposeInfo', 'MPSConfig', 'TNConfig']

import dataclasses
import re
from typing import Dict, Optional, Union, Literal

import numpy as np

from cuquantum.bindings import cutensornet as cutn
from ._internal.network_state_utils import MPO_OPTION_MAP, MPS_STATE_ATTRIBUTE_MAP, GAUGE_OPTION_MAP
from .. import configuration
from ..tensor import QRMethod, SVDMethod, SVDInfo
from ..._internal import utils
from .._internal.decomposition_utils import SVD_ALGORITHM_MAP, NORMALIZATION_MAP


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


@dataclasses.dataclass
class TNConfig:

    """A data class for contraction based tensor network simulation configuration that can be provided to the :class:`NetworkState` object.

    Attributes:
        num_hyper_samples (int) : The number of hyper samples for the state contraction optimizer.
    """

    num_hyper_samples : Optional[int] = None

    def __post_init__(self):
        if self.num_hyper_samples is not None:
            if not isinstance(self.num_hyper_samples, int) or self.num_hyper_samples < 0:
                message = f"Invalid value ({self.num_hyper_samples}) for 'num_hyper_samples'. Expect positive integer or None"
                raise ValueError(message)
    
    def _configure_into_state(self, handle, state):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            attr = getattr(cutn.StateAttribute, 'CONFIG_' + field.name.upper())
            dtype = cutn.state_get_attribute_dtype(attr)
            value = np.asarray(value, dtype=dtype)
            cutn.state_configure(handle, state, attr, value.ctypes.data, value.dtype.itemsize)
        return


@dataclasses.dataclass
class MPSConfig:
    """A data class for MPS based tensor network simulation configuration that can be provided to the :class:`NetworkState` object.

    Attributes:
        max_extent (int) : The maximal extent for truncation. If not provided, no extent truncation will be performed.
        canonical_center (int) : The canonical center for the final MPS. If not provided, no canonicalization will be performed.
        abs_cutoff (float) : The absolute value cutoff for MPS SVD truncation. If not provided, no truncation will be performed.
        rel_cutoff (float) : The relative value cutoff for MPS SVD truncation. If not provided, no truncation will be performed.
        normalization (str) : The normalization option for MPS SVD operations. If not provided, no normalization will be performed.
        discarded_weight_cutoff (float) : The discarded weight cutoff for MPS SVD truncation. If not provided, no truncation will be performed.
        algorithm (str) : The SVD algorithm for the MPS SVD computation. It can be ``"gesvd"`` (default), ``"gesvdj"``, ``"gesvdp"`` or ``"gesvdr"``.
        mpo_application (str) : The option for MPS-MPO operations. It can be ``"approximate"`` (default) or ``"exact"``.
        gauge_option (str) : The option for MPS gauge. It can be ``"free"`` (default) which consider no gauging or ``"simple"`` which uses simple update algorithm to compute gauges.
        gesvdj_tol (float) : The tolerance for gesvdj computation.
        gesvdj_max_sweeps (int) : The maximal number of sweeps for gesvdj computation.
        gesvdr_oversampling (int) : The oversampling parameter for gesvdr computation.
        gesvdr_niters (int) : The number of iterations for gesvdr computation.
    """
    max_extent : Optional[int] = None
    canonical_center : Optional[int] = None
    abs_cutoff : Optional[float] = None
    rel_cutoff : Optional[float] = None
    normalization : Optional[str] = None
    discarded_weight_cutoff : Optional[float] = None
    algorithm : Optional[str] = None
    mpo_application : Optional[str] = None
    gauge_option : Optional[str] = None
    gesvdj_tol : Optional[float] = 0
    gesvdj_max_sweeps : Optional[int] = 0
    gesvdr_oversampling : Optional[int] = 0
    gesvdr_niters : Optional[int] = 0

    def __post_init__(self):
        if self.algorithm is not None:
            assert self.algorithm in SVD_ALGORITHM_MAP
        if self.mpo_application is not None:
            assert self.mpo_application in MPO_OPTION_MAP
        if self.gauge_option is not None:
            assert self.gauge_option in GAUGE_OPTION_MAP
        if self.normalization is not None:
            assert self.normalization in NORMALIZATION_MAP
    
    def _configure_into_state(self, handle, state):
        algorithm = 'gesvd'
        for key, attr in MPS_STATE_ATTRIBUTE_MAP.items():
            value = getattr(self, key)
            if value is None:
                continue
            dtype = cutn.state_get_attribute_dtype(attr)
            if key == 'normalization':
                value = NORMALIZATION_MAP[value]
            elif key == 'algorithm':
                algorithm = value
                value = SVD_ALGORITHM_MAP[value]
            elif key == 'mpo_application':
                value = MPO_OPTION_MAP[value]
            elif key == 'gauge_option':
                value = GAUGE_OPTION_MAP[value]
            value = np.asarray(value, dtype=dtype)
            cutn.state_configure(handle, state, attr, value.ctypes.data, value.dtype.itemsize)

        if algorithm in ('gesvdj', 'gesvdr'):
            dtype = cutn.tensor_svd_algo_params_get_dtype(SVD_ALGORITHM_MAP[algorithm])
            algo_params = np.zeros(1, dtype=dtype)
            for name in dtype.names:
                value = getattr(self, f'{algorithm}_{name}')
                if value != 0:
                    algo_params[name] = value
            cutn.state_configure(handle, state, cutn.StateAttribute.CONFIG_MPS_SVD_ALGO_PARAMS, algo_params.ctypes.data, algo_params.dtype.itemsize)
        return

    def _is_fixed_extent_truncation(self):
        return self.abs_cutoff is None and self.rel_cutoff is None and self.discarded_weight_cutoff is None