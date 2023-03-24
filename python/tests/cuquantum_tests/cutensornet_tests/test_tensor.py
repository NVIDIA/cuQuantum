# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys

import cupy
import dataclasses
import numpy
import pytest

from cuquantum import tensor
from cuquantum.cutensornet._internal.decomposition_utils import DECOMPOSITION_DTYPE_NAMES
from cuquantum.cutensornet._internal.utils import infer_object_package

from .approxTN_utils import tensor_decompose, verify_split_QR, verify_split_SVD
from .data import backend_names, tensor_decomp_expressions
from .test_options import _OptionsBase, TestNetworkOptions
from .test_utils import DecomposeFactory
from .test_utils import deselect_decompose_tests, gen_rand_svd_method


@pytest.mark.uncollect_if(func=deselect_decompose_tests)
@pytest.mark.parametrize(
    "stream", (None, True)
)
@pytest.mark.parametrize(
    "order", ("C", "F")
)
@pytest.mark.parametrize(
    "dtype", DECOMPOSITION_DTYPE_NAMES
)
@pytest.mark.parametrize(
    "xp", backend_names
)
@pytest.mark.parametrize(
    "decompose_expr", list(set([expr[0] for expr in tensor_decomp_expressions])) # filter out duplicated expressions
)
class TestDecompose:
    
    def _run_decompose(self, decompose_expr, xp, dtype, order, stream, method, **kwargs):
        factory = DecomposeFactory(decompose_expr)
        operand = factory.generate_operands(factory.input_shapes, xp, dtype, order)[0]
        backend = sys.modules[infer_object_package(operand)]

        if stream:
            if backend is numpy:
                stream = cupy.cuda.Stream()
            else:
                stream = backend.cuda.Stream()
        
        return_info = kwargs.pop("return_info", False)
        outputs = tensor.decompose(decompose_expr, 
                                operand, 
                                method=method, 
                                return_info=return_info, 
                                stream=stream)
        if stream:
            stream.synchronize()
        
        if isinstance(method, tensor.QRMethod):
            q, r = outputs
            assert type(q) is type(r)
            assert type(q) is type(operand)
            assert verify_split_QR(decompose_expr, operand, q, r, None, None)
        elif isinstance(method, tensor.SVDMethod):
            svd_kwargs = dataclasses.asdict(method)
            outputs_ref = tensor_decompose(decompose_expr, operand, method="svd", return_info=return_info, **svd_kwargs)
            if return_info:
                u, s, v, info = outputs
                u_ref, s_ref, v_ref, info_ref = outputs_ref
                assert isinstance(info, tensor.SVDInfo)
                info = dataclasses.asdict(info)
            else:
                u, s, v = outputs
                u_ref, s_ref, v_ref = outputs_ref
                info = None
                info_ref = None
            
            assert type(u) is type(v)
            assert type(u) is type(operand)
            if method.partition is None:
                assert type(u) is type(s)
            else:
                assert s is None

            assert verify_split_SVD(decompose_expr, 
                                    operand, 
                                    u, s, v, 
                                    u_ref, s_ref, v_ref,
                                    info=info,
                                    info_ref=info_ref,
                                    **svd_kwargs)
    
    def test_qr(self, decompose_expr, xp, dtype, order, stream):
        self._run_decompose(decompose_expr, xp, dtype, order, stream, tensor.QRMethod())
    
    @pytest.mark.parametrize(
        "svd_method_seed", (None, 0, 1, 2)
    )
    @pytest.mark.parametrize(
        "return_info", (False, True)
    )
    def test_svd(self, decompose_expr, xp, dtype, order, stream, return_info, svd_method_seed):
        method = gen_rand_svd_method(seed=svd_method_seed)
        self._run_decompose(decompose_expr, xp, dtype, order, stream, method, return_info=return_info)


class TestDecompositionOptions(TestNetworkOptions):

    options_type = tensor.DecompositionOptions


class TestSVDMethod(_OptionsBase):

    options_type = tensor.SVDMethod

    def test_max_extent(self):
        self.create_options({'max_extent': 6})
    
    def test_abs_cutoff(self):
        self.create_options({'abs_cutoff': 0.2})
    
    def test_rel_cutoff(self):
        self.create_options({'rel_cutoff': 0.1})
    
    @pytest.mark.parametrize(
        'partition', [None, 'U', 'V', 'UV']
    )
    def test_partition(self, partition):
        self.create_options({'partition': partition})
    
    @pytest.mark.parametrize(
        'normalization', [None, 'L1', 'L2', 'LInf']
    )
    def test_normalization(self, normalization):
        self.create_options({'normalization': normalization})


class TestSVDInfo(_OptionsBase):

    options_type = tensor.SVDInfo

    # All fields are required. Therefore we test them all at once.
    def test_svd_info(self):
        self.create_options({'reduced_extent': 6, 'full_extent': 8, 'discarded_weight': 0.02})
