# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
import sys

import cupy
import numpy
import pytest

from cuquantum import tensor
from cuquantum.cutensornet._internal.decomposition_utils import DECOMPOSITION_DTYPE_NAMES
from cuquantum.cutensornet._internal.utils import infer_object_package

from .approxTN_utils import tensor_decompose, verify_split_QR, verify_split_SVD, SingularValueDegeneracyError
from .data import backend_names, tensor_decomp_expressions
from .test_options import _OptionsBase, TestNetworkOptions
from .test_utils import DecomposeFactory
from .test_utils import deselect_decompose_tests, get_svd_methods_for_test
from .test_utils import get_stream_for_backend


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
    "decompose_expr", tensor_decomp_expressions
)
@pytest.mark.parametrize(
    "blocking", (True, "auto")
)
class TestDecompose:
    
    def _run_decompose(
            self, decompose_expr, xp, dtype, order, stream, method, **kwargs):
        decompose_expr, shapes = copy.deepcopy(decompose_expr)
        factory = DecomposeFactory(decompose_expr, shapes=shapes)
        operand = factory.generate_operands(factory.input_shapes, xp, dtype, order)[0]
        backend = sys.modules[infer_object_package(operand)]

        if stream:
            stream = get_stream_for_backend(backend)
        
        return_info = kwargs.pop("return_info", False)        
        outputs = tensor.decompose(decompose_expr, 
                                   operand, 
                                   method=method,
                                   options={"blocking": kwargs["blocking"]},
                                   stream=stream,
                                   return_info=return_info)
        if stream:
            stream.synchronize()
        
        if isinstance(method, tensor.QRMethod):
            q, r = outputs
            assert type(q) is type(r)
            assert type(q) is type(operand)
            assert verify_split_QR(decompose_expr, operand, q, r, None, None)
        elif isinstance(method, tensor.SVDMethod):
            svd_kwargs = dataclasses.asdict(method)
            try:
                outputs_ref = tensor_decompose(decompose_expr, operand, method="svd", return_info=return_info, **svd_kwargs)
            except SingularValueDegeneracyError:
                pytest.skip("Test skipped due to singular value degeneracy issue")
            if return_info:
                u, s, v, info = outputs
                u_ref, s_ref, v_ref, info_ref = outputs_ref
                assert isinstance(info, tensor.SVDInfo)
                info = dataclasses.asdict(info)
            else:
                u, s, v = outputs
                u_ref, s_ref, v_ref = outputs_ref
                info = {'algorithm': method.algorithm}
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
    
    def test_qr(self, decompose_expr, xp, dtype, order, stream, blocking):
        self._run_decompose(
            decompose_expr, xp, dtype, order, stream, tensor.QRMethod(),
            blocking=blocking)
    
    @pytest.mark.parametrize(
        "return_info", (False, True)
    )
    def test_svd(
            self, decompose_expr, xp, dtype, order, stream, return_info, blocking):
        methods = get_svd_methods_for_test(3, dtype)
        for method in methods:
            self._run_decompose(
                decompose_expr, xp, dtype, order, stream, method,
                blocking=blocking, return_info=return_info)


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
    
    def test_discarded_weight_cutoff(self):
        self.create_options({'discarded_weight_cutoff': 0.1})
    
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

    @pytest.mark.parametrize(
        'algorithm', ['gesvd', 'gesvdj', 'gesvdp', 'gesvdr']
    )
    def test_algorithm(self, algorithm):
        options = {'algorithm': algorithm}
        if algorithm == 'gesvdj':
            options['gesvdj_tol'] = 1e-16
            options['gesvdj_max_sweeps'] = 80
        elif algorithm == 'gesvdr':
            options['gesvdr_oversampling'] = 4
            options['gesvdr_niters'] = 8
        self.create_options(options)


class TestSVDInfo(_OptionsBase):

    options_type = tensor.SVDInfo

    # All fields are required. Therefore we test them all at once.
    @pytest.mark.parametrize(
        'algorithm', ['gesvd', 'gesvdj', 'gesvdp', 'gesvdr']
    )
    def test_svd_info(self, algorithm):
        info = {'reduced_extent': 6, 'full_extent': 8, 'discarded_weight': 0.02, 'algorithm': algorithm}
        if algorithm == 'gesvdj':
            info['gesvdj_sweeps'] = 12
            info['gesvdj_residual'] = 1e-12
        elif algorithm == 'gesvdp':
            info['gesvdp_err_sigma'] = 1e-8
        self.create_options(info)
