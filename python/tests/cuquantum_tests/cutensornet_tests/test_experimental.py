# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import dataclasses

import cupy
import numpy
import opt_einsum as oe
import pytest

from cuquantum import tensor, OptimizerInfo
from cuquantum.cutensornet.experimental import contract_decompose, ContractDecomposeAlgorithm, ContractDecomposeInfo
from cuquantum.cutensornet.experimental._internal.utils import is_gate_split
from cuquantum.cutensornet._internal.decomposition_utils import DECOMPOSITION_DTYPE_NAMES, parse_decomposition
from cuquantum.cutensornet._internal.utils import infer_object_package

from .approxTN_utils import split_contract_decompose, tensor_decompose, verify_split_QR, verify_split_SVD
from .data import backend_names, contract_decompose_expr
from .test_options import _OptionsBase
from .test_utils import DecomposeFactory, deselect_contract_decompose_algorithm_tests, deselect_decompose_tests, gen_rand_svd_method


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
    "decompose_expr", contract_decompose_expr
)
class TestContractDecompose:
    
    def _run_contract_decompose(self, decompose_expr, xp, dtype, order, stream, algorithm):
        if isinstance(decompose_expr, list):
            decompose_expr, options, optimize, kwargs = decompose_expr
        else:
            options, optimize, kwargs = {}, {}, {}
        return_info = kwargs.get('return_info', True)
        kwargs['return_info'] = return_info

        factory = DecomposeFactory(decompose_expr)
        operands = factory.generate_operands(factory.input_shapes, xp, dtype, order)
        backend = sys.modules[infer_object_package(operands[0])]

        contract_expr, decomp_expr = split_contract_decompose(decompose_expr)
        _, input_modes, output_modes, _, _, _,  max_mid_extent= parse_decomposition(decompose_expr, *operands)
        if not is_gate_split(input_modes, output_modes, algorithm):
            if algorithm.qr_method is not False and algorithm.svd_method is not False: # QR assisted contract SVD decomposition
                pytest.skip("QR assisted SVD decomposition not support for more than three operands")

        shared_mode_out = (set(output_modes[0]) & set(output_modes[1])).pop()
        shared_mode_idx_left = output_modes[0].index(shared_mode_out)
        shared_mode_idx_right = output_modes[1].index(shared_mode_out)

        if stream:
            if backend is numpy:
                stream = cupy.cuda.Stream()
            else:
                stream = backend.cuda.Stream()
        outputs = contract_decompose(decompose_expr, *operands, 
            algorithm=algorithm, stream=stream, options=options, optimize=optimize, **kwargs)

        if stream:
            stream.synchronize()

        #NOTE: The reference here is based on splitting the contract_decompose problem into two sub-problems
        #       - 1. contraction. The reference is based on opt_einsum contract
        #       - 2. decomposition. The reference is based on tensor_decompose in approxTN_utils
        # note that a naive reference implementation here may not find the optimal reduce extent, for example:
        # A[x,y] B[y,z] with input extent x=4, y=2, z=4 -> contract QR decompose -> A[x,k]B[k,z] . 
        # When naively applying the direct algorithm above, the mid extent k in the output will be 2.
        # This case is already consider in contract_decompose. Here make following modifications for correctness testing
        # For contract and QR decompose, we check the output extent is correct
        # For contract and SVD decompose, we inject this mid_extent in the args to the reference implementation when needed.
        intm = oe.contract(contract_expr, *operands)

        if algorithm.svd_method is False:
            if return_info:
                q, r, info = outputs
                assert isinstance(info, ContractDecomposeInfo)
            else:
                q, r = outputs
            assert type(q) is type(r)
            assert type(q) is type(operands[0])
            assert q.shape[shared_mode_idx_left] == max_mid_extent
            assert r.shape[shared_mode_idx_right] == max_mid_extent
            assert verify_split_QR(decomp_expr, intm, q, r, None, None)
        else:
            svd_kwargs = dataclasses.asdict(algorithm.svd_method)
            max_extent = svd_kwargs.get('max_extent')
            if max_extent in [0, None] or max_extent > max_mid_extent:
                svd_kwargs['max_extent'] = max_mid_extent
            outputs_ref = tensor_decompose(decomp_expr, intm, method="svd", return_info=return_info, **svd_kwargs)
            if return_info:
                u, s, v, info = outputs
                assert isinstance(info, ContractDecomposeInfo)
                u_ref, s_ref, v_ref, info_ref = outputs_ref 
                info = info.svd_info
                assert isinstance(info, tensor.SVDInfo)
                info =  dataclasses.asdict(info)
            else:
                u, s, v = outputs
                u_ref, s_ref, v_ref = outputs_ref
                info = info_ref = None
            assert type(u) is type(v)
            assert type(u) is type(operands[0])
            if algorithm.svd_method.partition is None:
                assert type(u) is type(s)
            else:
                assert s is None
            assert verify_split_SVD(decomp_expr, 
                                    intm, 
                                    u, s, v, 
                                    u_ref, s_ref, v_ref,
                                    info=info,
                                    info_ref=info_ref,
                                    **svd_kwargs)


    def test_contract_qr_decompose(self, decompose_expr, xp, dtype, order, stream):
        algorithm = ContractDecomposeAlgorithm(qr_method={}, svd_method=False)
        self._run_contract_decompose(decompose_expr, xp, dtype, order, stream, algorithm)

    
    @pytest.mark.parametrize(
        "svd_method_seed", (None, 0, 1, 2)
    )
    def test_contract_svd_decompose(self, decompose_expr, xp, dtype, order, stream, svd_method_seed):
        svd_method = gen_rand_svd_method(seed=svd_method_seed)
        algorithm = ContractDecomposeAlgorithm(qr_method=False, svd_method=svd_method)
        self._run_contract_decompose(decompose_expr, xp, dtype, order, stream, algorithm)

    
    @pytest.mark.parametrize(
        "svd_method_seed", (None, 0, 1, 2)
    )
    def test_contract_qr_assisted_svd_decompose(self, decompose_expr, xp, dtype, order, stream, svd_method_seed):
        svd_method = gen_rand_svd_method(seed=svd_method_seed)
        algorithm = ContractDecomposeAlgorithm(qr_method={}, svd_method=svd_method)
        self._run_contract_decompose(decompose_expr, xp, dtype, order, stream, algorithm)


class TestContractDecomposeAlgorithm(_OptionsBase):

    options_type = ContractDecomposeAlgorithm
    
    @pytest.mark.uncollect_if(func=deselect_contract_decompose_algorithm_tests)
    @pytest.mark.parametrize(
        'svd_method', [False, {}, tensor.SVDMethod()]
    )
    @pytest.mark.parametrize(
        'qr_method', [False, {}]
    )
    def test_contract_decompose_algorithm(self, qr_method, svd_method):
        self.create_options({'qr_method': qr_method, 'svd_method': svd_method})


class TestContractDecomposeInfo(_OptionsBase):

    options_type = ContractDecomposeInfo

    # Not all fields are optional so we test them all at once
    @pytest.mark.uncollect_if(func=deselect_contract_decompose_algorithm_tests)
    @pytest.mark.parametrize(
        'optimizer_info', [None, OptimizerInfo(largest_intermediate=100.0,
                                        opt_cost=100.0,
                                        path=[(0, 1), (0, 1)],
                                        slices=[("a", 4), ("b", 3)],
                                        num_slices=10,
                                        intermediate_modes=[(1, 3), (2, 4)])]
    )
    @pytest.mark.parametrize(
        'svd_info', [None, tensor.SVDInfo(reduced_extent=2, full_extent=4, discarded_weight=0.01)]
    )
    @pytest.mark.parametrize(
        'svd_method', [False, {}, tensor.SVDMethod()]
    )
    @pytest.mark.parametrize(
        'qr_method', [False, {}]
    )
    def test_contract_decompose_info(self, qr_method, svd_method, svd_info, optimizer_info):
        self.create_options({
            "qr_method": qr_method,
            "svd_method": svd_method,
            "svd_info": svd_info,
            "optimizer_info": optimizer_info,
        })