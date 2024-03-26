# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import pytest

import cuquantum

from .data import einsum_expressions
from .test_utils import compute_and_normalize_numpy_path
from .test_utils import EinsumFactory
from .test_utils import set_path_to_optimizer_options


@pytest.mark.parametrize(
    "einsum_expr_pack", einsum_expressions
)
class TestContractPath:

    def _test_runner(
            self, func, einsum_expr_pack, use_numpy_path, **kwargs):
        einsum_expr = copy.deepcopy(einsum_expr_pack)
        if isinstance(einsum_expr_pack, list):
            einsum_expr, network_opts, optimizer_opts, overwrite_dtype = einsum_expr
            dtype = overwrite_dtype
        else:
            network_opts = optimizer_opts = None
            dtype = "float32"
        assert isinstance(einsum_expr, (str, tuple))

        factory = EinsumFactory(einsum_expr)
        # backend/dtype/order do not matter, so we just pick one here
        operands = factory.generate_operands(
            factory.input_shapes, "cupy", dtype, "C")

        path = None
        if use_numpy_path:
            try:
                path = compute_and_normalize_numpy_path(
                    factory.convert_by_format(operands, dummy=True),
                    len(operands))
            except NotImplementedError:
                # we can't support the returned NumPy path, just skip
                pytest.skip("NumPy path is either not found or invalid")

        data = factory.convert_by_format(operands)
        try:
            if func is cuquantum.contract_path:
                if path is not None:
                    optimizer_opts = set_path_to_optimizer_options(
                        optimizer_opts, path)
                path, info = func(
                    *data, options=network_opts, optimize=optimizer_opts)
                assert isinstance(path, list)
                assert isinstance(info, cuquantum.OptimizerInfo)
            else:  # cuquantum.einsum_path()
                optimize = kwargs.pop('optimize')
                assert optimize == True
                path, info = func(*data, optimize=optimize)
                assert path[0] == "einsum_path"
                path = path[1:]
        except MemoryError as e:
            if "Insufficient memory" in str(e):
                # not enough memory available to process, just skip
                pytest.skip("Insufficient workspace memory available.")
            else:
                raise

        # sanity checks; the correctness checks are done in the contract() tests
        assert len(path) == len(operands)-1
        operand_ids = list(range(len(operands))) if path else [-1]    # handle single operand case.
        for i, j in path:
            op_i, op_j = operand_ids[i], operand_ids[j]
            operand_ids.remove(op_i)
            operand_ids.remove(op_j)
            operand_ids.append(-1)  # placeholder for intermediate
        # all input tensors are contracted
        assert len(operand_ids) == 1
        assert operand_ids[0] == -1

    @pytest.mark.parametrize(
        "use_numpy_path", (False, True)
    )
    def test_contract_path(self, einsum_expr_pack, use_numpy_path):
        self._test_runner(
            cuquantum.contract_path, einsum_expr_pack, use_numpy_path)

    def test_einsum_path(self, einsum_expr_pack):
        # We only support optimize=True and don't allow setting the path
        self._test_runner(
            cuquantum.einsum_path, einsum_expr_pack, False, optimize=True)
