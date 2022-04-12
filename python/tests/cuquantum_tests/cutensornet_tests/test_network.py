# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import itertools
from copy import deepcopy

from .testutils import *


class NetworkProxyFixture(ProxyFixtureBase):
    def __init__(self, network_options_pack):
        super().__init__(network_options_pack)

    def test_contract_path(
        self,
        options_constructor_mode,
        optimize_constructor_mode,
    ):
        network_einsum = network_dispatcher(
            self.einsum_expr,
            self.data_operands,
            self.options,
            mode=options_constructor_mode
        )
        
        network_einsum.contract_path(
            optimize=optimizer_options_dispatcher(
                self.optimize,
                mode=optimize_constructor_mode
            )
        )

        network_einsum.free()

        network_interleaved = network_dispatcher(
            None,
            None,
            self.options,
            mode=options_constructor_mode,
            interleaved_inputs=self.interleaved_inputs
        )

        network_interleaved.contract_path(
            optimize=optimizer_options_dispatcher(
                self.optimize,
                mode=optimize_constructor_mode
            )
        )

        network_interleaved.free()

    def _test_contract(
        self,
        options_constructor_mode,
        optimize_constructor_mode,
        skip_sync,
        use_numpy_einsum_path
    ):
        network_einsum = network_dispatcher(
            self.einsum_expr,
            self.data_operands,
            self.options,
            mode=options_constructor_mode,
        )

        network_interleaved = network_dispatcher(
            self.einsum_expr,
            self.data_operands,
            self.options,
            mode=options_constructor_mode,
            interleaved_inputs=self.interleaved_inputs
        )

        optimize = deepcopy(self.optimize)

        if use_numpy_einsum_path:
            optimize["path"] = self.numpy_einsum_path[0][1:]

        network_einsum.contract_path(
            optimize=optimizer_options_dispatcher(
                optimize,
                mode=optimize_constructor_mode
            )
        )

        network_interleaved.contract_path(
            optimize=optimizer_options_dispatcher(
                optimize,
                mode=optimize_constructor_mode
            )
        )

        for stream_name in stream_names:
            if stream_name is not None and stream_name != self.tensor_package: continue
            network_einsum.autotune(iterations=self.iterations, stream=streams[stream_name])  # if iterations=0, autotune is skipped
            stream_name_sync_dispatcher(stream_name, skip=skip_sync)
            cutensornet_contract = network_einsum.contract(stream=streams[stream_name])
            stream_name_sync_dispatcher(stream_name, skip=skip_sync)
            allclose(self.source, self.dtype_name, cutensornet_contract, self.einsum)

        network_einsum.free()

        for stream_name in stream_names:
            if stream_name is not None and stream_name != self.tensor_package: continue
            network_interleaved.autotune(iterations=self.iterations, stream=streams[stream_name])  # if iterations=0, autotune is skipped
            stream_name_sync_dispatcher(stream_name, skip=skip_sync)
            cutensornet_contract = network_interleaved.contract(stream=streams[stream_name])
            stream_name_sync_dispatcher(stream_name, skip=skip_sync)
            allclose(self.source, self.dtype_name, cutensornet_contract, self.einsum)

        network_interleaved.free()

    def test_contract(
        self,
        skip_sync,
        use_numpy_einsum_path
    ):
        self._test_contract(
            self.options_cmode,
            self.optimize_cmode,
            skip_sync,
            use_numpy_einsum_path
        )

    def run_tests(self):
        self.test_contract_path(self.options_cmode, self.optimize_cmode)

        self.test_contract(False, False)
        self.test_contract(False, True)
        self.test_contract(True, True)
        self.test_contract(True, False)

@pytest.fixture
def NetworkFixture(request):
    return NetworkProxyFixture(request.param)

class TestNetwork:

    @pytest.mark.parametrize(
        "NetworkFixture",
        itertools.product(
            sources_devices_dtype_names,
            array_orders,
            einsum_expressions,
            network_options,
            optimizer_options,
            opt_cmodes,
            opt_cmodes,
            iterations
        ),
        indirect=["NetworkFixture"]
    )
    def test_network(self, NetworkFixture):
        NetworkFixture.run_tests()
