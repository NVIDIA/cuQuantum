import pytest
import itertools
from copy import deepcopy

from cuquantum import contract, einsum

from .testutils import *


class ContractProxyFixture(ProxyFixtureBase):
    def __init__(self, network_options_pack):
        super().__init__(network_options_pack)
    def _test_contract(
        self,
        options_constructor_mode,
        optimize_constructor_mode,
        skip_sync,
        use_numpy_einsum_path
    ):
        for stream_name in stream_names:
            optimize = deepcopy(self.optimize)

            if use_numpy_einsum_path:
                optimize["path"] = self.numpy_einsum_path[0][1:]

            self.cutensornet_einsum = einsum(
                self.einsum_expr,
                *self.data_operands
            )

            self.cutensornet_interleaved_einsum = einsum(
                *self.interleaved_inputs
            )

            self.cutensornet_contract = contract(
                self.einsum_expr,
                *self.data_operands,
                options=network_options_dispatcher(self.options, mode=options_constructor_mode),
                optimize=optimizer_options_dispatcher(optimize, mode=optimize_constructor_mode),
                stream=streams[stream_name]
            )
            
            self.cutensornet_interleaved_contract = contract(
                *self.interleaved_inputs,
                options=network_options_dispatcher(self.options, mode=options_constructor_mode),
                optimize=optimizer_options_dispatcher(optimize, mode=optimize_constructor_mode),
                stream=streams[stream_name]
            )

            stream_name_sync_dispatcher(stream_name, skip=skip_sync)
            allclose(self.source, self.dtype_name, self.cutensornet_interleaved_einsum, self.cutensornet_einsum)
            allclose(self.source, self.dtype_name, self.cutensornet_interleaved_contract, self.cutensornet_contract)
            allclose(self.source, self.dtype_name, self.cutensornet_einsum, self.einsum)
            allclose(self.source, self.dtype_name, self.cutensornet_contract, self.einsum)
            
            tensor_class_equal(self.tensor_class, self.data_operands, self.cutensornet_einsum)
            tensor_class_equal(self.tensor_class, self.data_operands, self.cutensornet_interleaved_einsum)
            tensor_class_equal(self.tensor_class, self.data_operands, self.cutensornet_contract)
            tensor_class_equal(self.tensor_class, self.data_operands, self.cutensornet_interleaved_contract)

            dtypes_equal(self.dtype, self.data_operands, self.cutensornet_einsum)
            dtypes_equal(self.dtype, self.data_operands, self.cutensornet_interleaved_einsum)
            dtypes_equal(self.dtype, self.data_operands, self.cutensornet_contract)
            dtypes_equal(self.dtype, self.data_operands, self.cutensornet_interleaved_contract)
            
    def test_contract(self, skip_sync, use_numpy_einsum_path):
        self._test_contract(
            self.options_cmode,
            self.optimize_cmode,
            skip_sync,
            use_numpy_einsum_path
        )

    def run_tests(self):
        self.test_contract(False, False)
        self.test_contract(False, True)
        self.test_contract(True, True)
        self.test_contract(True, False)

@pytest.fixture
def ContractFixture(request):
    return ContractProxyFixture(request.param)

class TestContract:
    @pytest.mark.parametrize(
        "ContractFixture",
        itertools.product(
            sources_devices_dtype_names,
            array_orders,
            einsum_expressions,
            network_options,
            optimizer_options,
            opt_cmodes,  # cmodes for network options
            opt_cmodes,  # cmodes for optimizer options
            [None]  # ignore iterations, autotune is not used
        ),
        indirect=["ContractFixture"]
    )
    def test_contract(self, ContractFixture):
        ContractFixture.run_tests()
