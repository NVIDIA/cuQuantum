import pytest
import itertools

from cuquantum import contract_path, einsum_path

from .testutils import *


class ContractPathProxyFixture(ProxyFixtureBase):
    def __init__(self, network_options_pack):
        super().__init__(network_options_pack)
    def test_contract_path(self):
        cutensornet_contract_path_einsum = contract_path(
            self.einsum_expr,
            *self.data_operands,
            optimize=optimizer_options_dispatcher(
                self.optimize,
                mode=self.optimize_cmode
            )
        )
        cutensornet_contract_path_einsum = einsum_path(
            self.einsum_expr,
            *self.data_operands
        )
        cutensornet_contract_path_interleaved = contract_path(
            *self.interleaved_inputs,
            optimize=optimizer_options_dispatcher(
                self.optimize,
                mode=self.optimize_cmode
            )
        )
        cutensornet_contract_path_interleaved = einsum_path(
            *self.interleaved_inputs
        )
    def run_tests(self):
        self.test_contract_path()

@pytest.fixture
def ContractPathFixture(request):
    return ContractPathProxyFixture(request.param)

class TestContractPath:
    @pytest.mark.parametrize(
        "ContractPathFixture",
        itertools.product(
            sources_devices_dtype_names,
            array_orders,
            einsum_expressions,
            [None],  # only consider a single network options path; others tested elsewhere
            [None],  # only consider a single optimizer options; others tested elsewhere
            [None],  # ignore network options constructor modes, options constructor mode is not used
            [None],  # only consider a single optimizer options constructor mode; others tested elsewhere
            [None]  # ignore iterations, autotune is not used
        ),
        indirect=["ContractPathFixture"]
    )
    def test_contract_path(self, ContractPathFixture):
        ContractPathFixture.run_tests()
