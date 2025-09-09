# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test the reference MPS implementation under cuquantum_tests/tensornet/experimental/_internal/mps_utils.MPS in the exact limit
"""

import pytest

from cuquantum.tensornet import contract

from ..utils.helpers import _BaseTester
from ..utils.circuit_ifc import CircuitHelper, QuantumStateTestHelper
from ._internal.state_matrix import CircuitStateMatrix
from ._internal.state_factory import StateFactory
from ._internal.mps_utils import MPS, get_mps_tolerance


class TestCircuitState(_BaseTester):
    @pytest.mark.parametrize("circuit", CircuitStateMatrix.L2())
    def test_circuit(self, circuit):
        backend = self._get_array_framework(circuit, "CircuitState")
        my_mps = MPS.from_circuit(circuit, backend=backend)
        sv = my_mps.compute_state_vector()
        sv_ref = CircuitHelper.compute_state_vector(circuit)
        tol = get_mps_tolerance(my_mps.dtype)
        QuantumStateTestHelper.verify_state_vector(sv, sv_ref, **tol)


@pytest.mark.parametrize(
    "gauge_option", ("free", "simple")
)
@pytest.mark.parametrize(
    "qudits", (4, (2, 3, 5, 4))
)
class TestGenericState(_BaseTester):

    def _test_factory(self, factory, **mps_options):
        expr = factory.get_sv_contraction_expression()
        sv = contract(*expr)
        
        my_mps = MPS.from_factory(factory, **mps_options)
        sv_mps = my_mps.compute_state_vector()
        # Generic state can be less accurate, so we don't enforce the high tolerance
        QuantumStateTestHelper.verify_state_vector(sv, sv_mps)

    @pytest.mark.parametrize(
        "adjacent_double_layer", (True, False)
    )
    def test_double_layer(self, qudits, gauge_option, adjacent_double_layer):
        rng = self._get_rng(qudits, gauge_option, adjacent_double_layer)
        backend = self._get_array_framework(qudits, gauge_option, adjacent_double_layer)
        factory = StateFactory(qudits, 
                               "complex128", 
                               'SDSDS', 
                               rng,
                               backend=backend,
                               adjacent_double_layer=adjacent_double_layer)
        self._test_factory(factory, mpo_application='exact', gauge_option=gauge_option)
    
    @pytest.mark.parametrize(
        "initial_mps_dim", (None, 2, 3)
    )
    def test_initial_mps(self, qudits, gauge_option, initial_mps_dim):
        rng = self._get_rng(qudits, gauge_option, initial_mps_dim)
        backend = self._get_array_framework(qudits, gauge_option, initial_mps_dim)
        factory = StateFactory(qudits, 
                               "float64", 
                               'SDSDS', 
                               rng,
                               backend=backend,
                               initial_mps_dim=initial_mps_dim,
                               adjacent_double_layer=False)
        self._test_factory(factory, mpo_application='exact', gauge_option=gauge_option)
    
    @pytest.mark.parametrize(
        "ct_target_place", ("first", "middle", "last")
    )
    def test_controlled_tensor(self, qudits, gauge_option, ct_target_place):
        rng = self._get_rng(qudits, gauge_option, ct_target_place)
        backend = self._get_array_framework(qudits, gauge_option, ct_target_place)
        factory = StateFactory(qudits, 
                               "float64", 
                               'SDCS', 
                               rng,
                               backend=backend,
                               ct_target_place=ct_target_place,
                               adjacent_double_layer=False)
        self._test_factory(factory, mpo_application='exact', gauge_option=gauge_option)
    
    @pytest.mark.parametrize(
        "mpo_bond_dim", (2, 3),
    )
    @pytest.mark.parametrize(
        "mpo_num_sites", (None, 3)
    )
    @pytest.mark.parametrize(
        "mpo_geometry", ("adjacent-ordered", "random", "random-ordered"),
    )
    def test_mpo(self, qudits, gauge_option, mpo_bond_dim, mpo_num_sites, mpo_geometry):
        rng = self._get_rng(qudits, gauge_option, mpo_bond_dim, mpo_num_sites, mpo_geometry)
        backend = self._get_array_framework(qudits, gauge_option, mpo_bond_dim, mpo_num_sites, mpo_geometry)
        factory = StateFactory(qudits, 
                               "float64", 
                               'SDMS', 
                               rng,
                               backend=backend,
                               mpo_bond_dim=mpo_bond_dim,
                               mpo_num_sites=mpo_num_sites,
                               mpo_geometry=mpo_geometry,
                               adjacent_double_layer=False)
        self._test_factory(factory, mpo_application='exact', gauge_option=gauge_option)
        