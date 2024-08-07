# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import dataclasses

import opt_einsum as oe
import pytest

import cupy as cp
import numpy as np

from cuquantum import contract, tensor, CircuitToEinsum, OptimizerInfo
from cuquantum.cutensornet.experimental import contract_decompose, ContractDecomposeAlgorithm, ContractDecomposeInfo, NetworkOperator, NetworkState, MPSConfig, TNConfig
from cuquantum.cutensornet.experimental._internal.utils import is_gate_split
from cuquantum.cutensornet.experimental._internal.network_state_utils import STATE_SUPPORTED_DTYPE_NAMES
from cuquantum.cutensornet._internal.decomposition_utils import DECOMPOSITION_DTYPE_NAMES, parse_decomposition
from cuquantum.cutensornet._internal.utils import infer_object_package

from .approxTN_utils import split_contract_decompose, tensor_decompose, verify_split_QR, verify_split_SVD, SingularValueDegeneracyError
from .circuit_data import backend, backend_cycle
from .circuit_tester import get_random_pauli_strings
from .circuit_utils import get_contraction_tolerance, get_mps_tolerance
from .data import backend_names, contract_decompose_expr
from .state_data import testing_circuits_mps, qudits_to_test, state_settings, approx_mps_options, factory_backend, factory_backend_cycle, svd_algorithm, svd_algorithm_cycle, create_vqc_states, STATE_UPDATE_CONFIGS
from .state_tester import StateFactory, NetworkStateFunctionalityTester, ExactStateAPITester, ApproximateMPSTester, MPS, get_random_network_operator, is_converter_mps_compatible
from .test_circuit_converter import CIRCUIT_TEST_SETTING
from .test_options import _OptionsBase
from .test_utils import DecomposeFactory, deselect_contract_decompose_algorithm_tests, deselect_decompose_tests, get_svd_methods_for_test, DEFAULT_RNG, gen_rand_svd_method
from .test_utils import get_stream_for_backend, deselect_network_operator_from_pauli_string_tests, deselect_invalid_device_id_tests, get_state_internal_backend_device


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
            stream = get_stream_for_backend(backend)
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
            try:
                outputs_ref = tensor_decompose(decomp_expr, intm, method="svd", return_info=return_info, **svd_kwargs)
            except SingularValueDegeneracyError:
                pytest.skip("Test skipped due to singular value degeneracy issue")
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

    
    def test_contract_svd_decompose(self, decompose_expr, xp, dtype, order, stream):
        methods = get_svd_methods_for_test(3, dtype)
        for svd_method in methods:
            algorithm = ContractDecomposeAlgorithm(qr_method=False, svd_method=svd_method)
            self._run_contract_decompose(decompose_expr, xp, dtype, order, stream, algorithm)

    
    def test_contract_qr_assisted_svd_decompose(self, decompose_expr, xp, dtype, order, stream):
        methods = get_svd_methods_for_test(3, dtype)
        for svd_method in methods:
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
        'svd_info', [None, tensor.SVDInfo(reduced_extent=2, full_extent=4, discarded_weight=0.01, algorithm='gesvdj')]
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

# Correctness tests will be performed in TestNetworkState
class TestNetworkOperator:
    @pytest.mark.uncollect_if(func=deselect_invalid_device_id_tests)
    @pytest.mark.parametrize("backend", backend_names)
    @pytest.mark.parametrize("state_dim_extents",(3, 4, 7, (3, 2, 4, 5), (4, 5, 2, 3, 2)))
    @pytest.mark.parametrize("dtype", STATE_SUPPORTED_DTYPE_NAMES)
    @pytest.mark.parametrize('device_id', (None, 0, 2))
    def test_network_operator(self, backend, state_dim_extents, dtype, device_id):
        if isinstance(state_dim_extents, int):
            state_dim_extents = (2, ) * state_dim_extents
        network_operator = get_random_network_operator(state_dim_extents, 
            backend=backend, rng=np.random.default_rng(2), num_repeats=2, dtype=dtype, options={'device_id': device_id})
        expected_backend, expected_device = get_state_internal_backend_device(backend, device_id)
        for tensors, _, _ in network_operator.mpos + network_operator.tensor_products:
            for o in tensors:
                assert (o.name, o.device_id) == (expected_backend, expected_device)
    
    @pytest.mark.uncollect_if(func=deselect_network_operator_from_pauli_string_tests)
    @pytest.mark.parametrize("backend", backend_names)
    @pytest.mark.parametrize("n_qubits", (3, 4, 5, 8, 12))
    @pytest.mark.parametrize("num_pauli_strings", (None, 1, 4))
    @pytest.mark.parametrize("dtype", STATE_SUPPORTED_DTYPE_NAMES)
    @pytest.mark.parametrize('device_id', (None, 0, 2))
    def test_from_pauli_strings(self, backend, n_qubits, num_pauli_strings, dtype, device_id):
        expected_backend, expected_device = get_state_internal_backend_device(backend, device_id)
        if backend == 'torch-gpu':
            backend = 'torch'
        pauli_strings = get_random_pauli_strings(n_qubits, num_pauli_strings, rng=np.random.default_rng(4))
        network_operator = NetworkOperator.from_pauli_strings(pauli_strings, dtype=dtype, backend=backend, options={'device_id': device_id})
        expected_device = 0 if device_id is None else device_id
        for tensors, _, _ in network_operator.mpos + network_operator.tensor_products:
            for o in tensors:
                assert (o.name, o.device_id) == (expected_backend, expected_device)


class TestNetworkStateFunctionality:

    @pytest.mark.uncollect_if(func=deselect_invalid_device_id_tests)
    @pytest.mark.parametrize("circuit", testing_circuits_mps)
    @pytest.mark.parametrize("dtype", ('complex64', 'complex128'))
    @pytest.mark.parametrize("config", (TNConfig(), MPSConfig()))
    @pytest.mark.parametrize('device_id', (None, 1))
    def test_circuit_state(self, circuit, dtype, config, backend, device_id):
        print(f"{backend=}")
        with NetworkState.from_circuit(circuit, dtype=dtype, backend=backend, config=config, options={'device_id': device_id}) as state:
            tester = NetworkStateFunctionalityTester(state, backend, is_normalized=True)
            tester.run_tests()

    @pytest.mark.parametrize("qudits", qudits_to_test)
    @pytest.mark.parametrize("dtype", ('float32', 'float64', 'complex64', 'complex128'))
    @pytest.mark.parametrize("config", (TNConfig(), MPSConfig(mpo_application='exact')))
    def test_custom_state(self, qudits, dtype, config, factory_backend):
        print(f"{factory_backend=}")
        adjacent_double_layer = False
        if isinstance(qudits_to_test, (tuple, list)) and len(set(qudits_to_test)) != 1:
            # for qudits with different dimensions, exact simulation only supports adjacent double layers
            adjacent_double_layer = True

        factory = StateFactory(qudits, dtype, layers='SDCMDS', 
            backend=factory_backend, rng=np.random.default_rng(10), initial_mps_dim=2, adjacent_double_layer=adjacent_double_layer)
        
        with factory.to_network_state(config=config) as state:
            tester = NetworkStateFunctionalityTester(state, factory_backend, is_normalized=False)
            tester.run_tests()


class TestNetworkStateCorrectness:
    @pytest.mark.parametrize("circuit", testing_circuits_mps)
    @pytest.mark.parametrize("dtype", ('complex64', 'complex128',))
    def test_exact_circuit_state(self, circuit, dtype, backend, svd_algorithm):
        # Results from CircuitToEinsum are compared with following methods:
        #       1. Tensor network simulation based on NetworkState
        #       3. Exact MPS simulation based on NetworkState APIs if no mulit-qubit gates exist in the circuit
        #       4. Exact MPS simulation based on a reference cupy implementation in `state_tester.MPS` if no multi-qubit gates exist in the circuit
        print(f"{backend=}, {svd_algorithm=}")
        state_tester = ExactStateAPITester.from_circuit(circuit, dtype, backend=backend, 
            mps_options={'algorithm': svd_algorithm}, **CIRCUIT_TEST_SETTING)
        state_tester.run_tests()

    @pytest.mark.parametrize("qudits", qudits_to_test)
    @pytest.mark.parametrize("dtype", ('float32', 'float64', 'complex64', 'complex128'))
    @pytest.mark.parametrize("state_setting", state_settings)
    def test_exact_custom_state(self, qudits, dtype, state_setting, factory_backend):
        print(f"{factory_backend=}")
        adjacent_double_layer, mpo_bond_dim, mpo_num_sites, mpo_geometry, ct_target_place, initial_mps_dim = state_setting
        if isinstance(qudits_to_test, (tuple, list)):
            # for qudits with different dimensions, exact simulation only supports adjacent double layers
            if len(set(qudits_to_test)) != 1:
                adjacent_double_layer = True
        factory = StateFactory(qudits, 
                               dtype, 
                               backend=factory_backend,
                               layers='SDCMDS',
                               adjacent_double_layer=adjacent_double_layer,
                               mpo_bond_dim=mpo_bond_dim,
                               mpo_num_sites=mpo_num_sites,
                               mpo_geometry=mpo_geometry,
                               ct_target_place=ct_target_place,
                               initial_mps_dim=initial_mps_dim)
        expr = factory.get_sv_contraction_expression()
        sv0 = contract(*expr)
        
        mps = MPS.from_factory(factory, mpo_application='exact')
        sv = mps.compute_state_vector()
        assert factory.backend.allclose(sv0, sv, **mps.tolerance)
        sv = None

        rdm0 = mps.compute_reduced_density_matrix((0,))

        for config in (TNConfig(), MPSConfig(mpo_application='exact')):
            with factory.to_network_state(config=config) as state:
                sv = state.compute_state_vector()
                rdm = state.compute_reduced_density_matrix((0,))
                assert factory.backend.allclose(sv, sv0, **mps.tolerance)
                assert factory.backend.allclose(rdm, rdm0, **mps.tolerance)


    @pytest.mark.parametrize("circuit", testing_circuits_mps)
    @pytest.mark.parametrize("mps_config_iter", (0, 1))
    def test_approximate_circuit_state(self, circuit, mps_config_iter, backend):
        # Results from cutensornet MPS state API are compared with the reference cupy implementation in `state_tester.MPS` if no multi-qubit gates exist in the circuit
        # restrict to single precision to avoid accuracy fallout
        print(f"{backend=}")
        dtype = 'complex128'
        converter = CircuitToEinsum(circuit, dtype=dtype, backend=backend)
        n_qubits = converter.n_qubits
        if not is_converter_mps_compatible(converter):
            pytest.skip("MPS test skipped due to multi-qubit gate")
        # restrict to gesvd algorithm to avoid accuracy fallout
        mps_options = gen_rand_svd_method(DEFAULT_RNG, 
                                          dtype, 
                                          fixed={'algorithm': 'gesvd'}, 
                                          exclude={'partition'})
        
        mps_options['canonical_center'] = DEFAULT_RNG.choice([None] + list(range(n_qubits)))
        try:
            approx_mps_tests = ApproximateMPSTester.from_converter(converter, mps_options, **CIRCUIT_TEST_SETTING)
            approx_mps_tests.run_tests()
        except SingularValueDegeneracyError:
            pytest.skip("Test skipped due to singular value degeneracy issue")

    @pytest.mark.parametrize("qudits", qudits_to_test)
    @pytest.mark.parametrize("state_setting", state_settings)
    @pytest.mark.parametrize("dtype", ('float64', 'complex128'))
    @pytest.mark.parametrize("mps_option", approx_mps_options)
    def test_approximate_custom_state(self, qudits, state_setting, dtype, mps_option, factory_backend):
        print(f"{factory_backend=}")
        adjacent_double_layer, mpo_bond_dim, mpo_num_sites, mpo_geometry, ct_target_place, initial_mps_dim = state_setting
        factory = StateFactory(qudits, 
                               dtype, 
                               layers='SDCMDS',
                               backend=factory_backend,
                               adjacent_double_layer=adjacent_double_layer,
                               mpo_bond_dim=mpo_bond_dim,
                               mpo_num_sites=mpo_num_sites,
                               mpo_geometry=mpo_geometry,
                               ct_target_place=ct_target_place,
                               initial_mps_dim=initial_mps_dim
                               )
        try:
            tester = ApproximateMPSTester.from_factory(factory, mps_option, rng=np.random.default_rng(2024))
            tester.run_tests()
        except SingularValueDegeneracyError:
            pytest.skip("Test skipped due to singular value degeneracy issue")
    

    @pytest.mark.parametrize("config", STATE_UPDATE_CONFIGS)
    def test_update_reuse_correctness(self, config):
        (state_a, op_two_body_x), (state_b, op_two_body_y), operator, two_body_op_ids = create_vqc_states(config)
        if isinstance(state_a.config, TNConfig):
            tolerance = get_contraction_tolerance("complex128")
        else:
            tolerance = get_mps_tolerance("complex128")
            if not state_a.config._is_fixed_extent_truncation():
                mps0 = state_a.compute_output_state()
                mps1 = state_b.compute_output_state()
                # for MPS with value based truncation, make sure that the test case is designed such that the output MPS have different shapes
                assert any(o0.shape != o1.shape for o0, o1 in zip(mps0, mps1))

        original_expec = []
        for state in [state_a, state_b]:
            e = state.compute_expectation(operator) / state.compute_norm()
            original_expec.append(e)
        
        for tensor_id in two_body_op_ids:
            state_a.update_tensor_operator(tensor_id, op_two_body_y, unitary=False)
            state_b.update_tensor_operator(tensor_id, op_two_body_x, unitary=False)
        
        updated_expec = []
        for state in [state_b, state_a]:
            e = state.compute_expectation(operator) / state.compute_norm()
            updated_expec.append(e)

        for e1, e2 in zip(original_expec, updated_expec):
            np.allclose(e1, e2, **tolerance)
        
        # we here first perform expectation check and then state vector check as caching in 24.08 is only activated for one compute object at one time.
        original_sv = []
        for state in [state_b, state_a]:
            original_sv.append(state.compute_state_vector())
        
        for tensor_id in two_body_op_ids:
            state_a.update_tensor_operator(tensor_id, op_two_body_x, unitary=False)
            state_b.update_tensor_operator(tensor_id, op_two_body_y, unitary=False)
        
        updated_sv = []
        for state in [state_a, state_b]:
            updated_sv.append(state.compute_state_vector())

        for sv1, sv2 in zip(original_sv, updated_sv):
            cp.allclose(sv1, sv2, **tolerance)