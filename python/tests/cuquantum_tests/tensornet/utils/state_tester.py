# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from collections import Counter

import numpy as np

import opt_einsum as oe

from cuquantum import CircuitToEinsum
from cuquantum.cutensornet.experimental import NetworkState, TNConfig, MPSConfig, NetworkOperator
from cuquantum.tensornet.experimental._internal.network_state_utils import get_pauli_map
from cuquantum.tensornet._internal.decomposition_utils import compute_mid_extent
from cuquantum._internal.utils import infer_object_package
from cuquantum._internal import tensor_wrapper

from .approxTN_utils import gate_decompose, tensor_decompose, SVD_TOLERANCE, verify_unitary
from .circuit_tester import BaseTester, get_random_pauli_strings, get_engine_iters, compute_sample_overlap
from .circuit_utils import (
    _BaseComputeEngine, 
    ConverterComputeEngine, 
    get_mps_tolerance, 
    reduced_density_matrix_from_sv, 
    amplitude_from_sv,
    batched_amplitude_from_sv,
    expectation_from_sv,
)
from .test_utils import DEFAULT_RNG, EMPTY_DICT, TensorBackend, get_or_create_tensor_backend, atol_mapper, rtol_mapper, get_dtype_name


# valid simulation setting for reference MPS class
MPS_VALID_CONFIGS = {'max_extent', 'abs_cutoff', 'rel_cutoff', 'discarded_weight_cutoff', 'normalization', 'canonical_center', 'mpo_application', 'gauge_option'}
STATE_PROPERTIES_NAMES = ('amplitude', 'batched_amplitudes', 'state_vector', 'reduced_density_matrix', 'expectation', 'sampling')

def is_converter_mps_compatible(converter):
    for _, qubits in converter.gates:
        if len(qubits) > 2:
            return False
    return True

def verify_mps_canonicalization(mps_tensors, canonical_center):
    dtype = get_dtype_name(mps_tensors[0].dtype)
    if canonical_center is None:
        return True
    is_canonical = True
    for i, t in enumerate(mps_tensors):
        if t.ndim == 3:
            modes = 'ipj'
        elif t.ndim == 2:
            if i == 0:
                modes = 'pj'
            elif i == (len(mps_tensors) - 1):
                modes = 'ip'
            else:
                raise ValueError
        else:
            raise ValueError
        if i < canonical_center:
            shared_mode = 'j'
        elif i > canonical_center:
            shared_mode = 'i'
        else:
            continue
        is_canonical = is_canonical and verify_unitary(t, modes, shared_mode, 
                    SVD_TOLERANCE[dtype], tensor_name=f"Site {i} canonicalization")
    return is_canonical


def get_device_id(options):
    if isinstance(options, dict):
        device_id = options.get('device_id', 0)
    else:
        device_id = getattr(options, 'device_id', 0)
    return device_id

def get_random_network_operator(state_dims, *, backend='cupy', rng=DEFAULT_RNG, num_repeats=2, dtype='complex128', options=None):
    device_id = get_device_id(options)
    backend = TensorBackend(backend=backend, device_id=device_id)    
    operator_obj = NetworkOperator(state_dims, dtype=dtype, options=options)
    def get_random_modes():
        num_rand_modes = rng.integers(2, len(state_dims))
        rand_modes = list(range(len(state_dims)))
        rng.shuffle(rand_modes)
        return rand_modes[:num_rand_modes]

    # adding tensor product
    for _ in range(num_repeats):
        coefficient = rng.random(1).item()
        if dtype.startswith('complex'):
            coefficient += 1j * rng.random(1).item()
        prod_modes = get_random_modes()
        prod_tensors = []
        for i in prod_modes:
            shape = (state_dims[i], ) * 2
            t = backend.random(shape, dtype, rng=rng)
            prod_tensors.append(t)
        prod_modes = [[i] for i in prod_modes]
        operator_obj.append_product(coefficient, prod_modes, prod_tensors)
    # adding MPOs
    for _ in range(num_repeats):
        coefficient = rng.random(1).item()
        if dtype.startswith('complex'):
            coefficient += 1j * rng.random(1).item()
        mpo_modes = get_random_modes()
        num_mpo_modes = len(mpo_modes)
        mpo_tensors = []
        bond_prev = None
        for i, m in enumerate(mpo_modes):
            bond_next = rng.integers(2, 5)
            dim = state_dims[m]
            if i == 0:
                shape = (dim, bond_next, dim)
            elif i == num_mpo_modes - 1:
                shape = (bond_prev, dim, dim)
            else:
                shape = (bond_prev, dim, bond_next, dim)
            t = backend.random(shape, dtype, rng=rng)
            mpo_tensors.append(t)
            bond_prev = bond_next
        operator_obj.append_mpo(coefficient, mpo_modes, mpo_tensors)
    return operator_obj

def compute_state_basic_property(state, property_name):
    func = getattr(state, f'compute_{property_name}')
    if property_name == 'amplitude':
        return func('0' * state.n, return_norm=True)
    elif property_name == 'batched_amplitudes':
        return func({0: 0, 1:1})
    elif property_name == 'state_vector':
        return func()
    elif property_name == 'reduced_density_matrix':
        return func((0, ))
    elif property_name == 'sampling':
        return func(1000, seed=1)
    elif property_name == 'expectation':
        if set(state.state_mode_extents) == set([2]):
            pauli_strings = {'X' * state.n: 0.1,
                             'Y' * state.n: 0.2,
                             'Z' * state.n: 0.4}
            expec, norm = func(pauli_strings, return_norm=True)
            return expec
    else:
        raise ValueError(f"{property_name} not supported")

def compute_state_basic_quantities(state):
    output = {}
    for property_name in STATE_PROPERTIES_NAMES:
        if property_name == 'expectation' and set(state.state_mode_extents) != set([2]):
            continue
        if property_name == 'amplitude':
            output[property_name], output['norm'] = compute_state_basic_property(state, property_name)
        else:
            output[property_name] = compute_state_basic_property(state, property_name)
    return output

def apply_factory_sequence(network_state, sequence):
    tensor_ids = []
    for op, modes, gate_info in sequence:
        if gate_info is None:
            if isinstance(op, (list, tuple)):
                # MPO
                tensor_id = network_state.apply_mpo(modes, op)
            else:
                # GATE
                tensor_id = network_state.apply_tensor_operator(modes, op)
        else:
            if 'probabilities' in gate_info:
                is_unitary_channel = gate_info['probabilities'] is not None
                if is_unitary_channel:
                    tensor_id = network_state.apply_unitary_tensor_channel(modes, op, gate_info['probabilities'])
                else:
                    tensor_id = network_state.apply_general_tensor_channel(modes, op)
            else:
                assert 'control_modes' in gate_info
                assert 'control_values' in gate_info
                # Controlled-Tensor
                # NetworkState currently only support immutable controlled tensors
                tensor_id = network_state.apply_tensor_operator(modes, op, control_modes=gate_info['control_modes'], control_values=gate_info['control_values'], immutable=True)
        tensor_ids.append(tensor_id)
    return tensor_ids

class FatalAssertionError(AssertionError):
    pass

class StatisticalAssertionError(AssertionError):
    pass

class StateFactory:
    def __init__(
        self, 
        qudits, 
        dtype, 
        layers='SDCMDS', 
        rng=None,
        backend='cupy',
        adjacent_double_layer=True,
        mpo_bond_dim=2,
        mpo_num_sites=None,
        mpo_geometry="adjacent-ordered",
        ct_target_place="last", # Controlled-Tensor: ct
        initial_mps_dim=None,
        options=None,
    ):
        if isinstance(qudits, (int, np.integer)):
            self.num_qudits = qudits
            self.state_dims = (2, ) * self.num_qudits
        else:
            self.num_qudits = len(qudits)
            self.state_dims = qudits

        self.device_id = get_device_id(options)
        self.backend = TensorBackend(backend=backend, device_id=self.device_id)
        self.dtype = get_dtype_name(dtype)

        dims = set(self.state_dims)
        if len(dims) == 1 and dims.pop() == 2:
            # unitary/general channel only supported for qubits
            assert set(layers).issubset(set('SDCMuUgG'))
        else:
            assert set(layers).issubset(set('SDCM'))
        self.layers = layers

        if rng is None:
            rng = np.random.default_rng(2024)
        self.rng = rng
        self.adjacent_double_layer = adjacent_double_layer

        # settings for MPO layer
        self.mpo_bond_dim = mpo_bond_dim
        if mpo_num_sites is None:
            self.mpo_num_sites = self.num_qudits
        else:
            mpo_num_sites = min(self.num_qudits, mpo_num_sites)
            assert mpo_num_sites >= 2
            self.mpo_num_sites = mpo_num_sites
        assert mpo_geometry in {"adjacent-ordered", "random-ordered", "random"}
        self.mpo_geometry = mpo_geometry
        assert ct_target_place in {"first", "middle", "last"}
        self.ct_target_place = ct_target_place
        self.initial_mps_dim = initial_mps_dim
        self.psi = None
        self._sequence = []

    def get_initial_state(self):
        if self.psi is None:
            self.psi = []
            if self.initial_mps_dim is None:
                for d in self.state_dims:
                    t = self.backend.zeros(d, dtype=self.dtype)
                    t[0] = 1
                    self.psi.append(t)
            else:
                virtual_dim = self.initial_mps_dim
                for i, d in enumerate(self.state_dims):
                    if i==0:
                        shape = (d, virtual_dim)
                    elif i == self.num_qudits - 1:
                        shape = (virtual_dim, d)
                    else:
                        shape = (virtual_dim, d, virtual_dim)
                    self.psi.append(self.backend.random(shape, self.dtype, rng=self.rng))
        return self.psi

    @property
    def sequence(self):
        if not self._sequence:
            self._generate_raw_sequence()
        return self._sequence

    def _generate_raw_sequence(self):
        double_layer_offset = 0
        for layer in self.layers:
            if layer == 'S':
                self._append_single_qudit_layer()
            elif layer == 'D':
                self._append_double_qudit_layer(offset=double_layer_offset)
                # intertwine double layers
                double_layer_offset = 1 - double_layer_offset
            elif layer == 'C':
                self._append_controlled_tensor_mpo_layer()
            elif layer == 'M':
                self._append_mpo_layer()
            elif layer.upper() == 'U':
                self._append_unitary_channel_layer(dense=layer=='U')
            elif layer.upper() == 'G':
                self._append_general_channel_layer(dense=layer=='G')
            else:
                raise ValueError(f"layer type {layer} not supported")
    
    def append_sequence(self, sequence):
        self._sequence.append(sequence)
    
    def _get_pauli_map(self):
        is_complex = 'complex' in self.dtype
        if not hasattr(self, 'pauli_map'):
            complex_dtype = self.dtype if is_complex else {'float32': 'complex64', 'float64': 'complex128'}[self.dtype]
            self.pauli_map = get_pauli_map(complex_dtype, backend=self.backend.name, device_id=self.device_id)
            if not is_complex:
                self.pauli_map = {p: self.pauli_map[p].real for p in 'IXZ'}
            pauli_keys = list(self.pauli_map.keys())
            for p1 in pauli_keys:
                for p2 in pauli_keys:
                    self.pauli_map[p1+p2] = self.backend.einsum('Aa,Bb->ABab', self.pauli_map[p1], self.pauli_map[p2])
        return self.pauli_map
    
    def _append_unitary_channel_layer(self, *, dense=True):
        is_complex = 'complex' in self.dtype
        pauli_map = self._get_pauli_map()
        qudits = list(range(self.num_qudits))
        self.rng.shuffle(qudits)
        if dense:
            self._sequence.append(([pauli_map['I'], pauli_map['X']], (qudits[0], ), {'probabilities': [0.95, 0.05]}))
            if is_complex:
                self._sequence.append(([pauli_map[p] for p in 'IXYZ'], (qudits[1], ), {'probabilities': [0.7, 0.15, 0.1, 0.05]}))
            else:
                self._sequence.append(([pauli_map[p] for p in 'IXZ'], (qudits[1], ), {'probabilities': [0.85, 0.15, 0.1]}))
        
        if (dense and self.num_qudits >= 4) or (not dense):
            target_qudits = qudits[2:4] if dense else qudits[:2]
            if is_complex:
                operands = [self.pauli_map[p] for p in ('XY', 'YZ', 'ZX')]
            else:
                operands = [self.pauli_map[p] for p in ('IX', 'IZ', 'XZ')]
            gate_info = {'probabilities': [0.7, 0.2, 0.1]}
            self._sequence.append((operands, target_qudits, gate_info))
        return  

    def _append_general_channel_layer(self, *, dense=True):
        is_complex = 'complex' in self.dtype
        pauli_map = self._get_pauli_map()
        qudits = list(range(self.num_qudits))
        self.rng.shuffle(qudits)
        if dense:
            g0, g1 = self.rng.random(2)
            dampling_channel = [
                self.backend.asarray([[1, 0], [0, np.sqrt(1 - g0)]], dtype=self.dtype),
                self.backend.asarray([[0, np.sqrt(g0)], [0, 0]], dtype=self.dtype)
            ]
            if is_complex:
                depolarizing_channel = [
                    (1 - g1) ** .5 * pauli_map['I'],
                    (g1 / 3.) ** .5 * pauli_map['X'],
                    (g1 / 3.) ** .5 * pauli_map['Y'],
                    (g1 / 3.) ** .5 * pauli_map['Z'],
                ]
            else:
                depolarizing_channel = [
                    (1 - g1) ** .5 * pauli_map['I'],
                    (g1 / 2.) ** .5 * pauli_map['X'],
                    (g1 / 2.) ** .5 * pauli_map['Z'],
                ]
            
            gate_info = {'probabilities': None}
            self._sequence.append((dampling_channel, (qudits[0], ), gate_info)) 
            self._sequence.append((depolarizing_channel, (qudits[1], ), gate_info)) 
        
        if (dense and self.num_qudits >= 4) or (not dense):
            target_qudits = qudits[2:4] if dense else qudits[:2]
            g = self.rng.random()
            if is_complex:
                operands = [
                    (1 - g) **.5 * pauli_map['XY'], 
                    (0.6 * g) ** .5 * pauli_map['YZ'], 
                    (0.4 * g) ** .5 * pauli_map['ZX'],
                ]
            else:
                operands = [
                    (1 - g) **.5 * pauli_map['IX'], 
                    (0.6 * g) ** .5 * pauli_map['IZ'], 
                    (0.4 * g) ** .5 * pauli_map['XZ'],
                ]
            gate_info = {'probabilities': None}
            self._sequence.append((operands, target_qudits, gate_info))
        return   
    
    def _append_single_qudit_layer(self):
        for i in range(self.num_qudits):
            shape = (self.state_dims[i], ) * 2
            t = self.backend.random(shape, self.dtype, rng=self.rng)
            t = t + t.conj().T
            t /= self.backend.norm(t)
            self._sequence.append((t, (i,), None))
    
    def _append_double_qudit_layer(self, offset=0):
        for i in range(offset, self.num_qudits-1, 2):
            j = i + 1 if self.adjacent_double_layer else self.rng.integers(i+1, self.num_qudits)
            shape = (self.state_dims[i], self.state_dims[j])* 2
            t = self.backend.random(shape, self.dtype, rng=self.rng)
            try:
                t = t + t.conj().transpose(2,3,0,1)
            except TypeError:
                t = t + t.conj().permute(2,3,0,1)
            t /= self.backend.norm(t)
            self._sequence.append((t, (i, j), None))
    
    def _append_mpo_layer(self):
        if self.mpo_geometry == "adjacent-ordered":
            start_site = self.rng.integers(0, self.num_qudits-self.mpo_num_sites+1)
            modes = list(range(start_site, start_site+self.mpo_num_sites))
        elif self.mpo_geometry in {"random-ordered", "random"}:
            modes = list(range(self.num_qudits))
            self.rng.shuffle(modes)
            modes = modes[:self.mpo_num_sites]
            if self.mpo_geometry == "random-ordered":
                modes = sorted(modes)
        else:
            raise ValueError(f"mpo geometry {self.mpo_geometry} not supported")
        mpo_tensors = []
        for i, q in enumerate(modes):
            phys_dim = self.state_dims[q]
            if i == 0:
                shape = (phys_dim, self.mpo_bond_dim, phys_dim)
                transpose_order = (2,1,0)
            elif i == self.mpo_num_sites - 1:
                shape = (self.mpo_bond_dim, phys_dim, phys_dim)
                transpose_order = (0,2,1)
            else:
                shape = (self.mpo_bond_dim, phys_dim, ) * 2
                transpose_order = (0,3,2,1)
            t = self.backend.random(shape, self.dtype, rng=self.rng)
            try:
                t = t + t.conj().transpose(*transpose_order)
            except TypeError:
                t = t + t.conj().permute(*transpose_order)
            t /= self.backend.norm(t)
            mpo_tensors.append(t)
        self._sequence.append((mpo_tensors, modes, None))

    def _append_controlled_tensor_mpo_layer(self):
        if self.mpo_geometry == "adjacent-ordered":
            start_site = self.rng.integers(0, self.num_qudits-self.mpo_num_sites+1)
            modes = list(range(start_site, start_site+self.mpo_num_sites))
        elif self.mpo_geometry in {"random-ordered", "random"}:
            modes = list(range(self.num_qudits))
            self.rng.shuffle(modes)
            modes = modes[:self.mpo_num_sites]
            modes = sorted(modes)
        else:
            raise ValueError(f"controlled-tensor-mpo geometry {self.mpo_geometry} not supported")
        
        target_modes = []
        control_modes = []
        if self.ct_target_place == "first":
            target_modes = [modes[0]]
            control_modes = modes[1:]
        elif self.ct_target_place == "middle":
            if(len(modes) < 3):
                raise ValueError(f"To apply target in the middle, #qubits should be > 2")
            idx = self.rng.integers(1, len(modes)-1)
            target_modes = [modes[idx]]
            modes.pop(idx)
            control_modes = modes
        elif self.ct_target_place == "last":
            target_modes = [modes[-1]]
            control_modes = modes[:-1]
        else:
            raise ValueError(f"controlled-tensor target place {self.ct_target_place} not supported")
        
        # control values
        control_values = []
        control_modes = sorted(control_modes)
        for cm in control_modes:
            v = self.rng.integers(0, self.state_dims[cm])
            control_values.append(v)
        # target data
        target_phys_dim = self.state_dims[target_modes[0]] # Currently we support only single-qubit target
        shape = (target_phys_dim, target_phys_dim)
        transpose_order = (1, 0)
        t = self.backend.random(shape, self.dtype, rng=self.rng)
        try:
            t = t + t.conj().transpose(*transpose_order)
        except TypeError:
            t = t + t.conj().permute(*transpose_order)
        t /= self.backend.norm(t)
        gate_info = {'control_modes': control_modes, 'control_values': control_values}
        self._sequence.append((t, target_modes, gate_info))

    def compute_control_tensor(self, control_dim, control_val, rank, direction):
        c1_rank3 = self.backend.asarray([1, 0, 0, 0,  0, 0, 0, 1]).reshape(2, 2, 2)
        c0_rank3_dwon = self.backend.asarray([0, 0, 1, 0,  0, 1, 0, 0]).reshape(2, 2, 2)
        c0_rank3_up = self.backend.asarray([0, 0, 0, 1,  1, 0, 0, 0]).reshape(2, 2, 2)
        c1_rank4_down = self.backend.asarray([1, 0, 0, 0,  0, 1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 1]).reshape(2, 2, 2, 2)
        c1_rank4_up = self.backend.asarray([1, 0, 1, 0,  0, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 1]).reshape(2, 2, 2, 2)
        c0_rank4_down = self.backend.asarray([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0]).reshape(2, 2, 2, 2)
        c0_rank4_up = self.backend.asarray([1, 0, 0, 0,  0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 0, 0]).reshape(2, 2, 2, 2)
        
        if control_dim == 2:
            if rank == "rank-3":
                if control_val == 1:
                    return c1_rank3
                elif control_val == 0 and direction == "down":
                     return c0_rank3_dwon
                elif control_val == 0 and direction == "up":
                        return c0_rank3_up
                else:
                    raise ValueError(f" Failed! Computing control tensor with rank {rank} and direction {direction} is not supported.")
            
            
            elif rank == "rank-4":
                if control_val == 1 and direction == "down":
                    return c1_rank4_down
                elif control_val == 1 and direction == "up":
                    return c1_rank4_up
                elif control_val == 0 and direction == "down":
                    return c0_rank4_down
                elif control_val == 0 and direction == "up":
                    return c0_rank4_up
                else:
                    raise ValueError(f" Failed! Computing control tensor with rank {rank} and direction {direction} is not supported.")
            else:
                raise ValueError(f" Failed! Control tensor can be rank-3 or rank-4, {rank} is not supported.")

        else: # It is a qudit with arbitrary extent/dimension
            if rank == "rank-3":
                if direction == "down":
                    shape = (control_dim, 2, control_dim)
                    ctrl_tensor = self.backend.zeros(shape)
                    for i in range(control_dim):
                        if i == control_val:
                            b = 1
                        else:
                            b = 0
                        ctrl_tensor[i][b][i] = 1
                    return ctrl_tensor
                elif direction == "up":
                    shape = (2, control_dim, control_dim)
                    ctrl_tensor = self.backend.zeros(shape)
                    for i in range(control_dim):
                        if i == control_val:
                            b = 1
                        else:
                            b = 0
                        ctrl_tensor[b][i][i] = 1
                    return ctrl_tensor
                else:
                    raise ValueError(f" Failed! Computing control tensor with direction {direction} is not supported.")

            elif rank == "rank-4":
                if direction == "down":
                    shape = (2, control_dim, 2, control_dim)
                    ctrl_tensor = self.backend.zeros(shape)
                    for b1 in range(2):
                        for i in range(control_dim):
                            if b1 == 1 and i == control_val:
                                b2 = 1
                            else:
                                b2 = 0
                            ctrl_tensor[b1][i][b2][i] = 1
                    return ctrl_tensor
                elif direction == "up":
                    shape = (2, control_dim, 2, control_dim)
                    ctrl_tensor = self.backend.zeros(shape)
                    for b2 in range(2):
                        for i in range(control_dim):
                            if b2 == 1 and i == control_val:
                                b1 = 1
                            else:
                                b1 = 0
                            ctrl_tensor[b1][i][b2][i] = 1
                    return ctrl_tensor
                else:
                    raise ValueError(f" Failed! Computing control tensor with direction {direction} is not supported.")

            else:
                raise ValueError(f" Failed! Control tensor can be rank-3 or rank-4, {rank} is not supported.")

    def compute_ct_mpo_tensors(self, control_modes, control_vals, target_modes, target_data):
        # Note: Input target_data has mode ABC...abc where ABCD are bra modes and abcd are ket modes
        # output MPO tensors (provide to apply_mpo) should have mode panA where p and n refers to modes connecting to previous and next site
        tensors = []
        modes = control_modes + target_modes
        modes = sorted(modes)
        n_qudits = len(modes)
        target_tensor_applied = 0
        for i, q in enumerate(modes):
            if i == 0: # rank-3
                if q == target_modes[0]: # target tensor
                    target = target_data.reshape(1, -1)[0]
                    t_dim = self.state_dims[q]
                    c0 = [0] * (t_dim ** 2)
                    for i in range(t_dim):
                        idx = i + (i * t_dim)
                        c0[idx] = 1
                    c0 = self.backend.asarray(c0)
                    t_temp = self.backend.vstack([c0, target])
                    # modes (n, A, a), i.e, virtual, bra, ket
                    t_temp = t_temp.reshape(2, t_dim, t_dim)
                    # convert to modes (a, n, A), i.e, ket, virtual, bra
                    try:
                        t_temp = t_temp.transpose(2, 0, 1)
                    except TypeError:
                        t_temp = t_temp.permute(2, 0, 1)
                    t = self.backend.asarray(t_temp, dtype=self.dtype)
                    target_tensor_applied = 1
                    tensors.append(t)
                else: # control tensor
                    c = self.compute_control_tensor(self.state_dims[q], control_vals[0], "rank-3", "down")
                    c = self.backend.asarray(c, dtype=self.dtype)
                    tensors.append(c)

            elif i == n_qudits-1: # rank-3
                if q == target_modes[0]: # target tensor
                    target = target_data.reshape(1, -1)[0]
                    t_dim = self.state_dims[q]
                    c0 = [0] * (t_dim ** 2)
                    for i in range(t_dim):
                        idx = i + (i * t_dim)
                        c0[idx] = 1
                    c0 = self.backend.asarray(c0)
                    # modes (p, A, a), i.e, virtual, bra, ket
                    t_temp = self.backend.vstack([c0, target])
                    t_temp = t_temp.reshape(2, t_dim, t_dim)
                    # convert to modes (p, a, A), i.e, virtual, ket, bra
                    try:
                        t_temp = t_temp.transpose(0, 2, 1)
                    except TypeError:
                        t_temp = t_temp.permute(0, 2, 1)
                    t = self.backend.asarray(t_temp, dtype=self.dtype)
                    target_tensor_applied = 1
                    tensors.append(t)
                else: # control tensor
                    c = self.compute_control_tensor(self.state_dims[q], control_vals[i-1], "rank-3", "up")
                    c = self.backend.asarray(c, dtype=self.dtype)
                    tensors.append(c)

            else: # rank-4
                if q == target_modes[0]: # target tensor
                    target = target_data.reshape(1, -1)[0]
                    t_dim = self.state_dims[q]
                    c0 = [0] * (t_dim ** 2)
                    for i in range(t_dim):
                        idx = i + (i * t_dim)
                        c0[idx] = 1
                    c0 = self.backend.asarray(c0)
                    t_temp = self.backend.vstack([c0, c0, c0, target])
                    # modes (p, n, A, a)
                    t_temp = t_temp.reshape(2, 2, t_dim, t_dim)
                    # convert to (p, a, n, A)
                    try:
                        t_temp = t_temp.transpose(0, 3, 1, 2)
                    except TypeError:
                        t_temp = t_temp.permute(0, 3, 1, 2)
                    t = self.backend.asarray(t_temp, dtype=self.dtype)
                    target_tensor_applied = 1
                    tensors.append(t)
                else: # control tensor
                    if target_tensor_applied == 1:
                        c = self.compute_control_tensor(self.state_dims[q], control_vals[i-1], "rank-4", "up")
                        c = self.backend.asarray(c, dtype=self.dtype)
                        tensors.append(c)
                    else:
                        c = self.compute_control_tensor(self.state_dims[q], control_vals[i], "rank-4", "down")
                        c = self.backend.asarray(c, dtype=self.dtype)
                        tensors.append(c)

        return tensors


    def get_sv_contraction_expression(self):
        operands = []
        # initial qudit mode
        qudit_modes = list(range(self.num_qudits))
        mode_frontier = self.num_qudits

        initial_state = self.get_initial_state()
        if self.initial_mps_dim is None:
            for i, t in enumerate(initial_state):
                operands += [t, [qudit_modes[i]]]            
        else:
            prev_mode = None
            for i, t in enumerate(initial_state):
                if i == 0:
                    modes = (qudit_modes[i], mode_frontier)
                elif i == self.num_qudits - 1:
                    modes = (prev_mode, qudit_modes[i])
                else:
                    modes = (prev_mode, qudit_modes[i], mode_frontier)
                prev_mode = mode_frontier
                mode_frontier += 1
                operands += [t, modes]
        
        for op, qudits, gate_info in self.sequence:
            if gate_info is not None:
                if 'control_values' in gate_info and 'control_modes' in gate_info:
                    # convert control tensor into MPO
                    ctrl_modes, ctrl_vals = gate_info['control_modes'], gate_info['control_values']
                    op = self.compute_ct_mpo_tensors(ctrl_modes, ctrl_vals, qudits, op)
                    qudits = qudits + ctrl_modes
                    qudits = sorted(qudits)
                    gate_info = None
                else:
                    raise RuntimeError("Not the expected code path")
            n_qudits = len(qudits)
            if isinstance(op, (list, tuple)):
                # for MPO
                prev_mode = None
                for i, q in enumerate(qudits):
                    mode_in = qudit_modes[q]
                    qudit_modes[q] = mode_frontier 
                    mode_frontier += 1
                    if i == 0:
                        modes = [mode_in, mode_frontier, qudit_modes[q]]
                        prev_mode = mode_frontier
                        mode_frontier += 1
                    elif i == n_qudits - 1:
                        modes = [prev_mode, mode_in, qudit_modes[q]]
                    else:
                        modes = [prev_mode, mode_in, mode_frontier, qudit_modes[q]]
                        prev_mode = mode_frontier
                        mode_frontier +=1
                    operands += [op[i], modes]
            else:
                modes_in = [qudit_modes[q] for q in qudits]
                modes_out = []
                for q in qudits:
                    qudit_modes[q] = mode_frontier
                    modes_out.append(mode_frontier)
                    mode_frontier +=1
                operands += [op, modes_out + modes_in]
        
        operands.append(qudit_modes)
        return operands

    def to_network_state(self, *, config=None, options=None):
        network_state = NetworkState(self.state_dims, dtype=self.dtype, config=config, options=options)
        if self.initial_mps_dim is not None:
            network_state.set_initial_mps(self.get_initial_state())
        apply_factory_sequence(network_state, self.sequence)
        return network_state


class MPS(_BaseComputeEngine):

    def __init__(
        self, 
        qudits,
        backend,
        qudit_dims=2,
        dtype='complex128',
        mps_tensors=None,
        gauge_option='free',
        mpo_application='approximate',
        canonical_center=None,
        sample_rng=DEFAULT_RNG,
        **svd_options
    ):
        self._qudits = list(qudits)
        self.backend = get_or_create_tensor_backend(backend)
        if isinstance(qudit_dims, (int, np.integer)):
            self.state_dims = (qudit_dims, ) * len(qudits)
        else:
            assert len(qudit_dims) == len(qudits), "qudit_dims must be either an integer or a sequence of integers with the same size as qudits"
            self.state_dims = tuple(qudit_dims)
        self.dtype = get_dtype_name(dtype)
        self.n = len(qudits)
        if mps_tensors is None:
            self.mps_tensors = []
            for i in range(self.n):
                data = [1, ] + [0, ] * (self.state_dims[i] - 1)
                t = self.backend.asarray(data, dtype=dtype).reshape(1, self.state_dims[i], 1)
                self.mps_tensors.append(t)
        else:
            # avoid in place modification
            self.mps_tensors = mps_tensors.copy()
            # potentially insert dummy labels for boundary tensors for consistent notation in this class
            if self.mps_tensors[0].ndim == 2:
                new_shape = (1, ) + self.mps_tensors[0].shape
                self.mps_tensors[0] = self.mps_tensors[0].reshape(*new_shape)
            if self.mps_tensors[-1].ndim == 2:
                new_shape = self.mps_tensors[-1].shape + (1, ) 
                self.mps_tensors[-1] = self.mps_tensors[-1].reshape(*new_shape)

        if canonical_center is not None:
            assert canonical_center >= 0 and canonical_center < self.n
        self.canonical_center = canonical_center
        self.sample_rng = sample_rng

        self.gauge_option = gauge_option 

        for key in svd_options.keys():
            if key not in MPS_VALID_CONFIGS:
                raise ValueError(f"{key} not supported")
        self.svd_options = {'partition': None if self.gauge_option == 'simple' else 'UV'}
        self.svd_options.update(svd_options)
        self.is_exact_svd = self.svd_options.get('normalization', None) is None
        for key in ('abs_cutoff', 'rel_cutoff', 'discarded_weight_cutoff'):
            self.is_exact_svd = self.is_exact_svd and self.svd_options.get(key, None) in (0, None)
        max_extent = self.svd_options.pop('max_extent', None)
        self.max_extents = []
        for i in range(self.n-1):
            max_shared_extent = min(np.prod(self.state_dims[:i+1]), np.prod(self.state_dims[i+1:]))
            extent = max_shared_extent if max_extent is None else min(max_extent, max_shared_extent)
            self.max_extents.append(extent)

        assert mpo_application in {'exact', 'approximate'}
        self.mpo_application = mpo_application
        self._tolerance = None
        self.sv = None
        self.norm = None 
        
        self.gauges = dict()
        if self.gauge_option == 'simple':
            # First canonicalization sweep to generate a left canonical MPS representation without gauges 
            self._minimal_compression(0, self.n-1, False, check_minimal=False)
            # Make it inverse canonical
            self._make_canonical()
        else: # gauge_option is 'free'
            # To generate a left canonical MPS representation without gauges
            self._minimal_compression(0, self.n-1, False, check_minimal=True) 

    @property
    def qudits(self):
        return self._qudits
    
    @property
    def qubits(self):
        # overload
        return self.qudits
    
    def __getitem__(self, key):
        assert key >= 0 and key < self.n
        return self.mps_tensors[key]
    
    def __setitem__(self, key, val):
        assert key >= 0 and key < self.n
        self.mps_tensors[key] = val
        # resetting SV and norm
        self.sv = self.norm = None
    
    @property
    def tolerance(self):
        if self._tolerance is None:
            self._tolerance = get_mps_tolerance(self.dtype)
        return self._tolerance
    
    def _compute_state_vector(self):
        self.absorb_gauges(self.mps_tensors)
        inputs = []
        output_modes = []
        for i, o in enumerate(self.mps_tensors):
            modes = [2*i, 2*i+1, 2*i+2]
            inputs.extend([o, modes])
            output_modes.append(2*i+1)
        inputs.append(output_modes)
        return oe.contract(*inputs)
    
    def compute_expectation(self, tn_operator):
        def _get_array(wrapped_tensor):
            if self.backend.full_name == 'numpy':
                return wrapped_tensor.tensor.get()
            elif self.backend.full_name == 'torch-cpu':
                return wrapped_tensor.to('cpu')
            return wrapped_tensor.tensor
        if isinstance(tn_operator, NetworkOperator):
            expec = 0
            for (mpo_tensors, modes, coeff) in tn_operator.mpos:
                mpo_tensors = [_get_array(o) for o in mpo_tensors]
                expec += coeff * self._get_mpo_expectation(mpo_tensors, modes)
            for (prod_tensors, modes, coeff) in tn_operator.tensor_products:
                prod_tensors = [_get_array(o) for o in prod_tensors]
                expec += coeff * self._get_product_expectation(prod_tensors, modes)
            return expec
        else:
            # pauli string or pauli string dictionaries
            return super().compute_expectation(tn_operator)
        
    def _get_mpo_expectation(self, mpo_tensors, mpo_modes):
        sv = self.compute_state_vector()
        modes = list(range(self.n))
        mode_frontier = self.n
        current_modes = modes.copy()
        operands = [sv, modes]
        prev_mode = None
        for i, m in enumerate(mpo_modes):
            ket_mode = current_modes[m]
            current_modes[m] = bra_mode = mode_frontier
            mode_frontier += 1
            next_mode = mode_frontier
            mode_frontier += 1
            if i == 0:
                operands += [mpo_tensors[i], (ket_mode, next_mode, bra_mode)]
            elif i == len(mpo_modes) - 1:
                operands += [mpo_tensors[i], (prev_mode, ket_mode, bra_mode)]
            else:
                operands += [mpo_tensors[i], (prev_mode, ket_mode, next_mode, bra_mode)]
            prev_mode = next_mode
        operands += [sv.conj(), current_modes]
        return oe.contract(*operands).item()

    def _get_product_expectation(self, prod_tensors, prod_modes):
        sv = self.compute_state_vector()
        modes = list(range(self.n))
        mode_frontier = self.n
        current_modes = modes.copy()
        operands = [sv, modes]
        for i, ms in enumerate(prod_modes):
            ket_modes = []
            bra_modes = []
            for m in ms:
                ket_modes.append(current_modes[m])
                bra_modes.append(mode_frontier)
                current_modes[m] = mode_frontier
                mode_frontier += 1
            operands += [prod_tensors[i], ket_modes + bra_modes]
        operands += [sv.conj(), current_modes]
        return oe.contract(*operands).item()
    
    def _swap(self, i, direction, exact=False):
        # swap function absorbing all 3 gauges instead of only common neighbor 
        assert direction in {'left', 'right'}
        if direction == 'left':
            a, b = i-1, i
        else:
            a, b = i, i+1
        assert a >= 0 and b <= self.n-1

        s_left = self.gauges.get(a, None)
        s = self.gauges.pop(b, None)
        s_right = self.gauges.get(b+1, None)
        # absorb 3 gauges
        self.mps_tensor_absorb_gauge(a, s_left, direction='left')
        self.mps_tensor_absorb_gauge(a, s, direction='right')
        self.mps_tensor_absorb_gauge(b, s_right, direction='right')

        extra_mode_a = 'xyzw'[:self[a].ndim-3]
        extra_mode_b = 'XYZW'[:self[b].ndim-3]
        input_a = f'iP{extra_mode_a}j'
        input_b = f'jQ{extra_mode_b}l'
        intm = 'iPQl' + extra_mode_a + extra_mode_b
        output_a = f'iQ{extra_mode_b}j'
        output_b = f'jP{extra_mode_a}l'

        tmp = self.backend.einsum(f'{input_a},{input_b}->{intm}', self[a], self[b])  #contract to form the tensor
        decompose_expr = f'{intm}->{output_a},{output_b}'
        size_dict = dict(zip(input_a, self[a].shape))
        size_dict.update(dict(zip(input_b, self[b].shape)))
        mid_extent = compute_mid_extent(size_dict, (input_a, input_b), (output_a, output_b))
        if exact:
            svd_options = {'max_extent': mid_extent, 'partition': None if self.gauge_option == 'simple' else 'UV'}
        else:
            svd_options = self.svd_options.copy()
            svd_options['max_extent'] = min(self.max_extents[a], mid_extent)

        self[a], s, self[b] = tensor_decompose(decompose_expr, tmp, method='svd', **svd_options)
        if s is not None:
            self.gauges[b] = self.backend.asarray(s)

        # remove gauge effect back
        self.mps_tensor_absorb_gauge(a, s_left, direction='left', inverse=True)
        self.mps_tensor_absorb_gauge(b, s_right, direction='right', inverse=True)

    def _canonicalize_site(self, i, direction, max_extent=None, **svd_options):
        if direction not in {'right', 'left'}:
            raise ValueError("Direction must be 'right' or 'left'")
    
        if direction == 'right':
            assert i >= 0 and i < self.n - 1
            left, right = i, i+1
            partition = 'V'
            # absorb the left gauge on tensor i
            gauge = self.gauges.pop(i, None) # remove this gauge from the state
            self.mps_tensor_absorb_gauge(i, gauge, direction='left')
            # absorb the shared gauge
            gauge = self.gauges.pop(i+1, None) # remove this gauge from the state
            self.mps_tensor_absorb_gauge(i+1, gauge, direction='left')
        else: # direction == 'left'
            assert i > 0 and i <= self.n - 1
            left, right = i-1, i
            partition = 'U'
            # absorb the right gauge on tensor i
            gauge = self.gauges.pop(i+1, None) # remove this gauge from the state
            self.mps_tensor_absorb_gauge(i, gauge, direction='right')
            # absorb the shared gauge
            gauge = self.gauges.pop(i, None) # remove this gauge from the state
            self.mps_tensor_absorb_gauge(i-1, gauge, direction='right')

        ti, tj = self[left], self[right]
        qr_min_extent_left = min(np.prod(ti.shape[:2]), ti.shape[-1])
        qr_min_extent_right = min(np.prod(tj.shape[1:]), tj.shape[0])
        min_exact_e = min(qr_min_extent_left, qr_min_extent_right)
        max_extent = min(max_extent, min_exact_e) if max_extent is not None else min_exact_e
        svd_options.pop('partition', None)
        if not svd_options and direction == 'right' and max_extent == qr_min_extent_left:
            self[left], r = tensor_decompose('ipj->ipx,xj', ti, method='qr')
            self[right] = self.backend.einsum('xj,jql->xql', r, tj)
        elif not svd_options and direction == 'left' and max_extent == qr_min_extent_right:
            self[right], r = tensor_decompose('jql->xql,jx', tj, method='qr')
            self[left] = self.backend.einsum('jx,ipj->ipx', r, ti)
        else:
            svd_options['partition'] = partition
            tmp = self.backend.einsum('ipj,jql->ipql', ti, tj)
            self[left], _, self[right] = tensor_decompose('ipql->ipj,jql', tmp, method='svd', max_extent=max_extent, **svd_options)
    
    def _compress_with_gauges(self, i, direction, max_extent=None, **svd_options):
        if direction not in {'right', 'left'}:
            raise ValueError("Direction must be 'right' or 'left'")

        if direction == 'right':
            assert i >= 0 and i < self.n - 1
            left, right = i, i+1
        else: # direction == 'left'
            assert i > 0 and i <= self.n - 1
            left, right = i-1, i
        
        s_left = self.gauges.get(left, None)
        s = self.gauges.pop(right, None)
        s_right = self.gauges.get(right+1, None)
        
        self.mps_tensor_absorb_gauge(left, s_left, direction='left')
        self.mps_tensor_absorb_gauge(left, s, direction='right')
        self.mps_tensor_absorb_gauge(right, s_right, direction='right')

        svd_options = svd_options.copy()
        svd_options['partition'] = None

        ti, tj = self[left], self[right]
        qr_min_extent_right = min(np.prod(ti.shape[:2]), ti.shape[-1])
        qr_min_extent_left = min(np.prod(tj.shape[1:]), tj.shape[0])
        min_exact_e = min(qr_min_extent_left, qr_min_extent_right)
        max_extent = min(max_extent, min_exact_e) if max_extent is not None else min_exact_e

        tmp = self.backend.einsum('ipj,jql->ipql', ti, tj)
        self[left], g, self[right] = tensor_decompose('ipql->ipj,jql', tmp, method='svd', max_extent=max_extent, **svd_options)
        self.gauges[right] = self.backend.asarray(g)
        # remove gauge effect
        self.mps_tensor_absorb_gauge(left, s_left, direction='left', inverse=True)
        self.mps_tensor_absorb_gauge(right, s_right, direction='right', inverse=True)
    
    def _minimal_compression(self, start, end, keep_gauges, *, check_minimal=False):
        if check_minimal:
            manageable = True
            for i in range(start, end+1):
                if i == self.n - 1:
                    break
                left_extent, shared_extent, right_extent = np.prod(self[i].shape[:2]), self[i].shape[-1], np.prod(self[i+1].shape[1:]) 
                shared_extent_manageble = shared_extent == min(left_extent, right_extent, shared_extent)
                manageable = manageable and shared_extent_manageble and shared_extent <= self.max_extents[i]
                if not manageable:
                    break
            if manageable:
                return

        for i in range(start, end+1):
            if i == self.n - 1: break
            if keep_gauges:
                self._compress_with_gauges(i, 'right') # keep gauges
            else:
                self._canonicalize_site(i, 'right') # remove gauges

        for i in range(end, start-1, -1):
            if i==0: break
            if keep_gauges:
                self._compress_with_gauges(i, 'left')
            else:
                self._canonicalize_site(i, 'left')  

    def _make_canonical(self, **svd_options):
        max_extent = svd_options.get("max_extent", None)
        for i in range(len(self.mps_tensors)-1):    
            # Decompose tensor using SVD
            self[i], s, V = tensor_decompose('ijk->ijm,mk', self[i], method='svd', max_extent=max_extent, partition=None)
            # Update gauge and tensor ( self[i] = U*S ---> self[i+1] = s * V * self[i+1])
            self.gauges[i+1] = self.backend.asarray(s) 
            prev_gauge = self.gauges.get(i,None)
            if prev_gauge is not None:
                self.mps_tensor_absorb_gauge(i, prev_gauge, direction='left', inverse=True)
            if i != len(self.mps_tensors) - 2:
                # For the last sites, no need to do absorb S
                V = self.backend.einsum('i,ij->ij', s, V)
            self[i+1] = self.backend.einsum('jlm,ij->ilm', self[i+1], V) 

    def mps_tensor_absorb_gauge(self, site, s, *, direction='left', inverse=False):
        assert direction in {'left', 'right'}
        if s is not None:
            if inverse:
                #TODO: consolidate epsilon
                s = self.backend.where(s < np.finfo(self.dtype).eps, 0.0, 1/s)
            if self[site].ndim == 3: 
                subscripts = {'left': 'i', 'right': 'k'}[direction] + ',ijk->ijk'
            elif self[site].ndim == 4: 
                subscripts = {'left': 'i', 'right': 'k'}[direction] + ',ijxk->ijxk'
            else: 
                raise ValueError(f"Unsupported number of dimensions ({self[site].ndim}) in self[site]")
            self[site] = self.backend.einsum(subscripts, s, self[site])
        return

    def _apply_gate_1q(self, i, operand):
        self[i] = self.backend.einsum('ipj,Pp->iPj', self[i], operand)
    
    def _apply_gate_2q(self, i, j, operand):
        if i > j:
            try:
                operand = operand.transpose(1,0,3,2)
            except TypeError:
                operand = operand.permute(1,0,3,2)
            return self._apply_gate_2q(j, i, operand)
        elif i == j:
            raise ValueError(f"gate acting on the same site {i} twice")
        elif i == j - 1:
            # two adjacent qubits
            sa = self.gauges.get(i, None) # left gauge of i 
            s = self.gauges.get(i+1, None) # shared gauge of i, j
            sb = self.gauges.get(j+1, None) # right gauge of j

            # absorb all gauges before contract & decompose
            self.mps_tensor_absorb_gauge(i, sa, direction="left")  #put on first tensor
            self.mps_tensor_absorb_gauge(i, s, direction="right")  #put on first tensor 
            self.mps_tensor_absorb_gauge(j, sb, direction="right") #put on second tensor

            size_dict = dict(zip('ipjjqkPQpq', self[i].shape + self[j].shape + operand.shape))

            mid_extent = compute_mid_extent(size_dict, ('ipj','jqk','PQpq'), ('iPj','jQk'))
            max_extent = min(mid_extent, self.max_extents[i]) 

            self[i], s, self[j] = gate_decompose('ipj,jqk,PQpq->iPj,jQk', self[i], self[j], operand, max_extent=max_extent, **self.svd_options)
            if s is not None:
                self.gauges[i+1] = self.backend.asarray(s)

            # remove gauge effect back
            self.mps_tensor_absorb_gauge(i, sa, direction="left", inverse=True)
            self.mps_tensor_absorb_gauge(j, sb, direction="right", inverse=True)

        else:
            # insert swap gates recursively
            swaps = []
            while (j != i+1):
                self._swap(i, 'right', False)
                swaps.append([i, 'right'])
                i += 1
                if (j == i+1):
                    break
                self._swap(j, 'left', False)
                swaps.append([j, 'left'])
                j -= 1
            self._apply_gate_2q(i, j, operand)
            for x, direction in swaps[::-1]:
                self._swap(x, direction=direction)
    
    def apply_gate(self, qudits, operand):
        gauge_option = self.gauge_option             
        if gauge_option == 'simple':
            assert self.svd_options['partition'] is None, "For MPS with gauges, SVD partition must be set to None"
        else:
            assert self.svd_options['partition'] == 'UV', "For MPS without gauges, SVD partition must be set to UV"

        sites = [self.qudits.index(q) for q in qudits]
        if len(sites) == 1:
            return self._apply_gate_1q(*sites, operand)
        elif len(sites) == 2:
            return self._apply_gate_2q(*sites, operand)
        else:
            raise NotImplementedError("Only single- and two- qubit gate supported")
    
    def absorb_gauges(self, mps_tensors):
        """
        Note that this method should only be called after all gates have been applied 
        and one is ready to move onto property computation phase
        """          
        for i, o in enumerate(mps_tensors):
            s = self.gauges.pop(i, None)
            self.mps_tensor_absorb_gauge(i, s, direction='left')
        return self.mps_tensors

    def apply_mpo(self, qudits, mpo_operands):
        # map from site to the associated qudit id
        qudits_order = list(range(self.n))
        sites = [self.qudits.index(q) for q in qudits]
        # Step 1: Boundary Contraction
        # 
        # ---X---Y---Z---W---            Y---Z---W---
        #    |   |   |   |     ->    | / |   |   |
        # ---A---B---C---D---     ---a---B---C---D---
        # 
        a = sites[0]
        self[a] = self.backend.einsum('ipj,pxP->iPxj', self[a], mpo_operands[0])
        exact_mpo = self.mpo_application == 'exact'
        svd_options = {'partition': None if self.gauge_option == 'simple' else 'UV'} if exact_mpo else self.svd_options.copy()
        def record_swap(i, direction):
            # utility function to record the current qudits_order after swap operation, no computation is done
            j = i + 1 if direction == 'right' else i - 1
            qudits_order[i], qudits_order[j] = qudits_order[j], qudits_order[i]
        # Step 2: Contract-Decompose for all remaining sites, swaps inserted if needed
        # 
        #        Y---Z---W---                Z---W---
        #    | / |   |   |     ->    |   | / |   |
        # ---a---B---C---D---     ---a---b---C---D---
        num_mpo_sites = len(qudits)
        for i in range(num_mpo_sites-1):
            operand = mpo_operands[i+1]
            qa = sites[i]
            qb = sites[i+1]
            q0 = qudits_order.index(qa)
            q1 = qudits_order.index(qb)
            forward_order = q1 > q0
            q0, q1 = sorted([q0, q1])
            while (q1 != q0 + 1):
                self._swap(q0, 'right', exact_mpo)
                record_swap(q0, 'right')
                q0 += 1
                if (q1 == q0+1):
                    break
                self._swap(q1, 'left', exact_mpo)
                record_swap(q1, 'left')
                q1 -= 1
            if not forward_order:
                # revert to original ordering
                q0, q1 = q1, q0
            explict_swap = False
            if i != num_mpo_sites - 2:
                q2 = qudits_order.index(sites[i+2])
                dis_02 = abs(q2-q0)
                dis_12 = abs(q2-q1)
                # if next mpo tensor is closer to q0, use contract decompose to perform swap
                explict_swap = dis_02 < dis_12
                if explict_swap:
                    record_swap(q0, 'right' if q1 > q0 else 'left')
            if forward_order:
                if i == num_mpo_sites - 2:
                    expr = 'iPxj,jql,xqQ->iPQl'
                    decompose_expr = 'iPQl->iPj,jQl'
                else:
                    expr = 'iPxj,jql,xqyQ->iPQly'
                    decompose_expr = 'iPQly->iQyj,jPl' if explict_swap else 'iPQly->iPj,jQyl'
            else:
                if i == num_mpo_sites - 2:
                    expr = 'jPxl,iqj,xqQ->iPQl'
                    decompose_expr = 'iPQl->jPl,iQj'
                else:
                    expr = 'jPxl,iqj,xqyQ->iPQly'
                    decompose_expr = 'iPQly->jQyl,iPj' if explict_swap else 'iPQly->jPl,iQyj'

            q_left, q_right = (q0, q1) if forward_order else (q1, q0)
            sa = self.gauges.get(q_left, None)  
            s = self.gauges.pop(q_right, None) 
            sb = self.gauges.get(q_right + 1, None) 

            self.mps_tensor_absorb_gauge(q_left, sa, direction="left")
            self.mps_tensor_absorb_gauge(q_left, s, direction="right")
            self.mps_tensor_absorb_gauge(q_right, sb, direction="right")

            tmp = self.backend.einsum(expr, self[q0], self[q1], operand)
            inputs  = expr.split('->')[0].split(',')
            outputs =  decompose_expr.split('->')[1].split(',')
            size_dict = dict(zip(inputs[0], self[q0].shape))
            size_dict.update(dict(zip(inputs[1], self[q1].shape)))
            size_dict.update(dict(zip(inputs[2], operand.shape)))
            mid_extent = compute_mid_extent(size_dict, inputs, outputs)
            if exact_mpo:
                max_extent = mid_extent
            else:
                max_extent = min(mid_extent, self.max_extents[min(q0,q1)])

            self[q0], s, self[q1] = tensor_decompose(decompose_expr, tmp, method='svd', max_extent=max_extent, **svd_options)
            if s is not None:
                self.gauges[q_right] = self.backend.asarray(s)

            self.mps_tensor_absorb_gauge(q_left, sa, direction="left", inverse=True)
            self.mps_tensor_absorb_gauge(q_right,sb, direction="right", inverse=True)

        # swap operations to revert back to the original ordering
        for i in range(self.n):
            site = qudits_order.index(i)
            while (site != i):
                self._swap(site, 'left', exact_mpo)
                record_swap(site, 'left')
                site -= 1
        # sanity check to make sure original ordering is obtained
        assert qudits_order == list(range(self.n))
        if self.mpo_application == 'exact':
            # when MPO is applied in exact fashion, bond may exceed the maximal required
            # TODO: Handle rare case where mpo_application is exact for MPS-MPO contraction but not for gate application process
            keep_gauges = self.gauge_option == 'simple'
            self._minimal_compression(min(qudits), max(qudits), keep_gauges, check_minimal=False)

    @classmethod
    def from_converter(cls, converter, **kwargs):
        dtype = get_dtype_name(converter.dtype)
        mps = cls(converter.qubits, converter.backend.__name__, dtype=dtype, **kwargs)
        for operand, qubits in converter.gates:
            if len(qubits) > 2:
                return None
            mps.apply_gate(qubits, operand)
        mps.canonicalize()
        return mps
    
    @classmethod
    def from_circuit(cls, circuit, backend, dtype='complex128', **kwargs):
        converter = CircuitToEinsum(circuit, backend=backend, dtype=dtype)
        return cls.from_converter(converter, **kwargs)
    
    @classmethod
    def from_factory(cls, factory, **kwargs):
        qudits = list(range(factory.num_qudits))
        if factory.initial_mps_dim is not None:
            mps_tensors = factory.get_initial_state()
        else:
            mps_tensors = None
        qudit_dims = factory.state_dims
        mps = cls(qudits, factory.backend, qudit_dims=qudit_dims, mps_tensors=mps_tensors, dtype=factory.dtype, **kwargs)
        for op, modes, gate_info in factory.sequence:
            if gate_info is None:
                if isinstance(op, (list, tuple)):
                    # MPO
                    mps.apply_mpo(modes, op)
                else:
                    # Gate
                    mps.apply_gate(modes, op)
            else:
                if 'control_values' in gate_info and 'control_modes' in gate_info:
                    ctrl_modes, ctrl_vals = gate_info['control_modes'], gate_info['control_values']
                    ct_tensors = factory.compute_ct_mpo_tensors(ctrl_modes, ctrl_vals, modes, op) 
                    new_modes = modes + ctrl_modes
                    new_modes = sorted(new_modes)
                    mps.apply_mpo(new_modes, ct_tensors)
                else:
                    raise RuntimeError("Not expected code path")
        mps.canonicalize()
        return mps
    
    def print(self):
        print([o.shape[2] for o in self.mps_tensors[:-1]])
    
    def canonicalize(self):
        center = self.canonical_center
        if center is None:
            for i in range(self.n-1):
                shared_e = self[i].shape[-1]
                max_e = self.max_extents[i]
                if (shared_e > max_e):
                    self._canonicalize_site(i, 'right', max_extent=max_e, **self.svd_options)
            return

        for i in range(center):
            self._canonicalize_site(i, 'right', max_extent=self.max_extents[i], **self.svd_options)
            
        for i in range(self.n-1, center, -1):
            self._canonicalize_site(i, 'left', max_extent=self.max_extents[i-1], **self.svd_options)


class ExactStateAPITester(BaseTester):

    @classmethod
    def from_circuit(cls, circuit, dtype, mps_options=EMPTY_DICT, backend='cupy', handle=None, rng=DEFAULT_RNG, **kwargs):
        converter = CircuitToEinsum(circuit, dtype, backend=backend)
        reference_engine = ConverterComputeEngine(converter, backend=backend, handle=handle, sample_rng=rng)
        target_engines = []
        algorithm = mps_options.pop('algorithm', 'gesvd')
        if mps_options:
            raise ValueError("Exact tests should exclude approximate settings")
        #TODO: add sv type TN computation
        for config in (TNConfig(), MPSConfig(algorithm=algorithm, mpo_application='exact', gauge_option='free'), MPSConfig(mpo_application='exact', gauge_option='simple')):
            if isinstance(config, MPSConfig) and not is_converter_mps_compatible(converter):
                # NOTE: MPS simulation are only functioning if no multicontrol gates exist in the circuit. 
                continue
            engine = NetworkState.from_converter(converter, config=config, options={'handle': handle})
            target_engines.append(engine)
        mps_cp_engine = MPS.from_converter(converter, sample_rng=rng, mpo_application='exact', gauge_option='free')
        if mps_cp_engine is not None:
            target_engines.append(mps_cp_engine)
        mps_cp_engine_su = MPS.from_converter(converter, sample_rng=rng, mpo_application='exact', gauge_option='simple')
        if mps_cp_engine_su is not None:
            target_engines.append(mps_cp_engine_su)
        
        return cls(reference_engine, target_engines, converter=converter, rng=rng, **kwargs)

    def test_misc(self):
        norm = self.reference_engine.compute_norm()
        assert np.allclose(norm, 1, **self.reference_engine.tolerance)
    
    def test_expectation(self):
        if self.reference_engine.dtype.startswith('complex'):
            pauli_strings = get_random_pauli_strings(self.n_qubits, 6, rng=self.rng)
            expec1 = self.reference_engine.compute_expectation(pauli_strings)        
            for engine in self.target_engines:
                if isinstance(engine, (NetworkState, MPS)):
                    expec2 = engine.compute_expectation(pauli_strings)
                    message = f"{engine.__class__.__name__} maxDiff={abs(expec1-expec2)}, tol={engine.tolerance}"
                    assert np.allclose(expec1, expec2, **engine.tolerance), message
                else:
                    raise ValueError


class ApproximateMPSTester(ExactStateAPITester):

    @classmethod
    def from_converter(cls, converter, mps_options, handle=None, rng=DEFAULT_RNG, **kwargs):
        reference_svd_options = {}
        for key, value in mps_options.items():
            # not all options are supported in reference MPS
            if key in MPS_VALID_CONFIGS:
                reference_svd_options[key] = value

        reference_engine = MPS.from_converter(converter, sample_rng=rng, **reference_svd_options)
        target_engines = [NetworkState.from_converter(converter, config=mps_options, options={'handle': handle})]
        return cls(reference_engine, target_engines, converter=converter, rng=rng, **kwargs)

    
    @classmethod
    def from_factory(cls, factory, mps_options, handle=None, rng=DEFAULT_RNG, **kwargs):
        reference_svd_options = {}
        for key, value in mps_options.items():
            # not all options are supported in reference MPS
            if key in MPS_VALID_CONFIGS:
                reference_svd_options[key] = value
        reference_engine = MPS.from_factory(factory, sample_rng=rng, **reference_svd_options)
        target_engines = [factory.to_network_state(config=mps_options, options={'handle': handle})]
        return cls(reference_engine, target_engines, converter=None, state_dims=factory.state_dims, rng=rng, **kwargs)

    def test_sampling(self):
        super().test_sampling()
        for engine in self.all_engines:
            if isinstance(engine, NetworkState):
                samples_0 = engine.compute_sampling(88, seed=23)
                samples_1 = engine.compute_sampling(88, seed=23)
                # determinism check
                assert samples_0 == samples_1
    
    def test_expectation(self):
        if self.is_qubit_system:
            super().test_expectation()
        
        operator = get_random_network_operator(self.state_dims, backend=self.backend.__name__, rng=self.rng, dtype=self.reference_engine.dtype)
        expec1 = self.reference_engine.compute_expectation(operator)        
        for engine in self.target_engines:
            for _ in get_engine_iters(engine):
                expec2 = engine.compute_expectation(operator)
                message = f"{engine.__class__.__name__} maxDiff={abs(expec1-expec2)}, tol={engine.tolerance}"
                assert np.allclose(expec1, expec2, **engine.tolerance), message
    
    def test_misc(self):
        canonical_center = self.reference_engine.canonical_center
        verify_mps_canonicalization(self.reference_engine.mps_tensors, canonical_center)
        mps_tensors = self.target_engines[0].compute_output_state()
        verify_mps_canonicalization(mps_tensors, canonical_center)


class NetworkStateFunctionalityTester:
    """Basic functionality tests"""

    def __init__(self, state, backend, is_normalized=False):
        self.state = state
        self.backend = backend
        self.package = self.backend.split('-')[0]
        self.is_normalized = is_normalized
        self.n = state.n
        self.dtype = state.dtype
        self.tolerance = {'atol': atol_mapper[self.dtype], 
                          'rtol': rtol_mapper[self.dtype]}
    
    def _check_tensor(self, tensor, shape=None):
        if shape is not None:
            assert tensor.shape == tuple(shape)
        assert infer_object_package(tensor) == self.package
        assert str(tensor.dtype).split('.')[-1] == self.dtype
        o = tensor_wrapper.wrap_operand(tensor)
        assert o.device_id == (None if self.backend in {'numpy', 'torch-cpu'} else self.state.device_id)

    def run_tests(self):
        # SV
        if self.state.config.__class__.__name__ == 'MPSConfig':
            mps_tensors = self.state.compute_output_state()
            for o in mps_tensors:
                self._check_tensor(o)
    
        sv, norm_0 = self.state.compute_state_vector(return_norm=True)
        self._check_tensor(sv, shape=self.state.state_mode_extents)

        # amplitude
        amplitude, norm_1 = self.state.compute_amplitude('0' * self.n, return_norm=True)
        assert np.allclose(sv.ravel()[0].item(), amplitude, **self.tolerance)

        # batched_amplitude
        fixed = {0:1, 1:0} if self.n > 2 else {0:1}
        batched_amplitude, norm_2 = self.state.compute_batched_amplitudes(fixed, return_norm=True)
        self._check_tensor(batched_amplitude, shape=self.state.state_mode_extents[len(fixed):])

        # RDM
        rdm = self.state.compute_reduced_density_matrix((0, ), fixed={1: 0})
        self._check_tensor(rdm, shape=(self.state.state_mode_extents[0], ) * 2)

        # sampling
        modes = (0, 1) if self.n >= 2 else (0, )
        samples_0 = self.state.compute_sampling(123, modes=modes, seed=3)
        samples_1 = self.state.compute_sampling(123, modes=modes, seed=3)
        assert samples_0 == samples_1 # deterministic checking
        assert all([len(bitstring) == len(modes) for bitstring in samples_0.keys()])
        assert sum(samples_0.values()) == 123

        norm_ref = 1 if self.is_normalized else (abs(sv) ** 2).sum().item()

        # expectation
        if (set(self.state.state_mode_extents) == set([2, ])) and self.state.dtype.startswith('complex'):
            expectation, norm_3 = self.state.compute_expectation('I' * self.n, return_norm=True)
            assert np.allclose(expectation, norm_ref, **self.tolerance)
        else:
            norm_3 = None
        
        # norm
        for norm in (norm_0, norm_1, norm_2, norm_3):
            if norm is not None:
                assert np.allclose(norm, norm_ref, **self.tolerance)

def channel_infos_to_full_configuration(channel_info):
    if not channel_info:
        # No channels in the system
        return [None]
    channel_ids, channel_ops = zip(*sorted(channel_info.items()))
    full_configurations = []
    for op_ids in itertools.product(*[range(len(ops)) for ops, _ in channel_ops]):
        channels = []
        for channel_id, (ops, _), op_id in zip(channel_ids, channel_ops, op_ids):
            channels.append([channel_id, op_id, ops[op_id]])
        full_configurations.append(channels)
    return full_configurations

class NetworkStateChannelTester:
    """
    Two simulation workflows are compared in this tester:

        - state_with_channel used to perform trajectory based simulation
        - state_reference used to compute exact results for each configuration with corresponding probabilities
    """
    def __init__(self, factory, config, num_trajectories = 100):
        self.factory = factory
        self.config = config
        self.dtype = factory.dtype
        self.num_trajectories = num_trajectories
        self.backend = factory.backend
        self.n = factory.num_qudits

        # parse factory to state
        self.state_with_channel = factory.to_network_state(config=config)

        if self.state_with_channel.config.__class__.__name__ == 'MPSConfig':
            self.tolerance = get_mps_tolerance(self.dtype)
        else:
            self.tolerance = {'atol': atol_mapper[self.dtype], 
                              'rtol': rtol_mapper[self.dtype]}
        
        self._parse_reference_state()
        
        self.pauli_strings = {
            'X' * self.n: 0.1,
            'Y' * self.n: 0.2,
            'Z' * self.n: 0.4,
        }
        # use an explicit NetworkOperator to activate caching for expectation computation speedup
        if 'complex' in self.dtype:
            self.pauli_operator = NetworkOperator.from_pauli_strings(self.pauli_strings, dtype=self.dtype, options=self.state_with_channel.options)
        else:
            self.pauli_operator = None
        self.num_sampling_shots = 5000
    
    def _parse_reference_state(self):
        """
        The methods perform the following:
            1. generate all configuration of potential channels in self.full_configuration
            2. create a reference NetworkState with all channels applied as a regular tensor gate (using first channel operand)
            3. compute the "conditional" probabilities for all configurations in self.conditional_probabilities

            # For example, for a system with 4 channels in total (u, g, u, g) where u denotes a unitary channel while g denotes a general channel

            # 1. Taking the third unitary channel as an example, 
            #    the keys are designed as (None, None, i) where the first two Nones means it's independent of the first two channels
            # 2. Taking the second general channel as an example, 
            #    the keys are designed as (i, j) which denotes the probability of the second general channel at index j given that the first unitary channel is at i
        """
        self.state_reference = NetworkState(self.factory.state_dims, dtype=self.dtype, config=self.config, options=self.state_with_channel.options)
        if self.factory.initial_mps_dim is not None:
            self.state_reference.set_initial_mps(self.factory.get_initial_state())

        self.conditional_probabilities = dict()        
        self.channel_info = dict()

        for op, modes, gate_info in self.factory.sequence:
            if gate_info is None and isinstance(op, (list, tuple)):
                # MPO, apply as it is
                self.state_reference.apply_mpo(modes, op)
            else:
                if gate_info is None:
                    # regular Gate, apply as it is
                    tensor_id = self.state_reference.apply_tensor_operator(modes, op)
                elif 'control_modes' in gate_info:
                    assert 'control_values' in gate_info
                    # Controlled-Tensor
                    # NetworkState currently only support immutable controlled tensors
                    tensor_id = self.state_reference.apply_tensor_operator(modes, op, control_modes=gate_info['control_modes'], control_values=gate_info['control_values'], immutable=True)
                else:
                    # Unitary/General Channels
                    assert 'probabilities' in gate_info
                    probs = gate_info['probabilities']
                    is_unitary_channel = probs is not None
                    if is_unitary_channel:
                        # For unitary channels, register the conditional probablities
                        for ix, p in enumerate(probs):
                            key = (None, ) * len(self.channel_info) + (ix, )
                            self.conditional_probabilities[key] = p
                    else:
                        # enumerate all potential configurations (including both unitary & general channels) till this general channel
                        channels_to_update = channel_infos_to_full_configuration(self.channel_info)

                        for channel in channels_to_update:
                            base_key = []
                            if channel is not None:
                                for channel_id, op_id, channel_op in channel:
                                    base_key.append(op_id)
                                    # normalize the channels to improve numerical stability
                                    probs = self.channel_info[channel_id][1]
                                    if probs is not None:
                                        # Unitary Channel
                                        self.state_reference.update_tensor_operator(channel_id, channel_op, unitary=True)
                                    else:
                                        # General Channel
                                        factor = self.conditional_probabilities.get(tuple(base_key), 1) ** .5
                                        self.state_reference.update_tensor_operator(channel_id, channel_op / factor, unitary=False)
                            rdm = self.state_reference.compute_reduced_density_matrix(modes)
                            matrix_dim = int(rdm.size**.5)
                            rdm = rdm.reshape(matrix_dim, matrix_dim)
                            p_tot = self.factory.backend.einsum('ii->', rdm).real.item()
                            for ix, general_channel in enumerate(op):
                                general_channel = general_channel.reshape(matrix_dim, matrix_dim)
                                p = self.factory.backend.einsum('ij,ik,kj->', general_channel, general_channel.conj(), rdm).real.item()
                                key = tuple(base_key) + (ix,)
                                self.conditional_probabilities[key] = p / p_tot  
                    # Now just apply the general channel as a regular tensor gate
                    tensor_id = self.state_reference.apply_tensor_operator(modes, op[0], unitary=is_unitary_channel)
                    self.channel_info[tensor_id] = (op, gate_info['probabilities'])

        self.full_configurations = channel_infos_to_full_configuration(self.channel_info)
        return

    def _compute_property_with_channel(self, property_name):
        data = []
        for _ in range(self.num_trajectories):
            if property_name == 'expectation':
                # use self.pauli_operators to activate caching for speedup
                expec, norm = self.state_with_channel.compute_expectation(self.pauli_operator, return_norm=True)
                output = expec / norm
            elif property_name == 'amplitude':
                # skip norm
                output = self.state_with_channel.compute_amplitude(self.n * '0')
            else:
                output = compute_state_basic_property(self.state_with_channel, property_name)
            data.append(output)
        return data

    def _get_probability(self, key):
        # P(key) = P(key[:-1]) * P(key[-1] | key[:-1])
        # where P(key[:-1]) means the probability of key[:-1] being active/selected
        # P(key[-1] | key[:-1]) means the conditional probability that when key[:-1] are active/selected, key[-1] is selected
        if key:
            prev_channels = key[:-1]
            active_channel = key[-1]
            if key in self.conditional_probabilities:
                # general channel
                conditional_prob = self.conditional_probabilities[key]
            else:
                # unitary channel
                unitary_key = (None, ) * len(prev_channels) + (active_channel, )
                assert unitary_key in self.conditional_probabilities
                conditional_prob = self.conditional_probabilities[unitary_key]
            return conditional_prob * self._get_probability(prev_channels)
        else:
            return 1.0
    
    def _compute_property_reference(self, property_name):
        data = []
        for entry in self.full_configurations:
            active_channels = []
            for (channel_id, op_id, operand) in entry:
                # normalize the channels
                active_channels.append(op_id)
                probs = self.channel_info[channel_id][1]
                if probs is not None:
                    # Unitary Channel
                    self.state_reference.update_tensor_operator(channel_id, operand, unitary=True)
                else:
                    # General Channel
                    factor = self.conditional_probabilities.get(tuple(active_channels), 1) ** .5
                    self.state_reference.update_tensor_operator(channel_id, operand / factor, unitary=False)
            p = self._get_probability(tuple(active_channels))
            if property_name == 'expectation':
                # use self.pauli_operators instead of a dictionary to activate caching for speedup
                expec, norm = self.state_reference.compute_expectation(self.pauli_operator, return_norm=True)
                output = expec / norm
            elif property_name == 'amplitude':
                # skip norm
                output = self.state_reference.compute_amplitude(self.n * '0')
            else:
                output = compute_state_basic_property(self.state_reference, property_name)
            data.append([p, output])
        return data

    def _verify_output(self, property_name, traj_output, reference_output, *, test_distribution=False):
        if property_name == 'sampling':
            # For sampling, since we fix the seed, we verify that each trajectory output has a high overlap with at least one of the reference output
            for snap_shot in traj_output:
                ovlp_with_reference = []
                for _, reference_sv in reference_output:
                    ovlp = compute_sample_overlap(snap_shot, reference_sv, None)
                    ovlp_with_reference.append(ovlp)
                assert max(ovlp_with_reference) >= 0.8, f"max overlap = {max(ovlp_with_reference)}"
        else:
            # making sure that all traj_output can be obtained in the reference_output
            traj_value_arrays = self.backend.asarray(traj_output)
            reference_value_array = self.backend.asarray([val for _, val in reference_output])
            diff = abs(traj_value_arrays[:, None]- reference_value_array[None,:])
            if diff.ndim > 2:
                diff = self.backend.sum(diff, axis=tuple(range(2, diff.ndim)))
                # diff shape becomes (n_trajectory, n_possible_configs)
            col_indices = self.backend.argmin(diff, axis=1)
            min_values = diff[np.arange(diff.shape[0]), col_indices]
            # make sure that each trajectory has a corresponding entry from the reference configurations
            if not self.backend.allclose(min_values, 0, **self.tolerance):
                # Calculate relative min values where possible (avoiding division by zero)
                ref_values = self.backend.abs(reference_value_array[col_indices])
                rel_min_values = min_values / (ref_values + 1e-10)
                min_val = self.backend.min(min_values)
                max_val = self.backend.max(min_values)
                mean_val = self.backend.mean(min_values)
                rel_min_val = self.backend.min(rel_min_values)
                
                error_msg = f"Trajectories simulation results on {property_name} not existing in reference.\n"
                error_msg += f"Min values summary: min={min_val}, max={max_val}, mean={mean_val}, rel_min={rel_min_val}"
                raise FatalAssertionError(error_msg)
            
            # NOTE: col_indices only indicates the closest value, it's not guaranteed to be exact matching
            # Therefore the check on min_values is critical before frequency analysis
            if test_distribution:
                trajectory_counter = Counter(col_indices.tolist())
                n_trajectories = sum(trajectory_counter.values())
                deviation = 0
                message = f"Frequency Check: {property_name=}, {n_trajectories=}\n"
                for config, count in trajectory_counter.items():
                    freq = count / n_trajectories
                    deviation += abs(freq - reference_output[config][0])
                    if property_name != 'state_vector':
                        message += f"{config=}, reference val={reference_output[config][1]} {freq=}, prob={reference_output[config][0]}\n"
                    else:
                        message += f"{config=}, {freq=}, prob={reference_output[config][0]}\n"
                message += f"Final {deviation=}"
                # make sure the general channel distribution is close to the probabilities
                if deviation > 0.1:
                    raise StatisticalAssertionError(message)
    
    def _run_property_test(self, property_name):
        if property_name == 'sampling':
            traj_output = self._compute_property_with_channel(property_name)
            reference_output = self._compute_property_reference('state_vector')
            self._verify_output(property_name, traj_output, reference_output, test_distribution=False)
            return
        reference_output = self._compute_property_reference(property_name)
        if property_name not in {'state_vector'}:
            # NOTE: for properties other than state_vector, we only test that the results from trajectory simulation
            # are all a subset of exact reference results. This is to reduce tester runtime
            traj_output = self._compute_property_with_channel(property_name)
            self._verify_output(property_name, traj_output, reference_output, test_distribution=False)
        else:
            num_trajectories = self.num_trajectories
            self.num_trajectories = 5 * num_trajectories # to be able to test distributions, we start with a trajectory count
            test_passed = False
            err = None
            for i in range(3): # max try
                try:
                    traj_output = self._compute_property_with_channel(property_name)
                    self._verify_output(property_name, traj_output, reference_output, test_distribution=True)
                    test_passed = True
                except FatalAssertionError as e:
                    # trajectory simulation results not exist in reference, no need to further increase trajectories
                    err = e
                    print(f"FATAL: {property_name} correctness test failing with {self.num_trajectories} trajectories")
                    break
                except StatisticalAssertionError as e:
                    # may be from statistical failure, increase num_trajectories for another attempt
                    err = e 
                    print(f"WARNING: {property_name} convergence test failing with {self.num_trajectories} trajectories")
                    self.num_trajectories *= 5
                if test_passed:
                    if self.num_trajectories != num_trajectories:
                        print(f"INFO: {property_name} convergence test passed with {self.num_trajectories} trajectories")
                    break
            # revert to the original num_trajectories
            self.num_trajectories = num_trajectories
            assert test_passed, err.args[0]

    def run_tests(self):
        for property_name in STATE_PROPERTIES_NAMES:
            if property_name == 'expectation' and self.pauli_operator is None:
                # currently skip expectation value tests for real type
                continue
            self._run_property_test(property_name)

        if self.state_with_channel.config.__class__.__name__ == 'MPSConfig':
            # check that if release_operators is set to True, all properties map to the same state/trajectory
            # NOTE: this test must be performed after _run_property_test as release_operators=True will capture MPS in state_with_channel and invalidate the stochastic property
            self.state_with_channel.compute_output_state(release_operators=True)
            sv = self.state_with_channel.compute_state_vector()

            where = (0, 1)
            rdm = self.state_with_channel.compute_reduced_density_matrix(where)
            assert self.backend.allclose(rdm, reduced_density_matrix_from_sv(sv, where), **self.tolerance)
            
            bitstring = '1' * self.n
            amp, norm = self.state_with_channel.compute_amplitude(bitstring, return_norm=True)
            assert self.backend.allclose(amp, amplitude_from_sv(sv, bitstring), **self.tolerance)
            assert self.backend.allclose(norm, (abs(sv)**2).sum(), **self.tolerance)

            fixed = {0: 1, 1: 0}
            batched_amp = self.state_with_channel.compute_batched_amplitudes(fixed)
            assert self.backend.allclose(batched_amp, batched_amplitude_from_sv(sv, fixed), **self.tolerance)

            if self.pauli_operator is not None:
                expectation = self.state_with_channel.compute_expectation(self.pauli_operator)
                assert self.backend.allclose(expectation, expectation_from_sv(sv, self.pauli_strings), **self.tolerance)

            nshots = 5000
            for _ in range(3):
                samples = self.state_with_channel.compute_sampling(nshots, seed=1)
                ovlp = compute_sample_overlap(samples, sv, None)
                test_passed = ovlp >= 0.95
                if test_passed:
                    print(f"INFO: sampling test passed with {nshots=}")
                    break
                else:
                    print(f"WARNING: sampling test failed with {nshots=}")
                    nshots *= 10
            assert test_passed

        self.state_with_channel.free()
        self.state_reference.free()
