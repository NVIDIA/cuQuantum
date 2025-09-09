# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import opt_einsum as oe

from cuquantum.tensornet import CircuitToEinsum
from cuquantum.tensornet._internal.decomposition_utils import compute_mid_extent

from ...utils.approxTN_utils import gate_decompose, tensor_decompose, SVD_TOLERANCE, verify_unitary
from ...utils.helpers import get_or_create_tensor_backend, get_dtype_name, get_contraction_tolerance


# valid simulation setting for reference MPS class
MPS_VALID_CONFIGS = {'max_extent', 'abs_cutoff', 'rel_cutoff', 'discarded_weight_cutoff', 'normalization', 'canonical_center', 'mpo_application', 'gauge_option'}

def trim_mps_config(mps_config):
    trimmed_config = mps_config.copy()
    for key in ("algorithm", "gesvdj_max_sweeps"):
        trimmed_config.pop(key, None)
    return trimmed_config

def is_converter_mps_compatible(converter):
    for _, qubits in converter.gates:
        if len(qubits) > 2:
            return False
    return True

def get_mps_tolerance(dtype):
    tolerance = get_contraction_tolerance(dtype)
    # relax the tolerance for SVD based simulations
    if dtype in ('float64', 'complex128'):
    # for double precision, relax the tolerance
        tolerance['atol'] += SVD_TOLERANCE[dtype] ** .5
        tolerance['rtol'] += SVD_TOLERANCE[dtype] ** .5
    else:
        tolerance['atol'] += SVD_TOLERANCE[dtype]
        tolerance['rtol'] += SVD_TOLERANCE[dtype]
    return tolerance

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

class MPS:

    def __init__(
        self, 
        qudits,
        backend,
        *,
        qudit_dims=2,
        dtype='complex128',
        mps_tensors=None,
        gauge_option='free',
        mpo_application='approximate',
        canonical_center=None,
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
    
    def compute_state_vector(self):
        if self.sv is None:
            self.absorb_gauges(self.mps_tensors)
            inputs = []
            output_modes = []
            for i, o in enumerate(self.mps_tensors):
                modes = [2*i, 2*i+1, 2*i+2]
                inputs.extend([o, modes])
                output_modes.append(2*i+1)
            inputs.append(output_modes)
            self.sv =  oe.contract(*inputs)
        return self.sv
    
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
