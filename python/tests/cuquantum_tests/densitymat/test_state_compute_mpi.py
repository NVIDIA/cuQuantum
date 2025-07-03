# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np
import pytest
from typing import Sequence
from numbers import Number

from cuquantum.densitymat import DenseMixedState, DensePureState, WorkStream

from mpi4py import MPI

# mark all tests in this file as mpi tests
pytestmark = pytest.mark.mpi

# TODO: Add tests for non-blocking execution, non-0 stream arg

# TEST_CORRECTNESS_EXPECTATION = True
TEST_FUNCTIONALITY_EXPECTATION_PURE = True
dtypes = {"float32", "float64", "complex64", "complex128"}
ATOLERANCES = dict(zip(dtypes, [np.finfo(np.dtype(_dtype)).eps * 1e2 for _dtype in dtypes]))

NUM_DEVICES = cp.cuda.runtime.getDeviceCount()
print("num_devices:", NUM_DEVICES)

def get_state(ctx, hilbert_space_dims, batch_size, package, dtype, mixed, init="random"):
    assert package == cp
    assert init == "random"
    cp.random.seed(1234)
    global_state_shape = (hilbert_space_dims * (2 if mixed else 1)) + (batch_size,)

    comm = ctx.get_communicator()
    global_state = np.zeros(global_state_shape, dtype=dtype)
    if comm.Get_rank() == 0:
        global_state += np.random.normal(0, 1, size=global_state_shape)
        if dtype in ("complex64", "complex128"):
            global_state += 1j * np.random.normal(0, 1, size=global_state_shape)
        norms = np.linalg.norm(global_state.reshape(-1, batch_size), axis=0)
        global_state /= np.linalg.norm(global_state.reshape(-1, batch_size), axis=0)
    comm.Bcast(global_state, root=0)
    with cp.cuda.Device(ctx.device_id):
        State = DensePureState if not mixed else DenseMixedState
        state = State(ctx, hilbert_space_dims, batch_size, dtype)
        size = state.storage_size
        state.attach_storage(cp.empty(size, dtype=dtype))
        state.view()[:] = cp.nan

        # print(state.local_info, hilbert_space_dims, batch_size)
        local_shape, offsets = state.local_info
        local_state = global_state.copy()
        for ind in range(len(local_shape)):
            local_state = np.take(
                local_state, np.arange(offsets[ind], offsets[ind] + local_shape[ind]), axis=ind
            )
        gpu_local_state = cp.asarray(local_state, order="F")
        state.view()[:] = gpu_local_state
    return state, global_state


class TestStateAPI:

    def setup_method(self):
        self.device_id = MPI.COMM_WORLD.Get_rank() % NUM_DEVICES
        self.ctx = WorkStream(device_id = self.device_id)
        # self.ctx._comm = MPI.COMM_WORLD.Dup()
        self.ctx.set_communicator(comm=MPI.COMM_WORLD, provider="MPI")
        cp.cuda.Device(self.device_id).use()

    def teardown_method(self):
        self.ctx=None

    @pytest.mark.parametrize("hilbert_space", [(10,), (4, 6), (2, 3), (4,), (7,), (3, 3, 3)])
    @pytest.mark.parametrize("package", [cp])
    @pytest.mark.parametrize(
        "dtype,batch_size,factors",
        [
            ("float32", 1, 2.0),
            ("float64", 1, 2.0),
            ("complex64", 1, 2.0),
            ("complex128", 1, 2.0),
            ("complex128", 1, 2.0 + 0.5j),
            pytest.param(
                "float64",
                2,
                2.0,
            ),
            pytest.param(
                "float64",
                2,
                (2.0, 3.0),
            ),
            pytest.param(
                "float64",
                2,
                np.array([2.0, 3.0]),
            ),
            pytest.param(
                "float64",
                2,
                cp.array([2.0, 3.0]),
            ),
            pytest.param(
                "complex128",
                2,
                cp.array([2.0, 3.0], dtype="complex128"),
            ),
            pytest.param("float32", 2, cp.array([np.sqrt(2.0), np.sqrt(3.0)])),
            pytest.param("float64", 2, (2.0, 3.0, 4.0), marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param(
                "float64",
                2,
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                marks=pytest.mark.xfail(raises=ValueError),
            ),
            pytest.param("float64", 2, {1.0, 2.0}, marks=pytest.mark.xfail(raises=TypeError)),
            pytest.param("float64", 2, (1, 2 + 3j), marks=pytest.mark.xfail(raises=TypeError)),
            pytest.param(
                "float64",
                2,
                cp.array([1 + 2j, 3 + 4j]),
                marks=pytest.mark.xfail(raises=TypeError),
            ),
        ],
    )
    @pytest.mark.parametrize("purity", ["PURE", "MIXED"])
    def test_state_inplace_scale(self, hilbert_space, package, dtype, batch_size, factors, purity):
        psi, global_state = get_state(
            self.ctx,
            hilbert_space,
            batch_size,
            package,
            dtype,
            init="random",
            mixed=(purity == "MIXED"),
        )
        shape, offsets = psi.local_info
        psi_arr = psi.view().get()
        assert not np.any(np.isnan(psi_arr))
        psi.inplace_scale(factors)
        scaled_psi_arr = psi.view().get()

        assert not np.any(np.isnan(scaled_psi_arr))

        if isinstance(factors, Sequence):
            factors_np = np.array(factors, dtype=psi.dtype)
        elif isinstance(factors, cp.ndarray):
            factors_np = factors.get()
        elif isinstance(factors, Number):  # single number or numpy array
            factors_np = np.ones(batch_size) * factors
        else:
            factors_np = factors
        ref = np.einsum(
            "...i,i->...i",
            psi_arr,
            factors_np[np.array(range(offsets[-1], offsets[-1] + shape[-1]))],
        )
        np.testing.assert_allclose(scaled_psi_arr, ref, rtol=1e-4, atol=ATOLERANCES[dtype])

    @pytest.mark.parametrize("hilbert_space", [(10,), (4, 6), (3, 7)])
    @pytest.mark.parametrize("package", [cp])
    @pytest.mark.parametrize(
        "dtype,batch_size,factors",
        [
            # ("float32", 1, 2.0),
            ("float64", 1, 2.0),
            ("complex64", 1, 2.0 + 3.0j),
            ("complex128", 1, 2.0 + 3.0j),
            (
                "float64",
                2,
                2.0,
            ),
            (
                "float64",
                2,
                (2.0, 3.0),
            ),
            (
                "float64",
                2,
                np.array([2.0, 3.0]),
            ),
            (
                "float64",
                2,
                cp.array([2.0, 3.0]),
            ),
            (
                "float32",
                2,
                cp.array([np.sqrt(2.0), np.sqrt(3.0)]),
            ),
            pytest.param("float64", 2, (2.0, 3.0, 4.0), marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param(
                "float64",
                2,
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                marks=pytest.mark.xfail(raises=ValueError),
            ),
            pytest.param("float64", 2, {1.0, 2.0}, marks=pytest.mark.xfail(raises=TypeError)),
            pytest.param("float64", 2, (1, 2 + 3j), marks=pytest.mark.xfail(raises=TypeError)),
            # ("float32", 2, np.array([1.0, 2.0])),
            pytest.param(
                "float64",
                2,
                cp.array([1 + 2j, 3 + 4j]),
                marks=pytest.mark.xfail(raises=TypeError),
            ),
        ],
    )
    @pytest.mark.parametrize("purity", ["PURE", "MIXED"])
    def test_state_inplace_accumulate(
        self, hilbert_space, package, dtype, batch_size, factors, purity
    ):
        psi1, _ = get_state(
            self.ctx,
            hilbert_space,
            batch_size,
            package,
            dtype,
            init="random",
            mixed=(purity == "MIXED"),
        )
        shape, offsets = psi1.local_info
        psi2, _ = get_state(
            self.ctx,
            hilbert_space,
            batch_size,
            package,
            dtype,
            init="random",
            mixed=(purity == "MIXED"),
        )
        psi1_arr = psi1.view().get()
        psi2_arr = psi2.view().get()
        assert not np.any(np.isnan(psi1_arr))
        assert not np.any(np.isnan(psi2_arr))
        psi1.inplace_accumulate(psi2, factors)
        accumulated_psi_arr = psi1.view().get()

        if isinstance(factors, Sequence):
            factors_np = np.array(factors, dtype=psi1.dtype)
        elif isinstance(factors, cp.ndarray):
            factors_np = factors.get()
        elif isinstance(factors, Number):  # single number or numpy array
            factors_np = np.ones(batch_size) * factors
        else:
            factors_np = factors
        ref = (
            np.einsum(
                "...i,i->...i",
                psi2_arr,
                factors_np[np.array(range(offsets[-1], offsets[-1] + shape[-1]))],
            )
            + psi1_arr
        )
        # np.testing.assert_allclose(np.unique(accumulated_psi_arr), np.unique(ref), rtol=1e-5, atol=1e-8)
        # print(accumulated_psi_arr / ref)
        np.testing.assert_allclose(accumulated_psi_arr, ref, rtol=1e-4, atol=ATOLERANCES[dtype])
        

    @pytest.mark.parametrize("hilbert_space", ((10,), (10, 2, 4), (3, 3, 3)))
    @pytest.mark.parametrize("package", (cp,))
    @pytest.mark.parametrize("dtype", ("float32", "float64", "complex64", "complex128"))
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("purity", ["PURE", "MIXED"])
    def test_state_inner_product(self, hilbert_space, package, dtype, batch_size, purity):
        psi1, _ = get_state(
            self.ctx,
            hilbert_space,
            batch_size,
            package,
            dtype,
            init="random",
            mixed=(purity == "MIXED"),
        )
        psi2, _ = get_state(
            self.ctx,
            hilbert_space,
            batch_size,
            package,
            dtype,
            init="random",
            mixed=(purity == "MIXED"),
        )
        psi1_arr = psi1.view().get()
        psi2_arr = psi2.view().get()
        slice_shape, offsets = psi1.local_info

        inner_prod = psi1.inner_product(psi2)
        inner_prod_arr = inner_prod.get()

        psi1_arr = psi1_arr.reshape((-1, psi1_arr.shape[-1]), order="F")
        psi2_arr = psi2_arr.reshape((-1, psi2_arr.shape[-1]), order="F")
        ref = np.zeros((batch_size,), dtype=inner_prod.dtype)
        reduced_ref = np.zeros((batch_size,), dtype=inner_prod.dtype)
        local_batch_size = psi1.view().shape[-1]
        for i in range(local_batch_size):
            ref[offsets[-1] + i] = np.vdot(psi1_arr[:, i], psi2_arr[:, i])
        comm = self.ctx.get_communicator()
        comm.Allreduce(ref, reduced_ref)
        np.testing.assert_allclose(inner_prod_arr, reduced_ref, rtol=1e-4, atol=ATOLERANCES[dtype])

    @pytest.mark.parametrize(
        "hilbert_space",
        (
            #(10,),
            (10, 2, 4),
            (3, 3, 3),
        ),
    )
    @pytest.mark.parametrize("package", (cp,))
    @pytest.mark.parametrize("dtype", ("float32", "float64", "complex64", "complex128"))
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("purity", ("PURE", "MIXED"))
    def test_state_norm(self, hilbert_space, package, dtype, batch_size, purity):
        psi, global_state = get_state(
            self.ctx,
            hilbert_space,
            batch_size,
            package,
            dtype,
            mixed=(purity == "MIXED"),
            init="random",
        )
        psi_arr = psi.view().get()
        norm = psi.norm().get()
        shape, offsets = psi.local_info
        # psi_arr = psi_arr.reshape((-1, psi_arr.shape[-1]), order="F")
        ref = np.zeros((batch_size,), dtype=psi.storage.real.dtype)
        reduced_ref = np.zeros((batch_size,), dtype=norm.dtype)
        local_batch_size = psi.view().shape[-1]
        for i in range(local_batch_size):
            ref[offsets[-1] + i] = np.vdot(psi_arr[..., i], psi_arr[..., i]).real
        global_ref = np.zeros((batch_size,), dtype=norm.dtype)
        for i in range(batch_size):
            global_ref[i] = np.vdot(global_state[..., i], global_state[..., i]).real
        global_state = None
        comm = self.ctx.get_communicator()
        comm.Allreduce(ref, reduced_ref)
        print(global_ref, ref, reduced_ref, norm)

        np.testing.assert_allclose(norm, global_ref, rtol=1e-4, atol=ATOLERANCES[dtype])
        
    @pytest.mark.parametrize(
        "hilbert_space",
        (   #(10,), #mark as expected fail in anticipation of rotation
            (10, 2, 4),
            (3, 3, 3),
        ),
    )
    @pytest.mark.parametrize("package", (cp,))
    @pytest.mark.parametrize("dtype", ("float32","float64", "complex64", "complex128"))
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("purity", ("PURE","MIXED"))
    def test_state_trace(self, hilbert_space, package, dtype, batch_size, purity):
        #hilbert_space = tuple(hilbert_space[i] + int(purity == "MIXED") for i in range(len(hilbert_space)))
        psi, global_state = get_state(
            self.ctx,
            hilbert_space,
            batch_size,
            package,
            dtype,
            mixed=(purity == "MIXED"),
            init="random",
        )
        psi_arr = psi.view().get()
        gpu_trace = psi.trace()
        cp.cuda.Device().synchronize()
        cpu_trace = gpu_trace.get()
        trace = cpu_trace
        shape, offsets = psi.local_info
        # psi_arr = psi_arr.reshape((-1, psi_arr.shape[-1]), order="F")
        local_batch_size = psi.view().shape[-1]
        global_ref = np.zeros((batch_size,), dtype=trace.dtype)
        matdim=np.prod(hilbert_space)
        if purity=="MIXED":
            for i in range(batch_size):
                global_ref[i] = np.trace(global_state[..., i].reshape(matdim,matdim))
        else:
            for i in range(batch_size):
                global_ref[i] = np.vdot(global_state[..., i], global_state[..., i]).real
        global_state = None
        if self.ctx.get_communicator().rank == 0:
            print(trace.dtype, trace, global_ref)
        np.testing.assert_allclose(trace, global_ref, rtol=1e-4, atol=ATOLERANCES[dtype])