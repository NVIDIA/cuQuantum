# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import math
import logging
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import (
    ElementaryOperator,
    OperatorTerm,
    Operator,
    operator_action
)


@jax.jit
def main():
    """
    Run operator action.
    """
    # Redefine parameters since they are jitted.
    time = 1.1
    params = jnp.array([4.2])
    state_in = jnp.array(
        jax.random.uniform(key, (*space_mode_extents, *space_mode_extents)),
        dtype=dtype)
    jax.debug.print("Defined input state data buffer.")

    # Data buffers for elementary operators. The buffers are filled with different values to avoid JAX aliasing.
    random_consts = jax.random.uniform(key, [3])
    n_data_empty = jnp.full((space_mode_extents[0], space_mode_extents[0]), random_consts[0], dtype=dtype)
    a_data_empty = jnp.full((space_mode_extents[1], space_mode_extents[1]), random_consts[1], dtype=dtype)
    ad_data_empty = jnp.full((space_mode_extents[1], space_mode_extents[1]), random_consts[2], dtype=dtype)
    jax.debug.print("Defined elementary operator data buffers.")

    def n_callback(t, args, storage):
        storage[:] = 0.0
        module = importlib.import_module(type(storage).__module__)
        for m in range(storage.shape[0]):
            for n in range(storage.shape[1]):
                storage[m, n] = m * n * module.tan(args[0] * t)
                if storage.dtype.kind == 'c':
                    storage[m, n] += 1j * m * n / module.tan(args[0] * t)

    def n_grad_callback(t, args, tensor_grad, params_grad):
        module = importlib.import_module(type(tensor_grad).__module__)
        for m in range(tensor_grad.shape[0]):
            for n in range(tensor_grad.shape[1]):
                params_grad[0] += 2 * (
                    tensor_grad[m, n] * (m * n * t / module.cos(args[0] * t) ** 2)
                    ).real
                if tensor_grad.dtype.kind == 'c':
                    params_grad[0] += 2 * (
                        tensor_grad[m, n] * (-1j * m * n * t / module.sin(args[0] * t) ** 2)
                        ).real

    def a_callback(t, args, storage):
        storage[:] = 0.0
        dim = storage.shape[0]
        for i in range(1, dim):
            storage[i - 1, i] = math.sqrt(i)

    def ad_callback(t, args, storage):
        storage[:] = 0.0
        dim = storage.shape[0]
        for i in range(1, dim):
            storage[i, i - 1] = math.sqrt(i)

    n_wrapped_callback = cudm.WrappedTensorCallback(n_callback, cudm.CallbackDevice.GPU)
    n_wrapped_grad_callback = cudm.WrappedTensorGradientCallback(n_grad_callback, cudm.CallbackDevice.GPU)
    a_wrapped_callback = cudm.WrappedTensorCallback(a_callback, cudm.CallbackDevice.GPU)
    ad_wrapped_callback = cudm.WrappedTensorCallback(ad_callback, cudm.CallbackDevice.GPU)
    jax.debug.print("Defined callbacks for elementary operators.")

    # Create elementary operators from the data arrays.
    n_elem_op = ElementaryOperator(n_data_empty, callback=n_wrapped_callback, grad_callback=n_wrapped_grad_callback)
    ad_elem_op = ElementaryOperator(ad_data_empty, callback=ad_wrapped_callback)
    a_elem_op = ElementaryOperator(a_data_empty, callback=a_wrapped_callback)
    jax.debug.print("Created elementary operator objects.")

    # Create the Hamiltonian and dissipators.
    H = OperatorTerm(space_mode_extents)
    Ls = OperatorTerm(space_mode_extents)

    H.append([n_elem_op], modes=[0], duals=[False], coeff=1.0)
    Ls.append([ad_elem_op, a_elem_op], modes=[1, 1], duals=[False, True], coeff=1.0)
    Ls.append([a_elem_op, ad_elem_op], modes=[1, 1], duals=[False, False], coeff=-0.5)
    Ls.append([ad_elem_op, a_elem_op], modes=[1, 1], duals=[True, True], coeff=-0.5)
    jax.debug.print("Constructed operator terms from elementary operators.")

    liouvillian = Operator(space_mode_extents)
    liouvillian.append(H, dual=False, coeff=-1.0j)
    liouvillian.append(H, dual=True, coeff=1.0j)
    liouvillian.append(Ls, dual=False, coeff=1.0)
    jax.debug.print("Constructed operator from operator terms.")

    state_out, vjp_func = jax.vjp(operator_action, liouvillian, time, state_in, params)
    jax.debug.print("Computed operator action and obtained the VJP function.")

    state_out_adj = jnp.conj(state_out)
    _, _, state_in_adj, params_grad = vjp_func(state_out_adj)
    jax.debug.print("Computed gradients with respect to parameters.")

    return params_grad


if __name__ == "__main__":
    # Global parameters.
    key = jax.random.key(42)
    space_mode_extents = (3, 5)
    dtype = jnp.complex128

    main()
    print("Finished computation and exit.")
