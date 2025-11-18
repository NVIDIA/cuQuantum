# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import logging

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from cuquantum.densitymat.jax import (
    ElementaryOperator,
    OperatorTerm,
    Operator,
    operator_action
)


@jax.jit
def main():
    time = 0.0
    state_in = jnp.asarray(
        jax.random.uniform(key, (*space_mode_extents, *space_mode_extents)),
        dtype=dtype)
    jax.debug.print("Defined input state data buffer.")

    n_data_dia = jnp.asarray(
        jnp.expand_dims(jnp.arange(space_mode_extents[0]), axis=1),
        dtype=dtype)
    a_data_dia = jnp.asarray(
        jnp.expand_dims(jnp.sqrt(jnp.arange(1, space_mode_extents[1] + 1)), axis=1),
        dtype=dtype)
    ad_data_dia = jnp.asarray(
        jnp.expand_dims(jnp.sqrt(jnp.arange(1, space_mode_extents[1] + 1)), axis=1),
        dtype=dtype)
    jax.debug.print("Defined elementary operator data buffers.")

    # Create elementary operators from the data arrays.
    n_elem_op = ElementaryOperator(n_data_dia, offsets=[0])
    a_elem_op = ElementaryOperator(a_data_dia, offsets=[1])
    ad_elem_op = ElementaryOperator(ad_data_dia, offsets=[-1])
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

    state_out = operator_action(liouvillian, time, state_in)
    jax.debug.print("Performed operator action on the input state.")

    return state_out


if __name__ == "__main__":
    # Global parameters.
    key = jax.random.key(42)
    space_mode_extents = (3, 5)
    dtype = jnp.complex128

    main()
    print(f"Finished computation and exit.")
