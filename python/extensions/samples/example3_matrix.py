# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator action example using full matrix operators.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from cuquantum.densitymat.jax import (
    MatrixOperator,
    OperatorTerm,
    Operator,
    operator_action
)

# Toggle logging from the cuQuantum Python JAX API.
ENABLE_LOGGING = False

if ENABLE_LOGGING:
    import logging
    logging.basicConfig(
        level=logging.INFO,  # logging level can be modified as well
        format='%(name)s [%(levelname)s] %(message)s'
    )


def print_device_info():
    """
    Print the information of the current device.
    """
    from cuda.bindings import runtime as cudart

    err, dev_id = cudart.cudaGetDevice()
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDevice failed with error code {err}")

    err, props = cudart.cudaGetDeviceProperties(dev_id)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDeviceProperties failed with error code {err}")

    print("===== device info ======")
    print("GPU-local-id:", dev_id)
    print("GPU-name:", props.name.decode())
    print("GPU-nSM:", props.multiProcessorCount)
    print("GPU-major:", props.major)
    print("GPU-minor:", props.minor)
    print("========================")


@jax.jit
def main():
    """
    Main computation.
    """
    state_in = jnp.asarray(jax.random.uniform(key, (*space_mode_extents, *space_mode_extents)), dtype=dtype)
    jax.debug.print("Defined input state data buffer.", ordered=True)

    # Single-mode operator data arrays.
    dim0, dim1 = space_mode_extents
    n_data = jnp.asarray(jnp.diag(jnp.arange(dim0)), dtype=dtype)
    a_data = jnp.asarray(jnp.diag(jnp.sqrt(jnp.arange(1, dim1)), k=1), dtype=dtype)
    ad_data = jnp.asarray(jnp.diag(jnp.sqrt(jnp.arange(1, dim1)), k=-1), dtype=dtype)
    jax.debug.print("Defined single-mode operator data buffers.", ordered=True)

    # Embed single-mode operators into the full Hilbert space via Kronecker product.
    # n acts on mode 0: N ⊗ I
    # a, a† act on mode 1: I ⊗ a, I ⊗ a†
    n_full = jnp.kron(n_data, jnp.eye(dim1, dtype=dtype)).reshape(*space_mode_extents, *space_mode_extents)
    a_full = jnp.kron(jnp.eye(dim0, dtype=dtype), a_data).reshape(*space_mode_extents, *space_mode_extents)
    ad_full = jnp.kron(jnp.eye(dim0, dtype=dtype), ad_data).reshape(*space_mode_extents, *space_mode_extents)
    jax.debug.print("Embedded operators into full Hilbert space.", ordered=True)

    n_mat_op = MatrixOperator(n_full)
    a_mat_op = MatrixOperator(a_full)
    ad_mat_op = MatrixOperator(ad_full)
    jax.debug.print("Created matrix operator objects.", ordered=True)

    # Create the Hamiltonian and dissipators.
    H = OperatorTerm(space_mode_extents)
    Ls = OperatorTerm(space_mode_extents)

    H.append([n_mat_op], duals=[False], coeff=1.0)
    Ls.append([ad_mat_op, a_mat_op], duals=[False, True], coeff=1.0)
    Ls.append([a_mat_op, ad_mat_op], duals=[False, False], coeff=-0.5)
    Ls.append([ad_mat_op, a_mat_op], duals=[True, True], coeff=-0.5)
    jax.debug.print("Constructed operator terms from matrix operators.", ordered=True)

    liouvillian = Operator(space_mode_extents)
    liouvillian.append(H, dual=False, coeff=-1.0j)
    liouvillian.append(H, dual=True, coeff=1.0j)
    liouvillian.append(Ls, dual=False, coeff=1.0)
    jax.debug.print("Constructed operator from operator terms.", ordered=True)

    state_out = operator_action(liouvillian, state_in)
    jax.debug.print("Performed operator action on the input state.", ordered=True)

    return state_out


if __name__ == "__main__":

    print_device_info()

    key = jax.random.key(42)
    space_mode_extents = (3, 5)
    dtype = jnp.complex128

    main()

    print("Finished computation and exit.")
