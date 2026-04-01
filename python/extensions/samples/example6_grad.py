# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator action gradient example.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import (
    ElementaryOperator,
    OperatorTerm,
    Operator,
    operator_action,
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


def coherent_state(n_levels, alpha):
    """
    Create a coherent state |alpha⟩ via the displacement operator.
    """
    # Create annihilation operator a
    a = jnp.diag(jnp.sqrt(jnp.arange(1, n_levels, dtype=jnp.complex128)), k=1)
    
    # Create creation operator a† (a_dag)
    a_dag = a.conj().T
    
    # Compute displacement operator D(α) = exp(α a† - α* a)
    displacement_arg = alpha * a_dag - jnp.conj(alpha) * a
    D = jax.scipy.linalg.expm(displacement_arg)
    
    # Create ground state |0⟩
    ground_state = jnp.zeros(n_levels, dtype=jnp.complex128)
    ground_state = ground_state.at[0].set(1.0)
    
    # Apply displacement operator: |α⟩ = D(α)|0⟩
    coherent = D @ ground_state
    
    return coherent


def main(omega, kappa, alpha0):
    """
    Compute oscillator population using cuQuantum Python JAX.
    """
    # initialize operators, initial state and saving times
    h_key = jax.random.key(41)
    h_data = jax.random.normal(h_key, (dims[0], dims[0]), dtype=jnp.complex128)
    h_data = omega * (h_data + h_data.conj().T) / 2
    jax.debug.print("Defined Hamiltonian elementary operator data buffer.", ordered=True)

    h = ElementaryOperator(h_data)
    jax.debug.print("Created Hamiltonian elementary operator.", ordered=True)

    # Construct operator term for the Hamiltonian
    H = OperatorTerm(dims)
    H.append([h], modes=modes)
    jax.debug.print("Constructed Hamiltonian operator term.", ordered=True)

    l_key = jax.random.key(42)
    l_data = jnp.sqrt(kappa) * jax.random.normal(l_key, (dims[0], dims[0]), dtype=jnp.complex128)
    jax.debug.print("Defined dissipation elementary operator data buffers.", ordered=True)

    # extract elementary operators
    l = ElementaryOperator(l_data)
    ld = ElementaryOperator(l_data.conj().T)
    jax.debug.print("Created dissipation elementary operators.", ordered=True)

    Ls = OperatorTerm(dims)
    Ls.append([l, ld], modes=(0, 0), duals=(False, True), coeff=1.0)
    Ls.append([l, ld], modes=(0, 0), duals=(False, False), coeff=-0.5)
    Ls.append([ld, l], modes=(0, 0), duals=(True, True), coeff=-0.5)
    jax.debug.print("Constructed dissipator operator term.", ordered=True)

    psi0 = coherent_state(dims[0], alpha0)
    rho0 = jnp.outer(psi0, psi0.conj())
    jax.debug.print("Created initial state data buffer.", ordered=True)
    
    liouvillian = Operator(dims)
    liouvillian.append(H, dual=False, coeff=-1.0j)
    liouvillian.append(H, dual=True, coeff=1.0j)
    liouvillian.append(Ls, dual=False, coeff=1.0)
    jax.debug.print("Constructed Liouvillian operator from operator terms.", ordered=True)

    rho1 = operator_action(liouvillian, rho0)
    jax.debug.print("Performed operator action on the input state.", ordered=True)

    number_op = jnp.diag(jnp.arange(dims[0], dtype=jnp.complex128))
    return (number_op @ rho1).trace().real


if __name__ == "__main__":

    print_device_info()

    # parameters
    dims = (5,)     # Hilbert space dimension
    modes = (0,)
    omega = 1.0     # frequency
    kappa = 0.1     # decay rate
    alpha0 = 1.0    # initial coherent state amplitude

    # Compute gradient with respect to omega, kappa and alpha
    result = jax.grad(main, argnums=(0, 1, 2))(omega, kappa, alpha0)
    print("Finished computation and exit.")
