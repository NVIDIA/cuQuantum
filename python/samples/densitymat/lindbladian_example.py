from typing import Callable, Union, Sequence, Tuple
from numbers import Number

import numpy as np
import cupy as cp


from cuquantum.densitymat import Operator, OperatorTerm, GPUCallback, CPUCallback, DenseMixedState, WorkStream

CallbackType = Union[GPUCallback, CPUCallback]
NDArrayType = Union[np.ndarray, cp.ndarray]
CoefficientType = Union[Number, NDArrayType, CallbackType, Tuple[NDArrayType, CallbackType]]

###
# This example shows how to define a lindbladian master equation using methods of OperatorTerm and Operator in the function `get_lindbladian` below.
# Other examples, e.g. dense_operator_example.py and multidiagonal_operator_example.py, show an alternative way to define the same type of master equation. This is example is based on multidiagonal_operator_example.py for simplicity.
# The methods used in `get_lindbladian` are convenience functionality.
# Applying liouvillian operators as shown here may be slower than applying the same liouvillian defined directly from instances of `DenseOperator` or `MultidiagonalOperator` as in the other examples.
###


def get_lindbladian(
    hamiltonian: OperatorTerm | Operator,
    jump_operators: Sequence[OperatorTerm],
    dissipation_strengths: CoefficientType | tuple[CoefficientType],
) -> OperatorTerm | Operator:
    """
    Assembles the operator corresponding to the Lindblad master equation given a Hamiltonian specifying the closed system evolution and a sequence of jump operators and their respective dissipation strength.

    Parameters:
    hamiltonian: Union[OperatorTerm, Operator]
        The Hamiltonian specifying the closed system evolution.
    jump_operators: Sequence[OperatorTerm]
        Jump operators specifying the coupling to the environment.
        Note that each element, i.e. jump operator, must act only on the ket-modes of the quantum state (which corresponds to the default duality argument in OperatorTerm creation.).
    dissipation_strengths: CoefficientType | tuple[CoefficientType]
        Coefficients of system environment coupling.
        If not specified as sequence, the same value is used for all jump operators.

    Returns:
    Union[OperatorTerm, Operator]
        The Lindbladian, returned as the same type as input `hamiltonian`.
    """
    if isinstance(hamiltonian, (OperatorTerm, Operator)):
        liouvillian = -1j * (hamiltonian + (hamiltonian.dual() * (-1)))
    else:
        raise TypeError(
            f"Expected input argument `hamiltonian` of type OperatorTerm or Operator, received type {type(hamiltonian)}."
        )
    dissipative_part = None
    for i, jump_op in enumerate(jump_operators):
        if not isinstance(jump_op, OperatorTerm):
            raise TypeError(
                f"Expected elements of input argument `jump_operators` of type OperatorTerm, received type {type(jump_op)}."
            )
        if not all(all(set(_dual) == {False} for _dual in dual) for dual in jump_op.duals):
            raise ValueError("Jump operators that act on bra modes are not supported in `get_lindbladian`.")
        squared_term = jump_op * jump_op.dag()
        two_sided_term = jump_op * jump_op.dual().dag()
        dissipation_strength = dissipation_strengths[i] if isinstance(dissipation_strengths, tuple) else dissipation_strengths
        if not dissipative_part:
            dissipative_part = -1 / 2 * (squared_term * dissipation_strength)
        else:
            dissipative_part += -1 / 2 * (squared_term * dissipation_strength)
        dissipative_part += -1 / 2 * (squared_term.dual() * (dissipation_strength))
        dissipative_part += two_sided_term * dissipation_strength
    liouvillian += dissipative_part
    return liouvillian


import cupy as cp
import numpy as np
from cuquantum.densitymat import (
    tensor_product,
    full_matrix_product,
    MultidiagonalOperator,
    DenseMixedState,
    WorkStream,
    Operator,
    GPUCallback,
)

dev = cp.cuda.Device()  # get current device
props = cp.cuda.runtime.getDeviceProperties(dev.id)
print("===== device info ======")
print("GPU-local-id:", dev.id)
print("GPU-name:", props["name"].decode())
print("GPU-clock:", props["clockRate"])
print("GPU-memoryClock:", props["memoryClockRate"])
print("GPU-nSM:", props["multiProcessorCount"])
print("GPU-major:", props["major"])
print("GPU-minor:", props["minor"])

# Parameters
dtype = "complex128"
hilbert_space_dims = (4, 3, 5, 7)
h_offsets = [
    [-1, 0, 1],
    [0],
    [-1, 0, 1],
    [-1],
]
l_offsets = [[-1, 0, 1], [1], [-2, -1, 0, 1, 2], [1, 2]]
num_modes = len(hilbert_space_dims)
modes_string = "abcdefghijkl"[: len(hilbert_space_dims)]
batch_size = 1


def take_complex_conjugate_transpose(arr):
    return arr.transpose(tuple(range(num_modes, 2 * num_modes)) + tuple(range(0, num_modes)) + (2 * num_modes,)).conj()


#############
# Hamiltonian
#############

h0_arr = cp.empty((hilbert_space_dims[0], len(h_offsets[0]), batch_size), dtype=dtype)
h0_arr[:] = cp.random.rand(*h0_arr.shape)
h1_arr = cp.empty((hilbert_space_dims[1], len(h_offsets[1]), batch_size), dtype=dtype)
h1_arr[:] = cp.random.rand(*h1_arr.shape)
h2_arr = cp.empty((hilbert_space_dims[2], len(h_offsets[2]), batch_size), dtype=dtype)
h2_arr[:] = cp.random.rand(*h2_arr.shape)
h3_arr = cp.empty((hilbert_space_dims[3], len(h_offsets[3]), batch_size), dtype=dtype)
h3_arr[:] = cp.random.rand(*h3_arr.shape)


def callback1(t, args, arr):  # inplace callback
    assert len(args) == 1, "Length of args should be 1."
    omega = args[0]
    for i in range(hilbert_space_dims[0]):
        for j in range(len(h_offsets[0])):
            for b in range(batch_size):
                arr[i, j, b] = (i + j) * np.sin(omega * t)


def callback2(t, args):  # out-of-place callback
    assert len(args) == 1, "Length of args should be 1."
    omega = args[0]
    arr = cp.empty((hilbert_space_dims[1], len(h_offsets[1]), batch_size), dtype=dtype)
    for i in range(hilbert_space_dims[1]):
        for j in range(len(h_offsets[1])):
            for b in range(batch_size):
                arr[i, j, b] = (i + j) * np.cos(omega * t)
    return arr


h0_callback = GPUCallback(callback1, is_inplace=True)
h1_callback = GPUCallback(callback2, is_inplace=False)

h0_op = MultidiagonalOperator(h0_arr, h_offsets[0], h0_callback)
h1_op = MultidiagonalOperator(h1_arr, h_offsets[1], h1_callback)
h2_op = MultidiagonalOperator(h2_arr, h_offsets[2])
h3_op = MultidiagonalOperator(h3_arr, h_offsets[3])

H = (
    tensor_product((h0_op, [0]), (h1_op, [1]))
    + tensor_product((h1_op, [1]), (h2_op, [2]))
    + tensor_product((h2_op, [2]), (h3_op, [3]))
)

print("Created an OperatorTerm for the Hamiltonian.")

#############
# Dissipators
#############

l0_arr = cp.empty((hilbert_space_dims[0], len(l_offsets[0]), batch_size), dtype=dtype)
l0_arr[:] = cp.random.rand(*l0_arr.shape)
l1_arr = cp.empty((hilbert_space_dims[1], len(l_offsets[1]), batch_size), dtype=dtype)
l1_arr[:] = cp.random.rand(*l1_arr.shape)
l2_arr = cp.empty((hilbert_space_dims[2], len(l_offsets[2]), batch_size), dtype=dtype)
l2_arr[:] = cp.random.rand(*l2_arr.shape)
l3_arr = cp.empty((hilbert_space_dims[3], len(l_offsets[3]), batch_size), dtype=dtype)
l3_arr[:] = cp.random.rand(*l3_arr.shape)

l0_op = MultidiagonalOperator(l0_arr, l_offsets[0])
l1_op = MultidiagonalOperator(l1_arr, l_offsets[1])
l2_op = MultidiagonalOperator(l2_arr, l_offsets[2])
l3_op = MultidiagonalOperator(l3_arr, l_offsets[3])

Ls = []
for i, l in enumerate([l0_op, l1_op, l2_op, l3_op]):
    Ls.append(
        tensor_product(
            (l, [i], [False]),
        )
    )

print("Created OperatorTerms for the Liouvillian.")

#############
# Liouvillian
#############
dissipation_strengths = tuple(np.random.rand(4))
liouvillian = Operator(hilbert_space_dims, (get_lindbladian(H, Ls, dissipation_strengths),))
print("Created the Liouvillian operator.")

################
# Density matrix
################
ctx = WorkStream()

rho = DenseMixedState(ctx, hilbert_space_dims, batch_size, dtype)
rho.attach_storage(cp.empty(rho.storage_size, dtype=dtype))
rho_arr = rho.view()
rho_arr[:] = cp.random.normal(size=rho_arr.shape)
if "complex" in dtype:
    rho_arr[:] += 1j * cp.random.normal(size=rho_arr.shape)
rho_arr += take_complex_conjugate_transpose(rho_arr)
rho_arr /= rho.trace()
print("Created a Haar random normalized mixed quantum state.")

rho_out = rho.clone(cp.zeros_like(rho.storage, order="F"))
print("Created zero initialized output state.")

#################
# Operator action
#################

liouvillian.prepare_action(ctx, rho)
liouvillian.compute_action(1.0, [3.5], rho, rho_out)

print("Finished computation and exit.")
