"""
Example illustrating lazy conjugation using tensor qualifiers.
"""
import numpy as np

from cuquantum import contract, tensor_qualifiers_dtype

a = np.random.rand(3, 2) + 1j * np.random.rand(3, 2)
b = np.random.rand(2, 3) + 1j * np.random.rand(2, 3)

# Specify tensor qualifiers for the second tensor operand 'b'.
qualifiers = np.zeros((2,), dtype=tensor_qualifiers_dtype)
qualifiers[1]['is_conjugate'] = True

r = contract("ij,jk", a, b, qualifiers=qualifiers)
s = np.einsum("ij,jk", a, b.conj())
assert np.allclose(r, s), "Incorrect results for a * conjugate(b)"
