"""
Example illustrating mode broadcasting.
"""
import numpy as np

from cuquantum import contract


a = np.random.rand(3,1)
b = np.random.rand(3,3)

expr = "ij,jk"

r = contract(expr, a, b)
s = np.einsum(expr, a, b)
assert np.allclose(r, s), "Incorrect results."
