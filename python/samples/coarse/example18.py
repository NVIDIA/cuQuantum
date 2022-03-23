"""
Example illustrating ellipsis broadcasting.
"""
import numpy as np

from cuquantum import contract


a = np.random.rand(3,1)
b = np.random.rand(3,3)

# Elementwise product of two matrices.
expr = "...,..."

r = contract(expr, a, b)
s = np.einsum(expr, a, b)
assert np.allclose(r, s), "Incorrect results."
