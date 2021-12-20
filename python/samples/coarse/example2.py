"""
Example using NumPy ndarrays with implicit Einstein summation.

The contraction result is also a NumPy ndarray.
"""
import numpy as np

from cuquantum import contract


a = np.ones((3,2))
b = np.ones((2,3))

r = contract("ij,jk", a, b)
print(r)

