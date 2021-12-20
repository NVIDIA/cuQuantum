"""
Example using NumPy ndarrays with explicit Einstein summation (Unicode characters).

The contraction result is also a NumPy ndarray.
"""
import numpy as np

from cuquantum import contract


a = np.ones((3,2))
b = np.ones((2,3))

r = contract("αβ,βγ->αγ", a, b)
print(r)

