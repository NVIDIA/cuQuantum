"""
Example using NumPy ndarrays. Provide contraction path.

The contraction result is also a NumPy ndarray.
"""
import numpy as np

from cuquantum import contract, OptimizerOptions


a = np.ones((3,2))
b = np.ones((2,3))
c = np.ones((3,3))

o = OptimizerOptions(path=[(0,2), (0,1)])
r = contract("ij,jk,kl->il", a, b, c, optimize=o)
print(r)

