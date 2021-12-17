"""
Example using NumPy ndarrays with interleaved format (explicit form for output indices).

The contraction result is also a NumPy ndarray.
"""
import numpy as np

from cuquantum import contract


a = np.ones((3,2))
b = np.ones((2,3))

r = contract(a, ['first', 'second'], b, ['second', 'third'], ['first', 'third'])
print(r)

