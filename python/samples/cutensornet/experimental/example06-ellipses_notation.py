# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example of Ellipses notation with contract_decompose

NumPy ndarrays are used as inputs.
"""
import numpy as np

from cuquantum import contract
from cuquantum.cutensornet.experimental import contract_decompose


a = np.ones((2,2,2,2))
b = np.ones((2,2,2,2))

# here we use contract and SVD decomposition to show usage of ellipsis
abs_cutoff = 1e-12
algorithm = {
    'qr_method' : False,
    'svd_method': {'abs_cutoff':abs_cutoff, 'partition': None} # singular values are partitioned onto A/B equally
}

################################################
### Case I. Ellipses in one input and one output
################################################

ellipsis_subscripts = 'abcd,cd...->abx,...x'
equivalent_subscripts = 'abcd,cdef->abx,efx'

a0, s0, b0 = contract_decompose(ellipsis_subscripts, a, b, algorithm=algorithm)
a1, s1, b1 = contract_decompose(equivalent_subscripts, a, b, algorithm=algorithm)

equal = np.allclose(s0, s1)
print(f"For the given operands, ``{ellipsis_subscripts}`` equal to ``{equivalent_subscripts}`` ? : {equal}")
assert equal

##################################
### Case II. Ellipses in one input
##################################

ellipsis_subscripts = 'abcd,d...->abx,cx'
equivalent_subscripts = 'abcd,defg->abx,cx'

a0, s0, b0 = contract_decompose(ellipsis_subscripts, a, b, algorithm=algorithm)
a1, s1, b1 = contract_decompose(equivalent_subscripts, a, b, algorithm=algorithm)

equal = np.allclose(s0, s1)
print(f"For the given operands, ``{ellipsis_subscripts}`` equal to ``{equivalent_subscripts}`` ? : {equal}")
assert equal

#############################################
### Case III. Ellipses in more than one input
#############################################
ellipsis_subscripts = 'ab...,bc...->ax,cx'
equivalent_subscripts = 'abef,bcef->ax,cx'

a0, s0, b0 = contract_decompose(ellipsis_subscripts, a, b, algorithm=algorithm)
a1, s1, b1 = contract_decompose(equivalent_subscripts, a, b, algorithm=algorithm)

equal = np.allclose(s0, s1)
print(f"For the given operands, ``{ellipsis_subscripts}`` equal to ``{equivalent_subscripts}`` ? : {equal}")
assert equal

###########################################################
### Case IV. Ellipses in more than one input and one output
##########################################################

ellipsis_subscripts = 'ab...,bc...->ax...,cx'
equivalent_subscripts = 'abef,bcef->axef,cx'

a0, s0, b0 = contract_decompose(ellipsis_subscripts, a, b, algorithm=algorithm)
a1, s1, b1 = contract_decompose(equivalent_subscripts, a, b, algorithm=algorithm)

equal = np.allclose(s0, s1)
print(f"For the given operands, ``{ellipsis_subscripts}`` equal to ``{equivalent_subscripts}`` ? : {equal}")
assert equal
