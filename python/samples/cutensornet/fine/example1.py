# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using operations on the Network object with torch tensors. This can be used to
amortize the cost of finding the best contraction path and autotuning the network across
multiple contractions.

The contraction result is also a torch tensor on the same device as the operands. 
"""
import torch

from cuquantum import Network


# The parameters of the tensor network.
expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]

device = 'cuda'
# Create torch tensors.
operands = [torch.rand(*shape, dtype=torch.float64, device=device) for shape in shapes]

# Create the network.
with Network(expr, *operands) as n:

    # Find the contraction path.
    path, info = n.contract_path({'samples': 500})

    # Autotune the network.
    n.autotune(iterations=5)

    # Perform the contraction.
    r1 = n.contract()
    print("Contract the network (r1):")
    print(r1)

    # Create new operands. 
    operands = [i*operand for i, operand in enumerate(operands, start=1)]

    # Reset the network operands.
    n.reset_operands(*operands)

    # Perform the contraction with the new operands.
    print("Reset the operands and perform the contraction (r2):")
    r2 = n.contract()
    print(r2)

    from math import factorial
    print(f"Is r2 the expected result?: {torch.allclose(r2, factorial(len(operands))*r1)}")

    # The operands can also be updated using in-place operations if they are on the GPU.
    for i, operand in enumerate(operands, start=1):
        operand /= i

    #The operands don't have to be reset for in-place operations. Perform the contraction.
    print("Reset the operands in-place and perform the contraction (r3):")
    r3 = n.contract()
    print(r3)
    print(f"Is r3 the expected result?: {torch.allclose(r3, r1)}")

# The context manages the network resources, so n.free() doesn't have to be called.
