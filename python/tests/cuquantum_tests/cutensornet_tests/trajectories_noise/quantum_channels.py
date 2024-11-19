# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from dataclasses import dataclass
from typing import Union


@dataclass
class QuantumChannel:
    """
    dataclass to store info of quantum channel

    probs=None corresponds to uniform distribution
    """

    probs: Union[np.array, None, list]
    ops: list[np.array]

    def choose_op(self, dtype: str = "complex128"):
        i_ = np.random.choice(np.arange(len(self.probs)), p=self.probs)
        return self.ops[i_].astype(dtype)

    def mul_left(self, op: np.array):
        """
        In-place tensor multiply by `op` from left.
        Args:
        - op : Operator to tensordot with
            The modes of the operand is expected to be ordered as
            ``ABC...abc...``, where ``ABC...`` denotes output bra modes and
            ``abc...`` denotes input ket modes corresponding to ``modes``
        """
        twodim = lambda x: x.reshape(2 ** (x.ndim) // 2, -1)
        self.ops = [
            np.tensordot(twodim(op), twodim(o_), axes=0)
            .transpose([0, 2, 1, 3])
            .reshape((2,) * (op.ndim + o_.ndim))
            for o_ in self.ops
        ]


@dataclass
class QuantumGates:
    I = np.array([[1, 0], [0, 1]]).astype("complex128")
    X = np.array([[0, 1], [1, 0]]).astype("complex128")
    Y = np.array([[0, -1j], [1j, 0]]).astype("complex128")
    Z = np.array([[1, 0], [0, -1]]).astype("complex128")
    eZZ = np.diag(np.exp(1j * np.array([1, -1, -1, 1]))).astype("complex128")


def depolarizing_channel(l: float) -> QuantumChannel:
    """
    https://en.wikipedia.org/wiki/Quantum_depolarizing_channel
    """
    return QuantumChannel(
        probs=[1 - 3 * l / 4, l / 4, l / 4, l / 4],
        ops=[QuantumGates.I, QuantumGates.X, QuantumGates.Y, QuantumGates.Z],
    )


def bitflip_channel(p: float) -> QuantumChannel:
    """
    Probability of X rotation is `p`
    Probability of I rotation is `1-p`
    """
    return QuantumChannel(probs=[p, 1 - p], ops=[QuantumGates.X, QuantumGates.I])
