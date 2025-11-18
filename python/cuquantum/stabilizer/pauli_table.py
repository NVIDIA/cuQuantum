# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Union
from .utils import _unpack_arrays, Array
import numpy as np

# Pauli operator encoding map: (x_bit, z_bit) -> Pauli operator
PAULI_MAP = {
    (0, 0): ".",
    (1, 0): "X",
    (0, 1): "Z",
    (1, 1): "Y",
}

def _move_to_cpu_if_needed(*args):
    ret = []
    for a in args:
        if isinstance(a, np.ndarray):
            ret.append(a)
        else:
            if hasattr(a, "get"):
                ret.append(a.get())
            else:
                raise ValueError(f"Unsupported array type: {type(a)}")
    return ret

class PauliFrame:
    """A weight-less Pauli string.

    This class is primarily used for visualizing the simulation output and is
    not intended to be constructed in batch quantities.  When provided with a
    device array, the bits are automatically transferred to the CPU and unpacked
    if needed.

    If the intention is to calculate a property derived from many Pauli strings,
    the best performance is achieved by directly manipulating the GPU-based Pauli table.

    """

    num_qubits: int


    def __init__(
        self,
        x_bits: Array,
        z_bits: Array,
        num_qubits: Union[int, None] = None,
        bit_packed: bool = False,
    ):
        """Initialize a PauliFrame.

        x_bits and z_bits can both be either a CPU array or a GPU array.
        If `bit_packed=False`: array with `num_qubits` elements of dtype `uint8` or `bool`
        If `bit_packed=True`: array with `ceil(num_qubits/8)` elements of dtype `uint32`

        Args:
            x_bits: X bits for each qubit.
            z_bits: Z bits for each qubit (same format as x_bits).
            num_qubits: Number of qubits. Must be specified if `bit_packed=True`
            bit_packed: Whether the input bits are in packed format (default: `False`)
        """

        x, z = _move_to_cpu_if_needed(x_bits, z_bits)
        if bit_packed:
            if num_qubits is None:
                raise ValueError("num_qubits must be specified if bit_packed=True")
            x, z = _unpack_arrays(num_qubits, x, z)
        self._xz = np.array([x, z]).T
        self.num_qubits = len(self._xz)

    def __getitem__(self, qubit_idx: int) -> str:
        """Get a Pauli operator for a specific qubit.

        Returns: a one-character string, one of `'IXZY'`
        """
        return PAULI_MAP[tuple(self._xz[qubit_idx])]

    def to_string(self) -> str:
        """Convert to string representation.

        Returns: a string of characters `'IXZY'`
        """
        paulis = [self[i] for i in range(self.num_qubits)]
        return "".join(paulis)

    def __repr__(self) -> str:
        return f"PauliString[{self.num_qubits}]('{self.to_string()}')"

    def __str__(self) -> str:
        return self.to_string()


class PauliTable:
    """Holds Pauli frame table data.

    The table can store data in one of 4 formats:
    - bit-packed on CPU
    - unpacked on GPU
    - bit-packed on GPU
    - unpacked on CPU

    Attributes:
        x_table: X bit table (NumPy or CuPy array)
        z_table: Z bit table (NumPy or CuPy array)
        bit_packed: Whether the tables are in bit-packed format (default: `False`)
        num_qubits: Number of qubits
        num_paulis: Number of Paulis
    """

    num_qubits: int
    num_paulis: int

    def __init__(
        self,
        x_table: Array,
        z_table: Array,
        num_paulis: int,
        num_qubits: int,
        bit_packed: bool = False,
    ):
        """Initialize a PauliTable.

        Args:
            x_table: X bit table (NumPy or CuPy array)
            z_table: Z bit table (NumPy or CuPy array)
            num_paulis: Number of Pauli samples 
            num_qubits: Number of qubits
            bit_packed: Whether tables are in bit-packed format (default: `False`)
        """
        self.bit_packed = bit_packed
        self.num_paulis = num_paulis
        self.num_qubits = num_qubits
        self.x_table = x_table.view("uint8").reshape(num_qubits, -1)
        self.z_table = z_table.view("uint8").reshape(num_qubits, -1)
        if bit_packed:
            assert (num_paulis + 31) // 32 == x_table.shape[
                1
            ] // 4, f"{(num_paulis+31)//32=} must be <= {x_table.shape[1]//4=} if bit_packed=True"
        else:
            assert (
                num_paulis == x_table.shape[1]
            ), "num_paulis must be x_table.shape[1] if bit_packed=False"

    def __getitem__(self, col_idx: int) -> PauliFrame:
        """Get a Pauli frame for the given index.

        Automatically transfers data from GPU to CPU and returns a PauliFrame
        representing all Paulis in that column.

        Args:
            col_idx: Pauli frame index, between 0 and `num_paulis - 1`

        Returns:
            :py:class:`PauliFrame` object for that column
        """
        if not self.bit_packed:
            xbits = self.x_table[:, col_idx]
            zbits = self.z_table[:, col_idx]
            return PauliFrame(xbits, zbits, num_qubits=self.num_qubits, bit_packed=False)

        xwords = self.x_table[:, col_idx // 8]
        zwords = self.z_table[:, col_idx // 8]
        xbits = xwords >> (col_idx % 8) & 1
        zbits = zwords >> (col_idx % 8) & 1
        return PauliFrame(xbits, zbits, num_qubits=self.num_qubits, bit_packed=False)

    def __len__(self) -> int:
        """Return the number of Pauli strings (columns)."""
        return self.num_paulis

    def __iter__(self):
        """Iterate over all PauliStrings in the table."""
        for i in range(self.num_paulis):
            yield self[i]
