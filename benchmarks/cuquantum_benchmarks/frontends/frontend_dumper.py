# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cmath
import logging
from math import pi

import numpy as np

from .frontend import Frontend
from .._utils import call_by_root


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


class Dumper(Frontend):
    """Special frontend for dumping the gate sequence as pure text to disk.

    Each gate (or operation) would be stored as 3 lines, with elements separated by 1 space:

      1. n_targets n_controls
      2. targets controls
      3. contiguity actual_matrix_data

    Note that the qubit IDs are zero-based. The matrix data is flattened to a 1D contiguous
    array of length 2**(2*n_targets). The contiguity is a single character "C" (for C-major,
    or row-major) or "F" (for Fortran-major, or column-major) for how to interpret the matrix.
    All complex numbers are stored as two real numbers (ex: 0.5-0.1j -> "0.5 -0.1").

    As an example, a CCX gate acting on qubit 0 and controlled by qubits 2 & 4 is stored as

      '''
      1 2\n
      0 2 4\n
      C 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0\n
      '''

    Currently the measurement operation at the end of the gate sequence is not stored.

    An empty line can be used to separate different gates/operations and improve readability,
    but it is not required.
    """

    def __init__(self, nqubits, config):
        precision = config['precision']
        self.dtype = np.complex64 if precision == 'single' else np.complex128
        self.dtype = np.dtype(self.dtype)
        circuit_filename = config['circuit_filename']
        self.circuit_filename = circuit_filename.replace('.pickle', '_raw.txt')
        self.nqubits = nqubits
        self.order = 'C'  # TODO
        self.digits = 12  # TODO

    def _dump_op(self, op, targets, controls=()):
        op = np.array2string(
            op.astype(self.dtype).reshape(-1, order=self.order).view(self.dtype.char.lower()),
            max_line_width=np.inf,
            precision=self.digits,
        )
        if isinstance(targets, int):
            targets = (targets,)
        if isinstance(controls, int):
            controls = (controls,)

        op_data = f"{len(targets)} {len(controls)}\n"
        for t in targets:
            op_data += f"{t} "
        for c in controls:
            op_data += f"{c} "
        op_data += f"\n{self.order} "
        op_data += f"{op[1:-1]}\n\n"

        return op_data

    def _get_rotation_matrix(self, theta, phi, lam):
        matrix = np.empty((2, 2), dtype=self.dtype)
        theta *= 0.5
        matrix[0, 0] = cmath.cos(theta)
        matrix[0, 1] = - cmath.sin(theta) * cmath.exp(1j*lam)
        matrix[1, 0] = cmath.sin(theta) * cmath.exp(1j*phi)
        matrix[1, 1] = cmath.cos(theta) * cmath.exp(1j*(phi+lam))
        matrix = np.asarray(matrix)
        return matrix

    def generateCircuit(self, gateSeq):
        circuit = ''

        for g in gateSeq:
            if g.id == 'h':
                circuit += self._dump_op(
                    np.asarray([[1, 1], [1, -1]])/np.sqrt(2), g.targets)

            elif g.id == 'x':
                circuit += self._dump_op(
                    np.asarray([[0, 1], [1, 0]]), g.targets)

            elif g.id == 'cnot':
                # TODO: use 4*4 version (merge targets & controls)?
                circuit += self._dump_op(
                    np.asarray([[0, 1], [1, 0]]), g.targets, g.controls)

            elif g.id == 'cz':
                # TODO: use 4*4 version (merge targets & controls)?
                circuit += self._dump_op(
                    np.asarray([[1, 0], [0, -1]]), g.targets, g.controls)

            elif g.id == 'rz':
                circuit += self._dump_op(
                    self._get_rotation_matrix(0, g.params, 0), g.targets)

            elif g.id == 'rx':
                circuit += self._dump_op(
                    self._get_rotation_matrix(g.params, -pi/2, pi/2), g.targets)

            elif g.id == 'ry':
                circuit += self._dump_op(
                    self._get_rotation_matrix(g.params, 0, 0), g.targets)

            elif g.id == 'czpowgate':
                matrix = np.eye(2, dtype=self.dtype)
                matrix[1, 1] = cmath.exp(1j*pi*g.params)
                circuit += self._dump_op(matrix, g.targets, g.controls)

            elif g.id == 'swap':
                assert len(g.targets) == 2
                matrix = np.eye(4, dtype=self.dtype)
                matrix[1:3, 1:3] = [[0, 1], [1, 0]]
                circuit += self._dump_op(matrix, g.targets)

            elif g.id == 'cu':
                circuit += self._dump_op(g.matrix, g.targets, g.controls)

            elif g.id == 'u':
                circuit += self._dump_op(g.matrix, g.targets)

            elif g.id == 'measure':
                pass  # treated as no-op for now

            else:
                raise NotImplementedError(f"the gate type {g.id} is not defined")

        def dump():
            logger.info(f"dumping (raw) circuit as {self.circuit_filename} ...")
            with open(self.circuit_filename, 'w') as f:
                f.write(circuit)

        call_by_root(dump)
