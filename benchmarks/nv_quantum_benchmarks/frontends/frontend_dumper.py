# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cmath
import logging
import os
from math import pi
import numpy as np

from .frontend import Frontend
from .._utils import call_by_root
from ..constants import LOGGER_NAME


# set up a logger
logger = logging.getLogger(LOGGER_NAME)


class Dumper(Frontend):
    """Special frontend for dumping the gate sequence as pure text to disk.

    The first line of the file is the number of qudits.

    The second line of the file contains a sequence of integers, one integer per qudit, 
    specifying qudit dimensions (qubit dimension equals 2).

    Following the header is the gate data. Each gate (or operation) would be stored as 3 lines, with elements separated by 1 space:

      1. n_targets n_controls
      2. targets controls
      3. contiguity actual_matrix_data
    
    Regarding item 3, there are two available options specified using the CUQUANTUM_BENCHMARKS_TCS_FULL_TENSOR environment variable. 
    When set to 0 (which is the default), only the target matrix is included in the output.
    Conversely, when set to 1, the output includes the full matrix, including control qubits, and when set to 2, both formats generate separately. 

    Note that the qubit IDs are zero-based. The matrix data is flattened to a 1D contiguous
    array of length 2**(2*n_targets). The contiguity is a single character "C" (for C-major,
    or row-major) or "F" (for Fortran-major, or column-major) for how to interpret the matrix.
    All complex numbers are stored as two real numbers (ex: 0.5-0.1j -> "0.5 -0.1").

    As an example, a CCX gate acting on qubit 0 and controlled by qubits 2 & 4, when CUQUANTUM_BENCHMARKS_TCS_FULL_TENSOR=0, is stored as

      '''
      5\n
      2 2 2 2 2\n\n
      1 2\n
      0 2 4\n
      C 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0\n\n
      '''

    Currently the measurement operation at the end of the gate sequence is not stored.

    An empty line can be used to separate different gates/operations and improve readability,
    but it is not required.

    Run command example: CUQUANTUM_BENCHMARKS_DUMP_GATES=1 CUQUANTUM_BENCHMARKS_TCS_FULL_TENSOR=2 quantum-benchmarks circuit 
    --frontend cirq --backend cirq --benchmark qaoa --nqubits 8
    """

    def __init__(self, nqubits, config):
        precision = config['precision']
        self.dtype = np.complex64 if precision == 'single' else np.complex128
        self.dtype = np.dtype(self.dtype)
        circuit_filename = config['circuit_filename']
        self.circuit_filename = circuit_filename.replace('.pickle', '_raw.txt')
        self.nqubits = nqubits
        self.order = 'C'  # TODO
        self.digits = 14  # TODO
        self.tcs_full = int(os.environ.get('CUQUANTUM_BENCHMARKS_TCS_FULL_TENSOR', 0))

    def _dump_op(self, op, targets, controls=()):
        np.set_printoptions(threshold=np.inf)
        flattened_op = op.astype(self.dtype).reshape(-1, order=self.order).flatten().view(self.dtype.char.lower())
        op = np.array2string(
            flattened_op,
            max_line_width=np.inf,
            precision=self.digits,
        )

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

    def generateSpecifiedCircuit(self, gateSeq):
        circuit = str(self.nqubits) + '\n'
        for i in range(self.nqubits):
            circuit += '2 '
        circuit += '\n\n'

        for g in gateSeq:
            if isinstance(g.targets, int):
                targets = [g.targets]
            else:
                targets = g.targets
            if isinstance(g.controls, int):
                controls = [g.controls]
            else:
                controls = g.controls

            if g.id == 'h':
                circuit += self._dump_op(np.asarray([[1, 1], [1, -1]])/np.sqrt(2), targets)
                
            elif g.id == 'x':
                circuit += self._dump_op(np.asarray([[0, 1], [1, 0]]), targets)

            elif g.id == 'rz':
                circuit += self._dump_op(self._get_rotation_matrix(0, g.params, 0), targets)

            elif g.id == 'rx':
                circuit += self._dump_op(self._get_rotation_matrix(g.params, -pi/2, pi/2), targets)

            elif g.id == 'ry':
                circuit += self._dump_op(self._get_rotation_matrix(g.params, 0, 0), targets)

            elif g.id == 'swap':
                assert len(targets) == 2
                matrix = np.eye(4, dtype=self.dtype)
                matrix[1:3, 1:3] = [[0, 1], [1, 0]]
                circuit += self._dump_op(np.asarray(matrix), targets)

            elif g.id == 'u':
                circuit += self._dump_op(np.asarray(g.matrix), targets)

            elif g.id == 'cnot':
                if self.tcs_full:
                    circuit += self._dump_op(
                    np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), controls+targets)
                else:
                    circuit += self._dump_op(
                    np.asarray([[0, 1], [1, 0]]), targets, controls)

            elif g.id == 'cz':
                if self.tcs_full:
                    circuit += self._dump_op(
                    np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]), controls+targets)
                else:
                    circuit += self._dump_op(
                    np.asarray([[1, 0], [0, -1]]), targets, controls)

            elif g.id == 'czpowgate':
                if self.tcs_full:
                    matrix = np.eye(4, dtype=self.dtype)
                    matrix[3, 3] = cmath.exp(1j*pi*g.params)
                    circuit += self._dump_op(np.asarray(matrix), controls+targets)
                else:
                    matrix = np.eye(2, dtype=self.dtype)
                    matrix[1, 1] = cmath.exp(1j*pi*g.params)
                    circuit += self._dump_op(np.asarray(matrix), targets, controls)

            elif g.id == 'cu':
                if self.tcs_full:
                    matrix = np.eye(4, dtype=self.dtype)
                    matrix[2:4, 2:4] = g.matrix
                    circuit += self._dump_op(np.asarray(matrix), controls+targets)
                else:
                    circuit += self._dump_op(np.asarray(g.matrix), targets, controls)
                    
            elif g.id == 'mcu':
                if self.tcs_full:
                    n = 1 << (len(targets) + len(controls)) 
                    matrix = np.eye(n, dtype=self.dtype)
                    matrix[n-2:n, n-2:n] = g.matrix
                    circuit += self._dump_op(np.asarray(matrix), controls+targets)
                else:
                    circuit += self._dump_op(np.asarray(g.matrix), targets, controls)
                    
            elif g.id == 'measure':
                pass  # treated as no-op for now

            else:
                raise NotImplementedError(f"the gate type {g.id} is not defined")
        return circuit
    
    def generateBothCircuits(self, gateSeq):
        circuit1 = str(self.nqubits) + '\n'
        for i in range(self.nqubits):
            circuit1 += '2 '
        circuit1 += '\n\n'

        circuit2 = str(self.nqubits) + '\n'
        for i in range(self.nqubits):
            circuit2 += '2 '
        circuit2 += '\n\n'

        for g in gateSeq:
            if isinstance(g.targets, int):
                targets = [g.targets]
            else:
                targets = g.targets
            if isinstance(g.controls, int):
                controls = [g.controls]
            else:
                controls = g.controls

            if g.id == 'h':
                h = self._dump_op(np.asarray([[1, 1], [1, -1]])/np.sqrt(2), targets)
                circuit1 += h
                circuit2 += h
                
            elif g.id == 'x':
                x = self._dump_op(np.asarray([[0, 1], [1, 0]]), targets)
                circuit1 += x
                circuit2 += x

            elif g.id == 'rz':
                rz = self._dump_op(self._get_rotation_matrix(0, g.params, 0), targets)
                circuit1 += rz
                circuit2 += rz

            elif g.id == 'rx':
                rx = self._dump_op(self._get_rotation_matrix(g.params, -pi/2, pi/2), targets)
                circuit1 += rx
                circuit2 += rx

            elif g.id == 'ry':
                ry = self._dump_op(self._get_rotation_matrix(g.params, 0, 0), targets)
                circuit1 += ry
                circuit2 += ry
            
            elif g.id == 'swap':
                assert len(targets) == 2
                matrix = np.eye(4, dtype=self.dtype)
                matrix[1:3, 1:3] = [[0, 1], [1, 0]]
                circuit1 += self._dump_op(np.asarray(matrix), targets)
                circuit2 += self._dump_op(np.asarray(matrix), targets)

            elif g.id == 'u':
                circuit1 += self._dump_op(np.asarray(g.matrix), targets)
                circuit2 += self._dump_op(np.asarray(g.matrix), targets)

            elif g.id == 'cnot':
                circuit1 += self._dump_op(np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), controls+targets)
                circuit2 += self._dump_op(np.asarray([[0, 1], [1, 0]]), targets, controls)

            elif g.id == 'cz':
                circuit1 += self._dump_op(np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]), controls+targets)
                circuit2 += self._dump_op(np.asarray([[1, 0], [0, -1]]), targets, controls)

            elif g.id == 'czpowgate':
                matrix1 = np.eye(4, dtype=self.dtype)
                matrix1[3, 3] = cmath.exp(1j*pi*g.params)
                circuit1 += self._dump_op(np.asarray(matrix1), controls+targets)
                matrix2 = np.eye(2, dtype=self.dtype)
                matrix2[1, 1] = cmath.exp(1j*pi*g.params)
                circuit2 += self._dump_op(np.asarray(matrix2), targets, controls)

            elif g.id == 'cu':
                matrix1 = np.eye(4, dtype=self.dtype)
                matrix1[2:4, 2:4] = g.matrix
                circuit1 += self._dump_op(np.asarray(matrix1), controls+targets)
                circuit2 += self._dump_op(np.asarray(g.matrix), targets, controls)
                    
            elif g.id == 'mcu':
                n = 1 << (len(targets) + len(controls)) 
                matrix1 = np.eye(n, dtype=self.dtype)
                matrix1[n-2:n, n-2:n] = g.matrix
                circuit1 += self._dump_op(matrix1, controls+targets)
                circuit2 += self._dump_op(np.asarray(g.matrix), targets, controls)
                    
            elif g.id == 'measure':
                pass  # treated as no-op for now

            else:
                raise NotImplementedError(f"the gate type {g.id} is not defined")
        return circuit1, circuit2

    def generateCircuit(self, gateSeq):
        circuit = []
        circuit1 = []
        circuit2 = []

        if self.tcs_full == 2: # Generate both full matrix and target matrix version of circuits. This is useful for random benchmark to generate the same circuits
            circuit1, circuit2 = self.generateBothCircuits(gateSeq)
        
        else:
            circuit = self.generateSpecifiedCircuit(gateSeq)

        def dump():
            # Split the file name and extension
            parts = self.circuit_filename.rsplit('.', 1)
            if self.tcs_full==2:
                file_name1 = parts[0]+'_tcs_full.tcs'
                file_name2 = parts[0]+'_tcs_target.tcs'
                logger.info(f"dumping (raw) circuit as {file_name1} ...")
                with open(file_name1, 'w') as f1:
                    f1.write(circuit1)
                logger.info(f"dumping (raw) circuit as {file_name2} ...")
                with open(file_name2, 'w') as f2:
                    f2.write(circuit2)
            else:
                if self.tcs_full==1:
                    file_name = parts[0]+'_tcs_full.tcs'
                else:
                    file_name = parts[0]+'_tcs_target.tcs'
                logger.info(f"dumping (raw) circuit as {file_name} ...")
                with open(file_name, 'w') as f:
                    f.write(circuit)

        call_by_root(dump)
