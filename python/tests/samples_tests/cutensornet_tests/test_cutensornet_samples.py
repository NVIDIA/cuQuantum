# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import re
import sys

# we could use packaging.version.Version too, but NumPy is our required
# dependency, packaging is not.
from numpy.lib import NumpyVersion as Version
import pytest

from ..test_utils import cuQuantumSampleTestError, run_sample


circuit_versions = dict()
try:
    import cirq
    circuit_versions['cirq'] = Version(cirq.__version__)
except ImportError:
    circuit_versions['cirq'] = Version('0.0.0')  # no cirq

try:
    import qiskit
    if hasattr(qiskit, '__qiskit_version__'):
        circuit_versions['qiskit'] = Version(qiskit.__qiskit_version__['qiskit']) # meta package version
    else:
        # qiskit 1.0
        circuit_versions['qiskit'] = Version(qiskit.__version__)
except ImportError:
    circuit_versions['qiskit'] = Version('0.0.0')  # no qiskit


# minimal versions to run samples/circuit_converter/cirq/qsikit_advanced.ipynb
# slightly higher than the minimal versions for CircuitToEinsum to work with cirq/qiskit

NOTEBOOK_MIN_VERSIONS = {'cirq': Version('0.7.0'),
                         'qiskit': Version('0.25.0')}

notebook_skip_messages = dict()
for circuit_type, current_version in circuit_versions.items():
    min_version = NOTEBOOK_MIN_VERSIONS[circuit_type]
    if current_version < min_version:
        notebook_skip_messages[circuit_type] = (
            f"testing {circuit_type} notebooks requires "
            f"{circuit_type}>={NOTEBOOK_MIN_VERSIONS[circuit_type].version}"
        )


samples_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'samples', 'cutensornet')
sample_files = glob.glob(samples_path+'/**/*.py', recursive=True)

# Handle MPI tests separately.
mpi_re = r".*_mpi[_]?.*\.py"
sample_files = list(filter(lambda f: not re.search(mpi_re, f), sample_files))


@pytest.mark.parametrize(
    'sample', sample_files
)
class TestcuTensorNetSamples:

    def test_sample(self, sample):
        run_sample(samples_path, sample)


notebook_files = glob.glob(samples_path+'/**/*.ipynb', recursive=True)


@pytest.mark.parametrize(
    'notebook', notebook_files
)
class TestNotebooks:

    def test_notebook(self, notebook):
        circuit_type = os.path.basename(notebook).split('_')[0]
        if circuit_type in notebook_skip_messages:
            pytest.skip(notebook_skip_messages[circuit_type])
        else:
            run_sample(samples_path, notebook)
