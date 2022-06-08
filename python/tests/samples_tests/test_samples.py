# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import subprocess
import sys

try:
    import nbmake
except ImportError:
    nbmake = None
import pytest

# we could use packaging.version.Version too, but NumPy is our required
# dependency, packaging is not.
from numpy.lib import NumpyVersion as Version

circuit_versions = dict()
try:
    import cirq
    circuit_versions['cirq'] = Version(cirq.__version__)
except ImportError:
    circuit_versions['cirq'] = Version('0.0.0')  # no cirq

try:
    import qiskit
    circuit_versions['qiskit'] = Version(qiskit.__qiskit_version__['qiskit'])  # meta package version
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


class cuQuantumSampleTestError(Exception):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


samples_path = os.path.join(
    os.path.dirname(__file__), '..', '..', 'samples')
sample_files = glob.glob(samples_path+'/**/*.py', recursive=True)


def run_sample(path, *args):
    fullpath = os.path.join(samples_path, path)
    result = subprocess.run(
        (sys.executable, fullpath) + args, capture_output=True, env=os.environ)
    if result.returncode:
        msg = f'Got error:\n'
        msg += f'{result.stderr.decode()}'
        if "ModuleNotFoundError: No module named 'torch'" in msg:
            pytest.skip('PyTorch uninstalled, skipping related tests')
        else:
            raise cuQuantumSampleTestError(msg)
    else:
        print(result.stdout.decode())


@pytest.mark.parametrize(
    'sample', sample_files
)
class TestSamples:

    def test_sample(self, sample):
        run_sample(sample)


notebook_files = glob.glob(samples_path+'/**/*.ipynb', recursive=True)


@pytest.mark.skipif(
    nbmake is None,
    reason="testing Jupyter notebooks requires nbmake"
)
@pytest.mark.parametrize(
    'notebook', notebook_files
)
class TestNotebooks:

    def test_notebook(self, notebook):
        circuit_type = os.path.basename(notebook).split('_')[0]
        if circuit_type in notebook_skip_messages:
            pytest.skip(notebook_skip_messages[circuit_type])
        else:
            status = pytest.main(['--nbmake', notebook])
            if status != 0:
                raise cuQuantumSampleTestError(f'{notebook} failed')
