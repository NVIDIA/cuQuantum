import glob
import os
import subprocess
import sys

import pytest


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
