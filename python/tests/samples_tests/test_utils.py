# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import gc
import os
import sys

import cupy as cp
try:
    import matplotlib
except ImportError:
    matplotlib = None
else:
    # disable plot windows from popping out when testing locally
    matplotlib.use('Agg')
import nbformat
from nbconvert import PythonExporter
import pytest


class cuQuantumSampleTestError(Exception):
    pass


def parse_python_script(filepath):
    if filepath.endswith('.py'):
        with open(filepath, "r", encoding='utf-8') as f:
            script = f.read()
    elif filepath.endswith('.ipynb'):
        # run all notebooks in the same process to avoid OOM & ABI issues
        with open(filepath, "r", encoding="utf-8") as f:
            nb = nbformat.reads(f.read(), nbformat.NO_CONVERT)
        script = PythonExporter().from_notebook_node(nb)[0]
    else:
        raise ValueError(f"{filepath} not supported")
    return script


def run_sample(samples_path, filename):
    fullpath = os.path.join(samples_path, filename)
    script = parse_python_script(fullpath)
    try:
        old_argv = sys.argv
        sys.argv = [fullpath]
        exec(script, {})
    except ImportError as e:
        # for samples/notebooks requiring any of optional dependencies
        for m in ('torch', 'qiskit', 'cirq'):
            if f"No module named '{m}'" in str(e):
                pytest.skip(f'{m} uninstalled, skipping related tests')
                break
        else:
            raise
    except Exception as e:
        msg = "\n"
        msg += f'Got error ({filename}):\n'
        msg += str(e)
        raise cuQuantumSampleTestError(msg) from e
    finally:
        sys.argv = old_argv
        # further reduce the memory watermark
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
