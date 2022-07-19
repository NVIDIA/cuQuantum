# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import pytest


class cuQuantumSampleTestError(Exception):
    pass


def run_sample(samples_path, filename):
    fullpath = os.path.join(samples_path, filename)
    with open(fullpath, "r", encoding='utf-8') as f:
        script = f.read()
    try:
        old_argv = sys.argv
        sys.argv = [fullpath]
        exec(script, {})
    except ImportError as e:
        if 'torch' not in str(e):
            raise
        else:
            pytest.skip('PyTorch uninstalled, skipping related tests')
    except Exception as e:
        msg = "\n"
        msg += f'Got error ({filename}):\n'
        msg += str(e)
        raise cuQuantumSampleTestError(msg) from e
    finally:
        sys.argv = old_argv
