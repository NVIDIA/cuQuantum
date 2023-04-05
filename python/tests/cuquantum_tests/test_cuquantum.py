# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys

import pytest


# TODO: mark this test as slow and don't run it every time
class TestModuleUtils:

    @pytest.mark.parametrize(
        'includes', (True, False)
    )
    @pytest.mark.parametrize(
        'libs', (True, False)
    )
    @pytest.mark.parametrize(
        'target', (None, 'custatevec', 'cutensornet', True)
    )
    def test_cuquantum(self, includes, libs, target):
        # We need to launch a subprocess to have a clean ld state
        cmd = [sys.executable, '-m', 'cuquantum']
        if includes:
            cmd.append('--includes')
        if libs:
            cmd.append('--libs')
        if target:
            if target is True:
                cmd.extend(('--target', 'custatevec'))
                cmd.extend(('--target', 'cutensornet'))
            else:
                cmd.extend(('--target', target))

        result = subprocess.run(cmd, capture_output=True, env=os.environ)
        if result.returncode:
            if includes is False and libs is False and target is None:
                assert result.returncode == 1
                assert 'usage' in result.stdout.decode()
                return
            msg = f'Got error:\n'
            msg += f'stdout: {result.stdout.decode()}\n'
            msg += f'stderr: {result.stderr.decode()}\n'
            assert False, msg

        out = result.stdout.decode().split()
        if includes:
            assert any([s.startswith('-I') for s in out])
        if libs:
            assert any([s.startswith('-L') for s in out])
        if target:
            assert any([s.startswith('-l') for s in out])
