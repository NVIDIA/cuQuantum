# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os

import pytest

from ..helpers import run_sample

sample_files = []
for sub_directory in ('pauliprop', 'bindings/cupauliprop'):
    samples_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'samples', sub_directory))
    sample_files += glob.glob(samples_path+'*/*.py', recursive=True)

@pytest.mark.parametrize("sample", sample_files)
class TestcuPauliPropSamples:

    def test_sample(self, sample):
        run_sample(samples_path, sample)
