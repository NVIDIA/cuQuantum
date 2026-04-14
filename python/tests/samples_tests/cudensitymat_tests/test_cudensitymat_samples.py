# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import re

import pytest

from ..helpers import run_sample

samples_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'samples', 'densitymat'))
sample_files = glob.glob(samples_path+'/**/*.py', recursive=True)

# Handle MPI and NCCL tests separately.
mpi_re = r".*_mpi[_]?.*\.py"
sample_files = list(filter(lambda f: not re.search(mpi_re, f), sample_files))

@pytest.mark.parametrize("sample", sample_files)
class TestcuDensityMatSamples:

    def test_sample(self, sample):
        run_sample(samples_path, sample)
