# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
try:
    import xdist
except ImportError:
    @pytest.fixture(scope="session")
    def worker_id(request):
        return "master"
else:
    del pytest, xdist
