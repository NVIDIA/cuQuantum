# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import sys


from cuquantum.tensornet import _internal
from cuquantum.bindings._utils import WHITESPACE_UNICODE


class TestGetSymbol:

    def test_no_whitespace(self):
        # Note: max(whitespace_s) = 12288
        out = []
        for i in range(0, 30000):
            s = _internal.circuit_converter_utils._get_symbol(i)
            assert not s.isspace()
            out.append(s)

        # check the mapping is unique
        assert len(set(out)) == 30000

    def test_whitespace_unicode_consistency(self):
        all_s = ''.join(chr(c) for c in range(sys.maxunicode+1))
        whitespace_s = ''.join(re.findall(r'\s', all_s))
        assert WHITESPACE_UNICODE == whitespace_s
