# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Memory specification regular expression.
"""

__all__ = ['MEM_LIMIT_RE_PCT', 'MEM_LIMIT_RE_VAL', 'MEM_LIMIT_DOC']

import re

MEM_LIMIT_RE_PCT = re.compile(r"(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*%\s*$")
MEM_LIMIT_RE_VAL = re.compile(r"(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(?P<units>[kmg])?(?P<binary>(?<=[kmg])i)?b\s*$", re.IGNORECASE)
MEM_LIMIT_DOC = """The memory limit must be specified in one of the following forms:
  (1) A number (int or float). If the number is between 0 and 1, the memory limit is interpreted as a fraction of the 
      total device memory. 
      Examples: 0.75, 50E6, 50000000, ...
  (2) A string containing a value followed by B, kB, MB, or GB for powers of 1000. 
      Examples: "0.05 GB", "50 MB", "50000000 B" ...
  (3) A string containing a value followed by kiB, MiB, or GiB for powers of 1024.
      Examples:  "0.05 GB", "51.2 MB", "53687091 B" ...
  (4) A string with value in the range (0, 100] followed by a %% symbol.
      Examples: "26%%", "82%%", ...
  
  Whitespace between values and units is optional.

The provided memory limit is "%s".
"""

