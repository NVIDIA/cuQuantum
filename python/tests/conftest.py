# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# The following configs are needed to deselect/ignore collected tests for
# various reasons, see pytest-dev/pytest#3730. In particular, this strategy
# is borrowed from https://github.com/pytest-dev/pytest/issues/3730#issuecomment-567142496.

from collections.abc import Iterable

import pytest


VALID_TEST_MARKERS = {"cudensitymat", "custatevec", "cutensornet", "utility"}

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "uncollect_if(*, func): function to unselect tests from parametrization"
    )

def mark_test_items(items):
    for item in items:
        path = str(item.fspath)
        for lib_name in ('cudensitymat', 'custatevec', 'cutensornet'):
            # bindings tests & sample tests
            if lib_name in path:
                item.add_marker(getattr(pytest.mark, lib_name))
        else:
            for module_name in ('densitymat', 'tensornet'):
                # tests for pythonic modules
                if f'cuquantum_tests/{module_name}/' in path:
                    item.add_marker(getattr(pytest.mark, f'cu{module_name}'))
    errors = []
    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        if not marker_names & VALID_TEST_MARKERS:
            errors.append(f"{item.nodeid} is missing a required group marker")

    if errors:
        error_message = "\n".join(errors)
        raise pytest.UsageError(f"The following tests are missing required markers:\n{error_message}")

def pytest_collection_modifyitems(config, items):
    mark_test_items(items)
    removed = []
    kept = []
    for item in items:
        is_removed = False
        m = item.get_closest_marker('uncollect_if')
        if m:
            funcs = m.kwargs['func']
            if not isinstance(funcs, Iterable):
                funcs = (funcs,)
            # loops over all deselect requirements
            for func in funcs:
                if func(**item.callspec.params):
                    removed.append(item)
                    is_removed = True
                    break
        if not is_removed:
            kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept
