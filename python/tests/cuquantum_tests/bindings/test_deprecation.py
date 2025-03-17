# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest

class TestBindingsDeprecation:

    def _verify_binding(self, lib_name):
        # make sure bindings are still accessible
        lib = importlib.import_module(f'cuquantum.bindings.{lib_name}')
        lib_deprecated = importlib.import_module(f'cuquantum.{lib_name}')
        binding_api_names = [name for name in dir(lib) if not name.startswith('_')]
        assert len(binding_api_names) > 10 # make sure this is not empty
        for name in binding_api_names:
            if name == 'logger_callback_holder':
                # From cutensornet.logger_set_callback_data
                continue
            assert getattr(lib, name, None) is getattr(lib_deprecated, name, None), f"{lib_name} {name} not matching"

    @pytest.mark.parametrize("lib_name", ('custatevec', 'cutensornet'))
    def test_binding_deprecation(self, lib_name):
        self._verify_binding(lib_name)
