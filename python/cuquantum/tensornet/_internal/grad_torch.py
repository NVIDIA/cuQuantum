# Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import weakref
from typing import Any

# TODO: right now, we use try-except throughout the codebase to check the
# presence of PyTorch, so if it exists it'd get imported. We should switch
# to use importlib.util.find_spec('torch') so as to reduce the import time.

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device


try:
    import torch
except ImportError:
    torch = None


def _requires_grad(*operands: Any) -> bool:
    """Return whether applying the Torch contraction will create an autograd graph."""
    return (
        torch is not None
        and torch.is_grad_enabled()
        and any(torch.is_tensor(operand) and operand.requires_grad for operand in operands)
    )


if torch is not None:

    class _TorchContract(torch.autograd.Function):

        @staticmethod
        def forward(
            context: Any,
            network: Any,
            optimize: Any,
            stream: Any,
            return_info: bool,
            requires_grad: bool,
            *operands: Any,
        ) -> Any:
            """Run the contraction and assign cleanup ownership for the Network."""

            # Save objects needed in the backward pass.
            context.network = network
            context.stream = stream
            # Release the Network if the autograd graph is discarded.
            context.network_finalizer = weakref.finalize(context, network.free)

            try:
                # Compute path.
                opt_info = network.contract_path(optimize=optimize)

                # Skip autotuning since the network is contracted only once.

                # Contraction.
                out = network.contract(stream=stream)
            except BaseException:
                context.network_finalizer()
                raise

            # No backward pass can occur when PyTorch did not create a graph.
            if not requires_grad:
                context.network_finalizer()

            return (out, opt_info) if return_info else out

        @staticmethod
        def backward(context, *output_grad):

            old_device = None

            try:
                # Retrieve cached objects.
                network = context.network
                stream = context.stream

                # Regardless of return_info, we only care about the gradient of
                # the first return value.
                output_grad = output_grad[0]

                # Compute backprop.
                # WAR: pytorch backward propagation seems to uninitialize the device
                old_device = Device()
                execution_device = Device(network.device_id)
                execution_device.set_current()
                input_grads = network.gradients(output_grad, stream=stream)

                # Rearrange return values based on the input format.
                if network.is_interleaved:
                    out = [None, None, None, None, None,
                           *itertools.chain(*itertools.zip_longest(input_grads, [None]))]
                    if network.has_user_output:
                        out.append(None)
                    out = tuple(out)
                else:
                    out = (None, None, None, None,
                           None, None, *input_grads)
            finally:
                # Free network resources explicitly.
                try:
                    if old_device is not None:
                        old_device.set_current()
                finally:
                    context.network_finalizer()

            return out

else:

    _TorchContract = None
