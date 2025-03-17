# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools

# TODO: right now, we use try-except throughout the codebase to check the
# presence of PyTorch, so if it exists it'd get imported. We should switch
# to use importlib.util.find_spec('torch') so as to reduce the import time.
try:
    import torch
except ImportError:
    torch = None


if torch is not None:

    class _TorchContract(torch.autograd.Function):

        @staticmethod
        def forward(context, network, optimize, stream, return_info, *operands):

            # Save objects needed in the backward pass.
            context.network = network
            context.stream = stream

            # Compute path.
            opt_info = network.contract_path(optimize=optimize)

            # Skip autotuning since the network is contracted only once.

            # Contraction.
            out = network.contract(stream=stream)

            if return_info:
                return out, opt_info
            else:
                return out

        @staticmethod
        def backward(context, *output_grad):

            try:
                # Retrieve cached objects.
                network = context.network
                stream = context.stream

                # Regardless of return_info, we only care about the gradient of
                # the first return value.
                output_grad = output_grad[0]

                # Compute backprop.
                input_grads = network.gradients(output_grad, stream=stream)

                # Rearrange return values based on the input format.
                if network.is_interleaved:
                    out = [None, None, None, None,
                           *itertools.chain(*itertools.zip_longest(input_grads, [None]))]
                    if network.has_user_output:
                        out.append(None)
                    out = tuple(out)
                else:
                    out = (None, None, None, None,
                           None, *input_grads)
            finally:
                # Free network resources explicitly.
                network.free()
 
            return out

else:

    _TorchContract = None
