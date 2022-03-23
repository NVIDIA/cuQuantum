import functools

import cupy
import numpy
try:
    import torch
except ImportError:
    torch = None

from cuquantum import Network
from cuquantum import NetworkOptions, OptimizerOptions

from .data import *


def infer_object_package(name):
    return name.split('.')[0]

def dtype_name_dispatcher(source, dtype_name):
    import sys
    return getattr(sys.modules[source], dtype_name)


stream_names = [
    "default",
    "cupy",
]

streams = dict(zip(
    stream_names,
    [None, cupy.cuda.Stream()]
))

if torch:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stream_names.append("torch")
    streams["torch"] = torch.cuda.Stream()


def stream_name_sync_dispatcher(stream_name, skip=False):
    stream = streams[stream_name]
    if not skip:
        if stream:
            stream.synchronize()

# TODO: record seed
def generate_data_dispatcher(source, device, shape, dtype_name, array_order):
    data = None
    dtype = dtype_name_dispatcher(source, dtype_name)
    if source == "numpy":
        if "int" in dtype_name:
            data = numpy.random.randint(-1, high=2, size=shape).astype(order=array_order)
        elif "complex" in dtype_name:
            data = (numpy.random.random(shape) +
                    1.j * numpy.random.random(shape)).astype(dtype, order=array_order)
        else:
            data = numpy.random.random(shape).astype(dtype, order=array_order)
    elif source == "cupy":
        if "int" in dtype_name:
            data = cupy.random.randint(-1, high=2, size=shape).astype(order=array_order)
        elif "complex" in dtype_name:
            data = (cupy.random.random(shape) +
                    1.j * cupy.random.random(shape)).astype(dtype, order=array_order)
        else:
            data = cupy.random.random(shape).astype(dtype, order=array_order)
    elif torch and source == "torch":
        if "int" in dtype_name:
            data = torch.randint(-1, 2, shape, dtype=dtype, device=device)
        else:
            data = torch.rand(shape, dtype=dtype, device=device)
    return data

def generate_data(source, device, shape, dtype_name, array_order):
    return generate_data_dispatcher(source, device, shape, dtype_name, array_order)

def generate_data_operands(source, device, dtype_name, array_order, einsum_expression):
    einsum_expr, orders, shapes = einsum_expression
    data = [generate_data(source, device, shape, dtype_name, array_order) for shape in shapes]
    return [data[order] for order in orders]

def data_to_numpy(source, data):
    if source == "numpy":
        return data
    elif source == "cupy":
        return cupy.asnumpy(data)
    elif torch and source == "torch":
        return data.cpu().numpy()

def data_operands_to_numpy(source, data_operands):
    return [data_to_numpy(source, data) for data in data_operands]

def interleaved_format_from_einsum(einsum_expr, data_operands):
    einsum_tuples = einsum_expr.split("->")[0].split(",")
    index_tuples = [[it for it in einsum_tuple] for einsum_tuple in einsum_tuples]
    inputs = []
    for index, data in enumerate(data_operands):
        inputs.append(data)
        inputs.append(index_tuples[index])
    return inputs

def einsum_dispatcher(source, einsum_expr, data_operands):
    if source == "numpy":
        return numpy.einsum(einsum_expr, *data_operands, optimize="optimal")
    elif source == "cupy":
        return cupy.einsum(einsum_expr, *data_operands)
    elif torch and source == "torch":
        return torch.einsum(einsum_expr, *data_operands)

def network_options_dispatcher(network_options, mode=None):
    if mode is None:
        return None
    elif mode == "dict":
        return network_options
    elif mode == "object":
        return NetworkOptions(**network_options)

def optimizer_options_dispatcher(optimizer_options, mode=None):
    if mode is None:
        return None
    elif mode == "dict":
        return optimizer_options
    elif mode == "object":
        return OptimizerOptions(**optimizer_options)

def network_dispatcher(einsum_expr, data_operands, network_options, mode=None, interleaved_inputs=None):
    if interleaved_inputs:
        return Network(
            *interleaved_inputs,
            options=network_options_dispatcher(network_options, mode=mode)
        )
    else:
        return Network(
            einsum_expr, 
            *data_operands,
            options=network_options_dispatcher(network_options, mode=mode)
        )

def machine_epsilon(dtype_name):
    dtype = dtype_name_dispatcher("numpy", dtype_name)
    return numpy.finfo(dtype).eps

machine_epsilon_values = [machine_epsilon(dtype_name) for dtype_name in dtype_names]

rtol_mapper = dict(zip(
    dtype_names,
    [numpy.sqrt(m_eps) for m_eps in machine_epsilon_values]
))

atol_mapper = dict(zip(
    dtype_names,
    [10 * m_eps for m_eps in machine_epsilon_values]
))

def allclose_dispatcher(source, dtype_name):
    if source == "numpy":
        return functools.partial(
            numpy.allclose, rtol=rtol_mapper[dtype_name],
            atol=atol_mapper[dtype_name]
        )
    elif source == "cupy":
        return functools.partial(
            cupy.allclose, rtol=rtol_mapper[dtype_name],
            atol=atol_mapper[dtype_name]
        )
    elif torch and source == "torch":
        return functools.partial(
            torch.allclose, rtol=rtol_mapper[dtype_name],
            atol=atol_mapper[dtype_name]
        )

def allclose(source, dtype_name, tensor, ref_tensor):
    allclose_func = allclose_dispatcher(source, dtype_name)
    assert allclose_func(tensor, ref_tensor)

def tensor_class_dispatcher(data_operands):
    return type(data_operands[0])

def tensor_class_equal(tensor_class, data_operands, result):
    for data_operand in data_operands:
        assert issubclass(tensor_class, type(data_operand))
    assert issubclass(tensor_class, type(result))

# TODO: document rationale ... torch inconsistent with numpy
def dtypes_equal(dtype, data_operands, result):
    for data_operand in data_operands:
        assert data_operand.dtype == dtype
    assert result.dtype == dtype

class NetworkRuntimeOptions:
    def __init__(self, runtime_options_pack):
        (sources_devices_dtype_name,
         array_order,
         einsum_expression,
         options,
         optimize,
         options_cmode,
         optimize_cmode,
         iterations) = runtime_options_pack
        self.source, self.device, self.dtype_name = sources_devices_dtype_name
        self.dtype = dtype_name_dispatcher(self.source, self.dtype_name)
        self.array_order = array_order
        self.einsum_expression = einsum_expression
        self.einsum_expr, self.orders, self.shapes = einsum_expression  # verbose, convenient
        self.options = options
        self.options_cmode = options_cmode
        self.optimize = optimize
        self.optimize_cmode = optimize_cmode
        self.iterations = iterations
 
class ProxyFixtureBase(NetworkRuntimeOptions):
    def __init__(self, network_options_pack):
        super().__init__(network_options_pack)
        self.data_operands = generate_data_operands(
            self.source,
            self.device,
            self.dtype_name,
            self.array_order,
            self.einsum_expression
        )
        self.numpy_data_operands = data_operands_to_numpy(
            self.source,
            self.data_operands
        )
        self.tensor_class = tensor_class_dispatcher(self.data_operands)
        self.tensor_package = infer_object_package(self.tensor_class.__module__)
        self.interleaved_inputs = interleaved_format_from_einsum(self.einsum_expr, self.data_operands)
        self.numpy_einsum_path = numpy.einsum_path(self.einsum_expr, *self.numpy_data_operands)
        self.einsum = einsum_dispatcher(self.source, self.einsum_expr, self.data_operands)
