import itertools
try:
    import torch
except ImportError:
    torch = None


# TODO: investigate test parallelism across cartesian product

sources = [
    "numpy",
    "cupy",
]
if torch:
    sources.append("torch")

devices = [
    "cpu",
    "cuda"
]

dtype_names = [
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128"
]

sources_devices_dtype_names = list(
    itertools.product(
        sources,
        devices,
        dtype_names
    )
)

array_orders = ["C", "F"]

einsum_expressions = [
    ("ea,fb,abcd,gc,hd->efgh",
    (1, 1, 0, 1, 1),
    [(10, 10, 10, 10), (10, 10)]),

    ("ea,fb,abcd,gc,hd",
    (1, 1, 0, 1, 1),
    [(10, 10, 10, 10), (10, 10)]),

    ("ij,jk,kl->il",
    (0, 1, 2),
    [(2, 2), (2, 5), (5, 2)]),

    ("ij,jk,kl",
    (0, 1, 2),
    [(2, 2), (2, 5), (5, 2)]),

    ("ij,jk,ki",
    (0, 1, 2),
    [(2, 2), (2, 5), (5, 2)])
]

compute_types = [None]
device_ids = [None]
handles = [None]
loggers = [None]
memory_limits = [
    int(1e8),
    "100 MiB",
    "80%"
]

opt_cmodes = [None, "dict", "object"]

network_options = [dict(zip(
    ("compute_type", "device_id", "handle", "logger", "memory_limit"),
    network_option_pack))
    for network_option_pack in itertools.product(compute_types, device_ids, handles, loggers, memory_limits)
]

samples = [None]
path = [None]
slicing = [None]
reconfiguration = [None]
seed = [None]

optimizer_options = [dict(zip(
    ("samples", "path", "slicing", "reconfiguration", "seed"),
    optimizer_options_pack))
    for optimizer_options_pack in itertools.product(samples, path, slicing, reconfiguration, seed)
]

iterations = [0, 7]  # 0 iterations is equivalent to no autotuning
