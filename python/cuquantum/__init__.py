from cuquantum import custatevec
from cuquantum import cutensornet
from cuquantum.cutensornet import (
    contract, contract_path, einsum, einsum_path, Network,
    NetworkOptions, OptimizerInfo, OptimizerOptions, PathFinderOptions, ReconfigOptions, SlicerOptions)
from cuquantum.utils import ComputeType, cudaDataType, libraryPropertyType
from cuquantum._version import __version__


# We patch all enum values so that they have the correct docstrings
for enum in (
        custatevec.Pauli,
        custatevec.MatrixLayout,
        custatevec.MatrixType,
        custatevec.Collapse,
        custatevec.SamplerOutput,
        cutensornet.ContractionOptimizerInfoAttribute,
        cutensornet.ContractionOptimizerConfigAttribute,
        cutensornet.ContractionAutotunePreferenceAttribute,
        ):
    cutensornet._internal.enum_utils.add_enum_class_doc(enum, chomp="_ATTRIBUTE|_PREFERENCE_ATTRIBUTE")
# these have yet another convention...
for v in cutensornet.GraphAlgorithm:
    v.__doc__ = f"See `CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM_{v.name}`."
cutensornet.MemoryModel.SLICER_HEURISTIC.__doc__ = \
    f"See `CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL_HEURISTIC`."
cutensornet.MemoryModel.SLICER_CUTENSOR.__doc__ = \
    f"See `CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL_CUTENSOR`."

del enum, utils, v
