# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import logging
import sys
import multiprocessing

from .backends import backends
from .config import benchmarks
from .config import backends as backend_config
from .frontends import frontends
from .run_interface import run_interface
from ._utils import str_to_seq, MPHandler, RawTextAndDefaultArgFormatter


frontend_names = [f for f in frontends.keys()]
backend_names = [b for b in backends.keys()]
benchmark_names = [b for b in benchmarks.keys()]


description = r"""
============= NVIDIA cuQuantum Circuit Performance Benchmark Suite =============

Supported Backends:

  - aer: runs Qiskit Aer's CPU backend
  - aer-cuda: runs the native Qiskit Aer GPU backend
  - aer-cusv: runs Qiskit Aer's cuStateVec integration
  - cusvaer: runs the *multi-GPU, multi-node* custom Qiskit Aer GPU backend, only
    available in the cuQuantum Appliance container
  - cirq: runs Cirq's native CPU backend (cirq.Simulator)
  - cutn: runs cuTensorNet by constructing the tensor network corresponding to the
    benchmark circuit (through cuquantum.CircuitToEinsum)
  - qsim: runs qsim's CPU backend
  - qsim-cuda: runs the native qsim GPU backend
  - qsim-cusv: runs qsim's cuStateVec integration
  - qsim-mgpu: runs the *multi-GPU* (single-node) custom qsim GPU backend, only
    available in the cuQuantum Appliance container

================================================================================
"""

parser = argparse.ArgumentParser(
    description=description,
    formatter_class=RawTextAndDefaultArgFormatter)
parser.add_argument('--frontend', type=str, required=True, choices=frontend_names,
                    help=f'set the simulator frontend')
parser.add_argument('--backend', type=str, required=True, choices=backend_names,
                    help=f'set the simulator backend that is compatible with the frontend')
# TODO
#parser.add_argument('--append', help='only add to existing benchmarking data rather than overwrite any data', action='store_true')
parser.add_argument('--benchmark', type=str, default='all', choices=benchmark_names+['all'],
                    help=f'pick the circuit to benchmark')
parser.add_argument('--new', help='create a new circuit rather than use existing circuit', action='store_true')
parser.add_argument('--nqubits', type=int, help='set the number of qubits for each benchmark circuit')
parser.add_argument('--nwarmups', type=int, default=3, help='set the number of warm-up runs for each benchmark')
parser.add_argument('--nrepeats', type=int, default=10, help='set the number of repetitive runs for each benchmark')
parser.add_argument('--cachedir', type=str, default='.', help='set the directory to cache generated data')
parser.add_argument('--verbose', help='output extra information during benchmarking', action='store_true')

backend = parser.add_argument_group(
    'backend-specific options', 'each backend has its own default config, see cuquantum_benchmarks/config.py for detail')
backend.add_argument('--ngpus', type=int, help='set the number of GPUs to use')
backend.add_argument('--ncputhreads', type=int, help='set the number of CPU threads to use')
backend.add_argument('--nshots', type=int, help='set the number of shots for quantum state measurement')
backend.add_argument('--nfused', type=int, help='set the maximum number of fused qubits for gate matrix fusion')
backend.add_argument('--precision', type=str, choices=('single', 'double'),
                     help='set the floating-point precision')
backend.add_argument('--cusvaer-global-index-bits', type=str_to_seq, nargs='?', const='', default=-1,
                     help='set the global index bits to represent the inter-node network structure, refer to the cusvaer backend '
                          'documentation for further detail. If not followed by any argument, the default (empty sequence) is used; '
                          'otherwise, the argument should be a comma-separated string. '
                          'Setting this option is mandatory for the cusvaer backend and an error otherwise')
backend.add_argument('--cusvaer-p2p-device-bits', type=int, nargs='?', const=0, default=-1,
                     help='set the number of p2p device bits, refer to the cusvaer backend documentation for further detail. '
                          'If not followed by any argument, the default (0) is used. '
                          'Setting this option is mandatory for the cusvaer backend and an error otherwise')

def run():
    args = parser.parse_args()

    selected_benchmarks = benchmarks if args.benchmark == 'all' else {args.benchmark: benchmarks[args.benchmark]}
    selected_backend = (args.backend, backend_config[args.backend])

    if args.frontend == 'qiskit' and args.backend not in ('cutn', *[k for k in backends.keys() if 'aer' in k]):
        raise ValueError(f'frontend {args.frontend} does not work with backend {args.backend}')
    if args.frontend == 'cirq' and args.backend not in ('cirq', 'cutn', *[k for k in backends.keys() if k.startswith('qsim')]):
        raise ValueError(f'frontend {args.frontend} does not work with backend {args.backend}')
    if args.backend == 'cusvaer':
        if args.cusvaer_global_index_bits == -1:
            raise ValueError("backend cusvaer requires setting --cusvaer-global-index-bits")
        if args.cusvaer_p2p_device_bits == -1:
            raise ValueError("backend cusvaer requires setting --cusvaer-p2p-device-bits")
    else:
        if args.cusvaer_global_index_bits != -1:
            raise ValueError(f"cannot set --cusvaer-global-index-bits for backend {args.backend}")
        if args.cusvaer_p2p_device_bits != -1:
            raise ValueError(f"cannot set --cusvaer-p2p-device-bits for backend {args.backend}")

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler = MPHandler(sys.stdout)
    handler.setFormatter(formatter)

    level = logging.DEBUG if args.verbose else logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.addHandler(handler)

    run_interface(benchmarks=selected_benchmarks,
                  nqubits_interface=args.nqubits,
                  ngpus_interface=args.ngpus,
                  ncpu_threads_interface=args.ncputhreads,
                  frontend=args.frontend,
                  backend=selected_backend,
                  #append=args.append,
                  nwarmups=args.nwarmups,
                  nrepeats=args.nrepeats,
                  nshots_interface=args.nshots,
                  nfused_interface=args.nfused,
                  precision_interface=args.precision,
                  new_circ=args.new,
                  save=True,
                  logger=logger,
                  cache_dir=args.cachedir,
                  cusvaer_global_index_bits=args.cusvaer_global_index_bits,
                  cusvaer_p2p_device_bits=args.cusvaer_p2p_device_bits)


if __name__ == "__main__":
    run()
