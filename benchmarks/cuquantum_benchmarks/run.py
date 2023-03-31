# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
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
from .run_interface import run_interface, BenchApiRunner
from ._utils import str_to_seq, MPHandler, RawTextAndDefaultArgFormatter


frontend_names = [f for f in frontends.keys()]
backend_names = [b for b in backends.keys()]
benchmark_names = [b for b in benchmarks.keys()]


main_description = api_description = r"""
=============== NVIDIA cuQuantum Performance Benchmark Suite ===============
"""


circuit_description = r"""
=============== NVIDIA cuQuantum Performance Benchmark Suite ===============

Supported Frontends:

  - cirq
  - qiskit
  - pennylane
  - qulacs

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
  - pennylane: runs PennyLane's native CPU backend
  - pennylane-lightning-gpu: runs the PennyLane-Lightning GPU backend 
  - pennylane-lightning-qubit: runs the PennyLane-Lightning CPU backend
  - pennylane-lightning-kokkos: runs the PennyLane-Lightning Kokkos backend
  - qulacs-gpu: runs the Qulacs GPU backend
  - qulacs-cpu: runs the Qulacs CPU backend

Note: all frontends and backends are optional and unavailable for use unless installed.

============================================================================
"""


# main parser
parser = argparse.ArgumentParser(
    description=main_description,
    formatter_class=RawTextAndDefaultArgFormatter)
subparsers = parser.add_subparsers(dest="cmd", required=True)


# "cuquantum-benchmarks circuit" subcommand
parser_circuit = subparsers.add_parser(
    'circuit',
    description=circuit_description,
    help="benchmark different classes of quantum circuits",
    formatter_class=RawTextAndDefaultArgFormatter)
parser_circuit.add_argument('--benchmark', type=str, default='all', choices=benchmark_names+['all'],
                            help=f'pick the circuit to benchmark')
parser_circuit.add_argument('--frontend', type=str, required=True, choices=frontend_names,
                            help=f'set the simulator frontend')
parser_circuit.add_argument('--backend', type=str, required=True, choices=backend_names,
                            help=f'set the simulator backend that is compatible with the frontend')
# TODO
#parser.add_argument('--append', help='only add to existing benchmarking data rather than overwrite any data', action='store_true')
parser_circuit.add_argument('--new', help='create a new circuit rather than use existing circuit', action='store_true')
# these options make sense to both circuit & api benchmarks, for better UX we need to copy/paste
parser_circuit.add_argument('--cachedir', type=str, default='.', help='set the directory to cache generated data')
parser_circuit.add_argument('--nqubits', type=int, help='set the number of qubits for each benchmark (circuit/api)')
parser_circuit.add_argument('--nwarmups', type=int, default=3, help='set the number of warm-up runs for each benchmark')
parser_circuit.add_argument('--nrepeats', type=int, default=10, help='set the number of repetitive runs for each benchmark')
parser_circuit.add_argument('-v', '--verbose', help='output extra information during benchmarking', action='store_true')

backend = parser_circuit.add_argument_group(
    'backend-specific options', 'each backend has its own default config, see cuquantum_benchmarks/config.py for detail')
backend.add_argument('--ngpus', type=int, help='set the number of GPUs to use')
backend.add_argument('--ncputhreads', type=int, help='set the number of CPU threads to use')
backend.add_argument('--nshots', type=int, help='set the number of shots for quantum state measurement')
backend.add_argument('--nfused', type=int, help='set the maximum number of fused qubits for gate matrix fusion')
backend.add_argument('--precision', type=str, choices=('single', 'double'),
                     help='set the floating-point precision')
backend.add_argument('--cusvaer-global-index-bits', type=str_to_seq, nargs='?', const='', default=-1,
                     help='set the global index bits to specify the inter-node network structure.  Please refer to the '
                          'cusvaer backend documentation for further details. If not followed by any argument, '
                          'the default (empty sequence) is used; '
                          'otherwise, the argument should be a comma-separated string. '
                          'Setting this option is mandatory for the cusvaer backend and an error otherwise')
backend.add_argument('--cusvaer-p2p-device-bits', type=int, nargs='?', const=0, default=-1,
                     help='set the number of p2p device bits.  Please refer to the cusvaer backend documentation '
                          'for further details. If not followed by any argument, the default (0) is used. '
                          'Setting this option is mandatory for the cusvaer backend and an error otherwise')
backend.add_argument('--cusvaer-data-transfer-buffer-bits', type=int, default=26,
                     help='set the size of the data transfer buffer in cusvaer.  The size is '
                          'specified as a positive integer.  The buffer sized used is (1 << [#bits]). '
                          'The default is 26 (64 MiB = 1 << 26)')
backend.add_argument('--cusvaer-comm-plugin-type', type=str, nargs='?', default='mpi_auto',
                     choices=['mpi_auto', 'mpi_openmpi', 'mpi_mpich', 'external', 'self'],
                     help='set the type of comm plugin used for multi-process simulation. '
                          'Required to set this option when one needs to use a custom comm plugin. '
                          'Acceptable values are mpi_auto, mpi_openmpi, mpi_mpich and external. '
                          'The default is mpi_auto.')
backend.add_argument('--cusvaer-comm-plugin-soname', type=str, nargs='?', default='',
                     help='specify the name of a shared library used for inter-process communication. '
                          'Required to set this option when one needs to use a custom comm plugin')


# "cuquantum-benchmarks api" subcommand
parser_api = subparsers.add_parser(
    'api',
    description=api_description,
    help="benchmark different APIs from cuQuantum's libraries",
    formatter_class=RawTextAndDefaultArgFormatter)
parser_api.add_argument('--benchmark', type=str, choices=('apply_matrix',),
                        help=f'pick the API to benchmark')
parser_api.add_argument('--precision', type=str, choices=('single', 'double'), default='single',
                        help='set the floating-point precision')
apply_matrix = parser_api.add_argument_group('apply_matrix-specific options')

targets = apply_matrix.add_mutually_exclusive_group(required=True)
targets.add_argument('--targets', type=str_to_seq,
                     help="set the (comma-separated) target qubit IDs")
targets.add_argument('--ntargets', type=int, help='set the number of target qubits')

controls = apply_matrix.add_mutually_exclusive_group(required=False)
controls.add_argument('--controls', type=str_to_seq,
                      help="set the (comma-separated) control qubit IDs")
controls.add_argument('--ncontrols', type=int, help='set the number of target qubits')

apply_matrix.add_argument('--layout', type=str, choices=('row', 'column'), default='row',
                          help='set the gate matrix layout')
apply_matrix.add_argument('--adjoint', action='store_true', help='apply the matrix adjoint')
apply_matrix.add_argument('--location', type=str, choices=('device', 'host'), default='host',
                          help='set the location of the gate matrix')
apply_matrix.add_argument('--nqubits', type=int, required=True,
                          help='set the total number of qubits')
apply_matrix.add_argument('--flush-cache', action='store_true', help='flush the L2 cache for more accurate timing')

# these options make sense to both circuit & api benchmarks, for better UX we need to copy/paste
parser_api.add_argument('--cachedir', type=str, default='.', help='set the directory to cache generated data')
parser_api.add_argument('--nwarmups', type=int, default=3, help='set the number of warm-up runs for each benchmark')
parser_api.add_argument('--nrepeats', type=int, default=10, help='set the number of repetitive runs for each benchmark')
parser_api.add_argument('-v', '--verbose', help='output extra information during benchmarking', action='store_true')


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)

# WAR: PennyLane mistakenly sets a stream handler to the root logger, so if PennyLane is
# installed, all of our logging is messed up. Let's just clean up the root logger. It's
# reported to & fixed in upstream (https://github.com/PennyLaneAI/pennylane/issues/3731).
root_logger = logging.getLogger()
for h in root_logger.handlers:
    h.close()
root_logger.handlers = []  # this private interface has been stable since 2002


def run(args=None):
    # we allow args to be a list of cmd options for potential private use cases and tests
    args = parser.parse_args(args)
    #print(args)

    # Since run() might be called multiple times, in such case we don't wanna make any changes
    # to the handler in the 2nd time onward, this ensures we write to the same I/O stream and
    # do not need to call hanlder.flush() manually.
    if not logger.hasHandlers():
        formatter = logging.Formatter(f"%(asctime)s %(levelname)-8s %(message)s")
        handler = MPHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        level = logging.DEBUG if args.verbose else logging.INFO
    except AttributeError:
        # user forgot to set the subcommand, let argparse kick in and raise
        pass
    else:
        logger.setLevel(level)

    # dispatch to subcommands
    cmd = args.cmd
    if cmd == "circuit":

        selected_benchmarks = benchmarks if args.benchmark == 'all' else {args.benchmark: benchmarks[args.benchmark]}
        selected_backend = (args.backend, backend_config[args.backend])

        if ((args.frontend == 'cirq' and args.backend not in ('cirq', 'cutn', *[k for k in backends.keys() if k.startswith('qsim')]))
                or (args.frontend == 'qiskit' and args.backend not in ('cutn', *[k for k in backends.keys() if 'aer' in k]))
                or (args.frontend == 'naive' and args.backend != 'naive')
                or (args.frontend == 'pennylane' and not args.backend.startswith('pennylane'))
                or (args.frontend == 'qulacs' and not args.backend.startswith('qulacs'))):
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
                      cache_dir=args.cachedir,
                      cusvaer_global_index_bits=args.cusvaer_global_index_bits,
                      cusvaer_p2p_device_bits=args.cusvaer_p2p_device_bits,
                      cusvaer_data_transfer_buffer_bits=args.cusvaer_data_transfer_buffer_bits,
                      cusvaer_comm_plugin_type=args.cusvaer_comm_plugin_type,
                      cusvaer_comm_plugin_soname=args.cusvaer_comm_plugin_soname)


    elif cmd == "api":
        del args.cmd
        runner = BenchApiRunner(**vars(args))

        # benchmark & dump result to cachedir
        runner.run()


if __name__ == "__main__":
    run()
