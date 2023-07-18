# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import glob
import os
import shutil
import sys
import subprocess

import pytest

from cuquantum_benchmarks.config import benchmarks


@pytest.fixture()
def visible_device(worker_id):
    """ Assign 1 device for each test workers to enable test parallelization.

    - If pytest-dist is not installed or unused (pytest -n ... is not set), just
      pass through CUDA_VISIBLE_DEVICES as is.
    - Otherwise, we assign 1 device for each worker. If there are more workers
      than devices, we round-robin.
      - In this case, CUDA_VISIBLE_DEVICES should be explicitly set, otherwise
        we just take device 0.
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if worker_id == "master":
        return visible_devices
    visible_devices = [int(i) for i in visible_devices.split(",")]
    total_devices = len(visible_devices)
    total_workers = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
    worker_id = int(worker_id.lstrip("gw"))
    if total_devices >= total_workers:
        return visible_devices[worker_id]
    else:
        # round robin + oversubscription
        return visible_devices[worker_id % total_devices]


@pytest.mark.parametrize(
    "combo", (
        # (frontend, backend, support_mpi)
        ("cirq", "cirq", False),
        ("cirq", "qsim", False),
        ("cirq", "qsim-cuda", False),
        ("cirq", "qsim-cusv", False),
        ("cirq", "qsim-mgpu", False),
        ("cirq", "cutn", True),
        ("qiskit", "aer", True),
        ("qiskit", "aer-cuda", True),
        ("qiskit", "aer-cusv", True),
        ("qiskit", "cusvaer", True),
        ("qiskit", "cutn", True),
        ("naive", "naive", False),
        ("pennylane", "pennylane", False),
        ("pennylane", "pennylane-lightning-gpu", False),
        ("pennylane", "pennylane-lightning-qubit", False),
        ("pennylane", "pennylane-lightning-kokkos", False),
        ("qulacs", "qulacs-cpu", False),
        ("qulacs", "qulacs-gpu", False),
    )
)
@pytest.mark.parametrize(
    "nqubits", (4,)
)
@pytest.mark.parametrize(
    "benchmark", tuple(benchmarks.keys())
)
@pytest.mark.parametrize(
    "precision", ("single", "double")
)
class TestCmdCircuit:

    # TODO: perhaps this function should live in the _utils module...?
    def _skip_if_unavailable(self, combo, nqubits, benchmark, precision):
        frontend, backend, support_mpi = combo

        # check frontend exists
        if frontend == "cirq":
            try:
                import cirq
            except ImportError:
                pytest.skip("cirq not available")
        elif frontend == "qiskit":
            try:
                import qiskit
            except ImportError:
                pytest.skip("qiskit not available")
        elif frontend == "naive":
            from cuquantum_benchmarks.frontends import frontends
            if "naive" not in frontends:
                pytest.skip("naive not available")
        elif frontend == "pennylane":
            try:
                import pennylane
            except ImportError:
                pytest.skip("pennylane not available")
        elif frontend == "qulacs":
            try:
                import qulacs
            except ImportError:
                pytest.skip("qulacs not available")

        # check backend exists
        if backend == "aer-cuda":
            skip = False
            try:
                from qiskit.providers.aer import AerSimulator
            except ImportError:
                skip = True
            else:
                # there is no other way :(
                s = AerSimulator()
                if 'GPU' not in s.available_devices():
                    skip = True
            if skip:
                pytest.skip("aer-cuda not available")
        elif backend in ("aer-cusv", "cusvaer"):
            # no way to tell if the Aer-cuStateVec integration is built, so only
            # check if we're inside the container, we're being conservative here...
            try:
                import cusvaer
            except ImportError:
                pytest.skip(f"{backend} not available")
        elif backend == "aer":
            try:
                from qiskit.providers.aer import AerSimulator
            except ImportError:
                pytest.skip("aer not available")
        elif backend == "qsim-cuda":
            from qsimcirq import qsim_gpu
            if qsim_gpu is None:
                pytest.skip("qsim-cuda not available")
        elif backend == "qsim-cusv":
            from qsimcirq import qsim_custatevec
            if qsim_custatevec is None:
                pytest.skip("qsim-cusv not available")
        elif backend == "qsim-mgpu":
            try:
                from qsimcirq import qsim_mgpu
            except ImportError:
                pytest.skip("qsim-mgpu not available")
        elif backend == "pennylane":
            try:
                import pennylane
            except ImportError:
                pytest.skip("pennylane not available")
        elif backend == "pennylane-lightning-gpu":
            try:
                from pennylane_lightning_gpu import LightningGPU
            except ImportError:
                pytest.skip("pennylane-lightning-gpu not available")
        elif backend == "pennylane-lightning-qubit":
            try:
                from pennylane_lightning.lightning_qubit import LightningQubit
            except ImportError:
                pytest.skip("pennylane-lightning-qubit not available")
        elif backend == "pennylane-lightning-kokkos":
            try:
                import pennylane_lightning_kokkos
            except ImportError:
                pytest.skip("pennylane-lightning-kokkos not available")
        elif backend == "qulacs-cpu":
            try:
                import qulacs
            except ImportError:
                pytest.skip(f"{backend} not available")
        elif backend == "qulacs-gpu":
            try:
                import qulacs.QuantumStateGpu
            except ImportError:
                pytest.skip(f"{backend} not available")

        # check MPI exists
        if support_mpi:
            if shutil.which('mpiexec') is None:
                pytest.skip('MPI not available')
            if backend == 'cutn' and os.environ.get('CUTENSORNET_COMM_LIB') is None:
                pytest.skip('CUTENSORNET_COMM_LIB is not set')

        if ((backend == 'cirq' or backend.startswith('qsim'))
                and precision == 'double'):
            return pytest.raises(subprocess.CalledProcessError), True
        if backend.startswith('qulacs') and precision == 'single':
            return pytest.raises(subprocess.CalledProcessError), True

        return contextlib.nullcontext(), False

    def test_benchmark(self, combo, nqubits, benchmark, precision, tmp_path, visible_device):
        frontend, backend, support_mpi = combo
        ctx, ret = self._skip_if_unavailable(combo, nqubits, benchmark, precision)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(visible_device)

        # internal loop: run the same test twice, without and with MPI
        if support_mpi:
            # TODO: this may not be robust against conda-forge Open MPI, need to turn
            # on MCA parameters via env var
            tests = ([], ['mpiexec', '-n', '1'])
        else:
            tests = ([], )

        # use default value from config.py for --ngpus
        cmd = [sys.executable, '-m', 'cuquantum_benchmarks', 'circuit',
                '--frontend', frontend,
                '--backend', backend,
                '--ncputhreads', '1',
                '--nqubits', str(nqubits),
                '--benchmark', benchmark,
                '--precision', precision,
                '--cachedir', str(tmp_path),
                # speed up the tests...
                '--nwarmups', '1',
                '--nrepeats', '1',
                '--verbose']
        if backend == 'cusvaer':
            cmd += ['--cusvaer-global-index-bits', '--cusvaer-p2p-device-bits']
        if backend == 'cutn':
            cmd += ['--nhypersamples', '2']

        for cmd_prefix in tests:
            result = subprocess.run(cmd_prefix+cmd, env=env, capture_output=True)

            with ctx:
                try:
                    assert bool(result.check_returncode()) == ret
                    cached_circuits = [f for f in glob.glob(str(tmp_path / f"circuits/{benchmark}_{nqubits}*.pickle")) if os.path.isfile(f)]
                    assert len(cached_circuits) == 1
                    cached_json = [f for f in glob.glob(str(tmp_path / f"data/{benchmark}.json")) if os.path.isfile(f)]
                    assert len(cached_json) == 1  # TODO: test aggregate behavior too?
                except:
                    # make debugging easier
                    print("stdout:\n", result.stdout.decode())
                    print("stderr:\n", result.stderr.decode())
                    raise
                finally:
                    print("cmd:\n", ' '.join(cmd_prefix+cmd))


# TODO: test invalid cases and ensure we raise errors
class TestCmdApi:

    @pytest.mark.parametrize(
        "args", (
            ["--nqubits", "4", "--ntargets", "2"],
            ["--nqubits", "4", "--targets", "2,3"],
            ["--nqubits", "6", "--ntargets", "3", "--controls", "3"],
            ["--nqubits", "4", "--targets", "2,3", "--ncontrols", "1"],
            ["--nqubits", "4", "--targets", "2,3", "--controls", "1"],
        )
    )
    @pytest.mark.parametrize(
        "matrix_prop", (
            [],  # default
            ["--layout", "column", "--adjoint"],
        )
    )
    @pytest.mark.parametrize(
        "precision", ("single", "double")
    )
    @pytest.mark.parametrize(
        "flush", (True, False)
    )
    def test_apply_matrix(self, args, matrix_prop, precision, flush, tmp_path, visible_device):
        benchmark = 'apply_matrix'
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(visible_device)

        cmd = [sys.executable, '-m', 'cuquantum_benchmarks', 'api',
                '--benchmark', benchmark,
                '--precision', precision,
                '--cachedir', str(tmp_path),
                # speed up the tests...
                '--nwarmups', '1',
                '--nrepeats', '1',
                '--verbose']
        cmd += args
        cmd += matrix_prop
        if flush:
            cmd += ['--flush-cache']

        result = subprocess.run(cmd, env=env, capture_output=True)

        try:
            assert bool(result.check_returncode()) == False
            cached_json = [f for f in glob.glob(str(tmp_path / f"data/{benchmark}.json")) if os.path.isfile(f)]
            assert len(cached_json) == 1  # TODO: test aggregate behavior too?
        except:
            # make debugging easier
            print("stdout:\n", result.stdout.decode())
            print("stderr:\n", result.stderr.decode())
            raise
        finally:
            print("cmd:\n", ' '.join(cmd))

    @pytest.mark.parametrize(
        "args", (
            ("--nqubits", "4", "--ntargets", "2",),
            ("--nqubits", "4", "--targets", "2,3",),
            ("--nqubits", "6", "--ntargets", "2", "--controls", "3",),
            ("--nqubits", "4", "--targets", "1,2", "--ncontrols", "1",),
            ("--nqubits", "4", "--targets", "2,3", "--controls", "1",),
        )
    )
    @pytest.mark.parametrize(
        "diag", (
            (),
            ("--has-diag", "--location-diag", "device",),
            ("--has-diag", "--precision-diag", "double", "--precision", "double",),
        )
    )
    @pytest.mark.parametrize(
        "perm", (
            ("--has-perm",),
            ("--has-perm", "--location-perm", "device",),
            ("--perm-table", "2,3,0,1",),  # this test assumes ntargets=2 always
        )
    )
    @pytest.mark.parametrize(
        "matrix_prop", (
            (),  # default
            ("--adjoint",),
        )
    )
    def test_apply_generalized_permutation_matrix(
            self, args, diag, perm, matrix_prop, tmp_path, visible_device):
        benchmark = 'apply_generalized_permutation_matrix'
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(visible_device)

        cmd = [sys.executable, '-m', 'cuquantum_benchmarks', 'api',
               '--benchmark', benchmark,
               '--cachedir', str(tmp_path),
               # speed up the tests...
               '--nwarmups', '1',
               '--nrepeats', '1',
               '--verbose']
        cmd += args
        cmd += diag
        cmd += perm
        cmd += matrix_prop
        result = subprocess.run(cmd, env=env, capture_output=True)

        try:
            assert bool(result.check_returncode()) == False
            cached_json = [f for f in glob.glob(str(tmp_path / f"data/{benchmark}.json")) if os.path.isfile(f)]
            assert len(cached_json) == 1  # TODO: test aggregate behavior too?
        except:
            # make debugging easier
            print("stdout:\n", result.stdout.decode())
            print("stderr:\n", result.stderr.decode())
            raise
        finally:
            print("cmd:\n", ' '.join(cmd))

    @pytest.mark.parametrize(
        "args", (
            ("--nqubits", "4", "--nbit-ordering", "2", "--nshots", "256"),
            ("--nqubits", "4", "--bit-ordering", "2,3", "--output-order", "random"),
        )
    )
    @pytest.mark.parametrize(
        "precision", ("single", "double")
    )
    def test_cusv_sampler(self, args, precision, tmp_path, visible_device):
        benchmark = 'cusv_sampler'
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(visible_device)

        cmd = [sys.executable, '-m', 'cuquantum_benchmarks', 'api',
                '--benchmark', benchmark,
                '--precision', precision,
                '--cachedir', str(tmp_path),
                # speed up the tests...
                '--nwarmups', '1',
                '--nrepeats', '1',
                '--verbose']
        cmd += args
        result = subprocess.run(cmd, env=env, capture_output=True)

        try:
            assert bool(result.check_returncode()) == False
            cached_json = [f for f in glob.glob(str(tmp_path / f"data/{benchmark}.json")) if os.path.isfile(f)]
            assert len(cached_json) == 1  # TODO: test aggregate behavior too?
        except:
            # make debugging easier
            print("stdout:\n", result.stdout.decode())
            print("stderr:\n", result.stderr.decode())
            raise
        finally:
            print("cmd:\n", ' '.join(cmd))

    @pytest.mark.parametrize(
        "args", (
            ["--expr", "abc->abx,xc", "--shape", "4,8,4"],
            ["--expr", "abcd->ax,bcdx", "--shape", "4,8,4,2"],
        )
    )
    @pytest.mark.parametrize(
        "method", (
            ("--method", "QR",),
            ("--method", "SVD",),
            ("--algorithm", "gesvd"),
            ("--algorithm", "gesvdj"),
            ("--algorithm", "gesvdr"),
            ("--algorithm", "gesvdp"),
        )
    )
    @pytest.mark.parametrize(
        "precision", ("single", "double")
    )
    @pytest.mark.parametrize(
        "is_complex", (True, False)
    )
    def test_tensor_decompose(self, args, method, precision, is_complex, tmp_path, visible_device):
        benchmark = 'tensor_decompose'
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(visible_device)

        cmd = [sys.executable, '-m', 'cuquantum_benchmarks', 'api',
                '--benchmark', benchmark,
                '--precision', precision,
                '--cachedir', str(tmp_path),
                # speed up the tests...
                '--nwarmups', '1',
                '--nrepeats', '1',
                '--verbose']
        cmd += args
        cmd += method
        if is_complex:
            cmd.append('--is-complex')

        result = subprocess.run(cmd, env=env, capture_output=True)

        try:
            assert bool(result.check_returncode()) == False
            cached_json = [f for f in glob.glob(str(tmp_path / f"data/{benchmark}.json")) if os.path.isfile(f)]
            assert len(cached_json) == 1  # TODO: test aggregate behavior too?
        except:
            # make debugging easier
            print("stdout:\n", result.stdout.decode())
            print("stderr:\n", result.stderr.decode())
            raise
        finally:
            print("cmd:\n", ' '.join(cmd))
