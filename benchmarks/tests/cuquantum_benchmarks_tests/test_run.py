import contextlib
import glob
import os
import shutil
import sys
import subprocess

import pytest

from cuquantum_benchmarks.config import benchmarks


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
class TestCmdline:

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
                pytest.skip(f"{backend} maybe not available")
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

        # check MPI exists
        if support_mpi:
            if shutil.which('mpiexec') is None:
                pytest.skip('MPI not available')
            if backend == 'cutn' and os.environ.get('CUTENSORNET_COMM_LIB') is None:
                pytest.skip('CUTENSORNET_COMM_LIB is not set')

        if ((backend == 'cirq' or backend.startswith('qsim'))
                and precision == 'double'):
            return pytest.raises(subprocess.CalledProcessError), True

        return contextlib.nullcontext(), False

    def test_benchmark(self, combo, nqubits, benchmark, precision, tmp_path):
        frontend, backend, support_mpi = combo
        ctx, ret = self._skip_if_unavailable(combo, nqubits, benchmark, precision)

        # internal loop: run the same test twice, without and with MPI
        if support_mpi:
            # TODO: this may not be robust against conda-forge Open MPI, need to turn
            # on MCA parameters via env var
            tests = ([], ['mpiexec', '-n', '1'])
        else:
            tests = ([], )

        # use default value from config.py for --ngpus
        cmd = [sys.executable, '-m', 'cuquantum_benchmarks',
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

        for cmd_prefix in tests:
            result = subprocess.run(cmd_prefix+cmd, env=os.environ, capture_output=True)

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
