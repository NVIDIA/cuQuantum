#!/bin/bash
#
# Unified launch script for cuquantum-Python tests

set -x

# The (path to) the MPI launcher.
MPIEXEC=mpirun

# The Python3 executable.
PYTHON3=python3

# The path to the Python 'samples' directory.
SAMPLES_DIR=../samples

# The path to the directory from which pytest collects "cuquantum" tests.
PYTEST_CUQUANTUM_DIR=./cuquantum_tests

# The path to the directory from which pytest collects "samples" tests.
PYTEST_SAMPLES_DIR=./samples_tests

###########################################################
# Error function adopted from the Google Shell Style Guide.
###########################################################
error() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

################################################################################
# Function to run the specified MPI samples. In case of error, the script name
#   and the error code are logged to stderr and a non-zero status corresponding
#   to the number of test failures is returned.
#
# Globals:
#   MPIEXEC
#   PYTHON3
# Arguments:
#   The number of processes to use.
#   The list of MPI samples (with path) to run.
################################################################################
run_python_mpi_samples() {
  nproc=$1
  mpi_samples="${@:2}"
  status=0
  for mpi_sample in ${mpi_samples}; do
    ${MPIEXEC} -np ${nproc} ${PYTHON3} ${mpi_sample}
    test_status=$?
    if [ ${test_status} -ne 0 ]; then
      error "Test \"${mpi_sample}\" exited with status ${test_status}."
    fi
    status=$((status+${test_status}))
  done
  return ${status}
}

STATUS=0

################################################################################
# Tests using pytest.
################################################################################

${PYTHON3} -m pytest ${PYTEST_CUQUANTUM_DIR}
test_status=$?
if [ ${test_status} -ne 0 ]; then
  error "pytest \"${PYTEST_CUQUANTUM_DIR}\" exited with status ${test_status}."
fi
STATUS=$((STATUS+${test_status}))

${PYTHON3} -m pytest -n 2 ${PYTEST_SAMPLES_DIR}
test_status=$?
if [ ${test_status} -ne 0 ]; then
  error "pytest \"${PYTEST_SAMPLES_DIR}\" exited with status ${test_status}."
fi
STATUS=$((STATUS+${test_status}))

################################################################################
# Test MPI samples.
################################################################################

# Find all the MPI sample programs.
mpi_samples=$(find ${SAMPLES_DIR} -name "*_mpi*.py")

run_python_mpi_samples 2 ${mpi_samples}
test_status=$?
if [ ${test_status} -ne 0 ]; then
  error "The MPI samples tests in \"${SAMPLES_DIR}\" exited with status ${test_status}."
fi
STATUS=$((STATUS+${test_status}))

exit ${STATUS}
