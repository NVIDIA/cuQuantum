#!/usr/bin/env bash
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# Generate pyproject.toml from pyproject.toml.template by detecting
# CUDA version and substituting template placeholders.
#
# Usage:
#   CUDA_PATH=/usr/local/cuda bash configure.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/pyproject.toml.template"
OUTPUT="${SCRIPT_DIR}/pyproject.toml"

# -------------------------------------------------------------------
# 1. Validate template exists (early, before doing any real work)
# -------------------------------------------------------------------
if [[ ! -f "${TEMPLATE}" ]]; then
    echo "ERROR: Template not found at ${TEMPLATE}" >&2
    exit 1
fi

# -------------------------------------------------------------------
# 2. Validate CUDA_PATH
# -------------------------------------------------------------------
if [[ -z "${CUDA_PATH:-}" ]]; then
    echo "ERROR: CUDA_PATH is not set. Please set it to your CUDA toolkit root." >&2
    exit 1
fi

CUDA_H="${CUDA_PATH}/include/cuda.h"
if [[ ! -f "${CUDA_H}" ]]; then
    echo "ERROR: Cannot find ${CUDA_H}. Is CUDA_PATH set correctly?" >&2
    exit 1
fi

# -------------------------------------------------------------------
# 3. Parse CUDA_VERSION from cuda.h  (mirrors setup.py logic)
#    Example line:  #define CUDA_VERSION 12020
#    12020 => major = 12020 / 1000 = 12
# -------------------------------------------------------------------
# Read grep output into a variable to avoid a grep|head|sed pipeline,
# which can behave inconsistently with pipefail across bash versions
# (head closes the pipe early, causing grep to receive SIGPIPE).
CUDA_H_MATCH=$(grep -E -m1 '^\s*#define\s+CUDA_VERSION\s+[0-9]+' "${CUDA_H}" || true)

if [[ -z "${CUDA_H_MATCH}" ]]; then
    echo "ERROR: Could not find CUDA_VERSION definition in ${CUDA_H}" >&2
    exit 1
fi

CUDA_VERSION_RAW=$(echo "${CUDA_H_MATCH}" | sed -E 's/.*#define\s+CUDA_VERSION\s+([0-9]+).*/\1/')

if [[ -z "${CUDA_VERSION_RAW}" ]]; then
    echo "ERROR: Could not parse CUDA_VERSION from ${CUDA_H}" >&2
    exit 1
fi

if (( CUDA_VERSION_RAW < 1000 )); then
    echo "ERROR: CUDA_VERSION ${CUDA_VERSION_RAW} from ${CUDA_H} is unexpectedly small (< 1000)" >&2
    exit 1
fi

CUDA_MAJOR=$(( CUDA_VERSION_RAW / 1000 ))

echo "Detected CUDA_VERSION=${CUDA_VERSION_RAW} (major=${CUDA_MAJOR}) from ${CUDA_H}"

# -------------------------------------------------------------------
# 4. Map CUDA major version to template variable values
# -------------------------------------------------------------------
case "${CUDA_MAJOR}" in
    12)
        JAX_VERSION_SPEC=">=0.5,<0.7"
        CUDA_CLASSIFIER="Environment :: GPU :: NVIDIA CUDA :: 12"
        ;;
    13)
        JAX_VERSION_SPEC=">=0.8,<0.9"
        CUDA_CLASSIFIER="Environment :: GPU :: NVIDIA CUDA :: 13"
        ;;
    *)
        echo "ERROR: Unsupported CUDA major version: ${CUDA_MAJOR}" >&2
        exit 1
        ;;
esac

# -------------------------------------------------------------------
# 5. Escape sed replacement-special characters (& and \) in values
# -------------------------------------------------------------------
sed_escape() {
    printf '%s' "$1" | sed -e 's/[&\\/]/\\&/g'
}

CUDA_MAJOR_ESC=$(sed_escape "${CUDA_MAJOR}")
JAX_VERSION_SPEC_ESC=$(sed_escape "${JAX_VERSION_SPEC}")
CUDA_CLASSIFIER_ESC=$(sed_escape "${CUDA_CLASSIFIER}")

# -------------------------------------------------------------------
# 6. Generate pyproject.toml via sed substitution
#    Write to a temp file then atomically move into place so a partial
#    write (e.g. Ctrl+C, disk full) never leaves a broken pyproject.toml.
# -------------------------------------------------------------------
TMPFILE=$(mktemp "${OUTPUT}.XXXXXX")
trap 'rm -f "${TMPFILE}"' EXIT

sed -e "s|@CUDA_MAJOR_VER@|${CUDA_MAJOR_ESC}|g" \
    -e "s|@JAX_VERSION_SPEC@|${JAX_VERSION_SPEC_ESC}|g" \
    -e "s|@CUDA_CLASSIFIER@|${CUDA_CLASSIFIER_ESC}|g" \
    -- "${TEMPLATE}" > "${TMPFILE}"

# Skip overwrite if output already exists and is identical (idempotency).
if [[ -f "${OUTPUT}" ]] && cmp -s "${TMPFILE}" "${OUTPUT}"; then
    echo "pyproject.toml is already up to date for CUDA ${CUDA_MAJOR}"
    exit 0
fi

mv -f "${TMPFILE}" "${OUTPUT}"
trap - EXIT

echo "Generated ${OUTPUT} for CUDA ${CUDA_MAJOR}"
