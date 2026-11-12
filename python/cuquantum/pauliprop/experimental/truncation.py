# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

__all__ = ["Truncation"]


@dataclass
class Truncation:
    """
    Truncation strategy for Pauli expansion operations.
    
    Truncation can be applied during gate application or as a standalone operation
    to reduce the number of terms in a Pauli expansion. Multiple truncation criteria
    can be specified together; terms are removed if they fail any criterion.

    In multi-process execution, every process must use an equivalent truncation
    strategy. Truncation is applied independently to each process's local
    terms.
    
    Args:
        pauli_weight_cutoff: Maximum Pauli weight (number of non-identity Paulis) to keep.
            Terms with Pauli weight (non-identity Pauli operators) greater than this value are removed. If None, no weight-based
            truncation is applied.
        pauli_coeff_cutoff: Coefficient magnitude at or below which terms are removed: terms with
            ``|coefficient| <= pauli_coeff_cutoff`` are removed. If None, no coefficient-based
            truncation is applied.
    
    Example:
        >>> # Keep only terms with weight <= 3 and |coef| > 1e-10
        >>> truncation = Truncation(pauli_weight_cutoff=3, pauli_coeff_cutoff=1e-10)
        >>> result = expansion.apply_gate(gate, truncation=truncation)
        
        >>> # Only weight-based truncation
        >>> truncation = Truncation(pauli_weight_cutoff=5)
        
        >>> # Only coefficient-based truncation
        >>> truncation = Truncation(pauli_coeff_cutoff=1e-8)
    """
    pauli_weight_cutoff: int | None = None
    pauli_coeff_cutoff: float | None = None


