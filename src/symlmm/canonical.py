"""Canonical group/occasion indexing helpers."""

from __future__ import annotations

from typing import Any

import sympy as sp  # type: ignore[import-untyped]


def group_size_symbol(group: Any) -> sp.Symbol:
    """Return the symbolic size of `group`."""
    return sp.Symbol(f"n_{group}", positive=True, integer=True)


def group_mean(group: Any, size: Any, p: int) -> sp.MatrixSymbol:
    """Return the fixed design matrix X_group."""
    return sp.MatrixSymbol(f"X_{group}", size, p)


def group_random_design(group: Any, size: Any, q: int) -> sp.MatrixSymbol:
    """Return the random design matrix Z_group."""
    return sp.MatrixSymbol(f"Z_{group}", size, q)


def group_residual(group: Any, size: Any) -> sp.MatrixSymbol:
    """Return the residual covariance matrix R_group."""
    return sp.MatrixSymbol(f"R_{group}", size, size)


def group_coefficients(group: Any, q: int) -> sp.MatrixSymbol:
    """Return the random-effect vector b_group."""
    return sp.MatrixSymbol(f"b_{group}", q, 1)
