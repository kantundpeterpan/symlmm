"""Structured design columns, residual precision, and Woodbury reductions."""

from __future__ import annotations

from typing import Any

import sympy as sp  # type: ignore[import-untyped]


def as_scalar(expr: Any) -> Any:
    """Return the scalar inside a 1x1 matrix, otherwise `expr`."""
    if isinstance(expr, (sp.MatrixBase, sp.MatrixExpr)) and expr.shape == (1, 1):
        return expr[0, 0]
    return expr


def inverse_matrix(matrix: Any) -> Any:
    """Return an inverse for explicit matrices, matrix expressions, and scalars."""
    if isinstance(matrix, sp.MatrixBase):
        return matrix.inv()
    if isinstance(matrix, sp.MatrixExpr):
        return matrix.inverse()
    return 1 / sp.sympify(matrix)


def woodbury_reduced_inverse(G: Any, Z: Any, R_inv: Any) -> Any:
    """Return (G^-1 + Z' R^-1 Z)^-1 without an n x n inverse."""
    gram = Z.T * R_inv * Z
    if isinstance(gram, (sp.MatrixBase, sp.MatrixExpr)) and gram.shape == (1, 1):
        gram = gram[0, 0]
    total = as_scalar(inverse_matrix(G)) + gram
    if isinstance(total, (sp.MatrixBase, sp.MatrixExpr)):
        return inverse_matrix(total)
    return 1 / total


def woodbury_shrinkage(G: Any, Z: Any, R_inv: Any) -> Any:
    """Return G Z' (R + Z G Z')^-1."""
    return woodbury_reduced_inverse(G, Z, R_inv) * Z.T * R_inv


def woodbury_covariance(G: Any, Z: Any, R_inv: Any) -> Any:
    """Return (G^-1 + Z' R^-1 Z)^-1."""
    return woodbury_reduced_inverse(G, Z, R_inv)


def woodbury_precision(G: Any, Z: Any, R_inv: Any) -> Any:
    """Return (R + Z G Z')^-1 using Woodbury."""
    reduced = woodbury_reduced_inverse(G, Z, R_inv)
    return R_inv - R_inv * Z * reduced * Z.T * R_inv
