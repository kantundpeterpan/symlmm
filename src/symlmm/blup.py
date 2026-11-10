"""BLUP moment expressions."""

from __future__ import annotations

from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import FormArityError, UnsupportedForm
from .forms import DualForm, Form, require_form


def _inverse(matrix: Any) -> Any:
    """Return the inverse of an explicit matrix or matrix expression."""
    if isinstance(matrix, sp.MatrixBase):
        return matrix.inv()
    return matrix.inverse()


def blup_mean(model: Any, form: str | Form | None = None) -> Any:
    """Return E[b | y] = G Z' V^-1 (y - X beta)."""
    form = require_form(form)
    if form is Form.SUM:
        raise UnsupportedForm("BLUP mean has no general sum form")
    n, q = model.dimensions()
    dense = getattr(model, "_dense", False)
    G = model.matrix_g() if dense else sp.MatrixSymbol("G", q, q)
    Z = model.matrix_z() if dense else sp.MatrixSymbol("Z", n, q)
    V = model.matrix_covariance()
    y = sp.MatrixSymbol("y", n, 1)
    expression = G * Z.T * _inverse(V) * (y - model.matrix_mean())
    if form is Form.MATRIX:
        return expression
    return DualForm(sum=None, matrix=expression)


def blup_covariance(model: Any, form: str | Form | None = None) -> Any:
    """Return Cov(b | y) = (G^-1 + Z' R^-1 Z)^-1."""
    form = require_form(form)
    if form is Form.SUM:
        raise UnsupportedForm("BLUP covariance has no general sum form")
    n, q = model.dimensions()
    dense = getattr(model, "_dense", False)
    G = model.matrix_g() if dense else sp.MatrixSymbol("G", q, q)
    Z = model.matrix_z() if dense else sp.MatrixSymbol("Z", n, q)
    R = model.matrix_r() if dense else sp.MatrixSymbol("R", n, n)
    expression = _inverse(_inverse(G) + Z.T * _inverse(R) * Z)
    if form is Form.MATRIX:
        return expression
    return DualForm(sum=None, matrix=expression)


def blup_cross_covariance(
    model: Any,
    i: Any | None = None,
    form: str | Form | None = None,
) -> Any:
    """Return Cov(y_i, b) = Z[i, :] G."""
    form = require_form(form)
    if i is None:
        raise FormArityError("blup_cross_covariance requires an index")
    if form is Form.SUM:
        raise UnsupportedForm("BLUP cross-covariance has no general sum form")
    n, q = model.dimensions()
    dense = getattr(model, "_dense", False)
    G = model.matrix_g() if dense else sp.MatrixSymbol("G", q, q)
    Z = model.matrix_z() if dense else sp.MatrixSymbol("Z", n, q)
    expression = Z[i, :] * G
    if form is Form.MATRIX:
        return expression
    return DualForm(sum=None, matrix=expression)
