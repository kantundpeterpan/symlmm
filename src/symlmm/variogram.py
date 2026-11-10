"""Variogram and covariogram queries."""

from __future__ import annotations

from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import InvalidKindError
from .forms import DualForm, Form, require_form
from .moments import covariance as _covariance

_VALID_KINDS = frozenset({"raw", "residual", "conditional"})


def _residual_variogram(model: Any, i: Any, j: Any, within: str | None = None) -> Any:
    """Return the mean-adjusted variogram entry."""
    return (
        model.covariance_entry(i, i, within)
        + model.covariance_entry(j, j, within)
        - 2 * model.covariance_entry(i, j, within)
    ) / 2


def _conditional_variogram(model: Any, i: Any, j: Any, within: str | None = None) -> Any:
    """Return the conditional variogram entry based on the residual covariance."""
    value = (
        model.residual.entry(i, i, model)
        + model.residual.entry(j, j, model)
        - 2 * model.residual.entry(i, j, model)
    ) / 2
    if within is not None and not model._dense:
        value = model._apply_within(value, i, j, within)
    return value


def covariogram(
    model: Any,
    i: Any,
    j: Any,
    form: str | Form | None = None,
    within: str | None = None,
) -> Any:
    """Return Cov(y_i, y_j); alias of ``moments.covariance``."""
    return _covariance(model, i, j, form, within)


def variogram(
    model: Any,
    i: Any,
    j: Any,
    kind: str,
    form: str | Form | None = None,
    within: str | None = None,
) -> Any:
    """Return a raw, residual, or conditional variogram entry."""
    form = require_form(form)
    if kind not in _VALID_KINDS:
        raise InvalidKindError(f"invalid variogram kind: {kind!r}")
    if kind == "residual":
        value = _residual_variogram(model, i, j, within)
    elif kind == "conditional":
        value = _conditional_variogram(model, i, j, within)
    else:
        mean_difference = model.fixed.mean_at(model, i) - model.fixed.mean_at(model, j)
        value = mean_difference**2 / 2 + _residual_variogram(model, i, j, within)
    if form is Form.SUM:
        return value
    matrix = sp.MatrixSymbol("Gamma", model.n, model.n)[i, j]
    if form is Form.MATRIX:
        return matrix
    return DualForm(sum=value, matrix=matrix)
