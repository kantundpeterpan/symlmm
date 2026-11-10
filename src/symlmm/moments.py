"""Moment queries for symbolic models."""

from __future__ import annotations

from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import FormArityError, UnsupportedForm
from .forms import DualForm, EntryMap, Form, require_form


def mean(model: Any, i: Any | None = None, form: str | Form | None = None) -> Any:
    """Return the marginal mean."""
    form = require_form(form)
    if i is not None:
        if form is Form.BOTH:
            return DualForm(
                sum=model.fixed.mean_at(model, i),
                matrix=model.matrix_mean()[i, 0],
            )
        if form is Form.SUM:
            return model.fixed.mean_at(model, i)
        return model.matrix_mean()[i, 0]
    if form is Form.SUM:
        raise FormArityError("mean requires an index for form='sum'")
    matrix = model.matrix_mean()
    if form is Form.MATRIX:
        return matrix
    return DualForm(sum=EntryMap(lambda k, _=0: model.fixed.mean_at(model, k)), matrix=matrix)


def covariance(
    model: Any,
    i: Any | None = None,
    j: Any | None = None,
    form: str | Form | None = None,
    within: str | None = None,
) -> Any:
    """Return the marginal covariance.

    ``within`` is only supported for sum-form entries.
    """
    form = require_form(form)
    if within is not None and form is not Form.SUM:
        raise UnsupportedForm("within is only defined for sum-form entries")
    if (i is None) != (j is None):
        raise FormArityError("covariance requires both indices or neither")
    if i is None:
        if form is Form.SUM:
            raise FormArityError("covariance requires indices for form='sum'")
        matrix = model.matrix_covariance()
        if form is Form.MATRIX:
            return matrix
        return DualForm(
            sum=EntryMap(lambda a, b: model.covariance_entry(a, b, within)),
            matrix=matrix,
        )
    if form is Form.SUM:
        return model.covariance_entry(i, j, within)
    if form is Form.MATRIX:
        return model.matrix_covariance()[i, j]
    return DualForm(
        sum=model.covariance_entry(i, j, within),
        matrix=model.matrix_covariance()[i, j],
    )


def variance(
    model: Any,
    i: Any | None = None,
    form: str | Form | None = None,
    within: str | None = None,
) -> Any:
    """Return the marginal variance vector or entry.

    ``within`` is only supported for sum-form entries.
    """
    form = require_form(form)
    if within is not None and form is not Form.SUM:
        raise UnsupportedForm("within is only defined for sum-form entries")
    if i is None:
        if form is Form.SUM:
            raise FormArityError("variance requires an index for form='sum'")
        # DiagonalOf is a column vector of variances, not a diagonal matrix.
        vector = sp.matrices.expressions.diagonal.DiagonalOf(model.matrix_covariance())
        if form is Form.MATRIX:
            return vector
        return DualForm(
            sum=EntryMap(lambda k: model.covariance_entry(k, k, within)),
            matrix=vector,
        )
    if form is Form.SUM:
        return model.covariance_entry(i, i, within)
    if form is Form.MATRIX:
        return model.matrix_covariance()[i, i]
    return DualForm(
        sum=model.covariance_entry(i, i, within),
        matrix=model.matrix_covariance()[i, i],
    )


def correlation(
    model: Any,
    i: Any | None = None,
    j: Any | None = None,
    form: str | Form | None = None,
    within: str | None = None,
) -> Any:
    """Return the marginal correlation.

    ``within`` is only supported for sum-form entries.
    """
    form = require_form(form)
    if within is not None and form is not Form.SUM:
        raise UnsupportedForm("within is only defined for sum-form entries")
    if (i is None) != (j is None):
        raise FormArityError("correlation requires both indices or neither")
    corr = sp.MatrixSymbol("Corr", model.n, model.n)
    if i is None:
        if form is Form.SUM:
            raise FormArityError("correlation requires indices for form='sum'")
        if form is Form.MATRIX:
            return corr
        return DualForm(
            sum=EntryMap(lambda a, b: _correlation_entry(model, a, b, within)),
            matrix=corr,
        )
    entry = _correlation_entry(model, i, j, within)
    if form is Form.SUM:
        return entry
    if form is Form.MATRIX:
        return corr[i, j]
    return DualForm(sum=entry, matrix=corr[i, j])


def _correlation_entry(model: Any, i: Any, j: Any, within: str | None = None) -> Any:
    vi = model.covariance_entry(i, i, within)
    vj = model.covariance_entry(j, j, within)
    return model.covariance_entry(i, j, within) / sp.sqrt(vi * vj)


def conditional_mean(
    model: Any,
    i: Any | None = None,
    form: str | Form | None = None,
) -> Any:
    """Return E[y | b].

    The sum form expands component-wise, where
    ``b_{group}_{term_index}[level, a]`` denotes a component of the stacked
    random-effect vector ``b``.
    """
    form = require_form(form)
    if i is None:
        n, q = model.dimensions()
        Z = sp.MatrixSymbol("Z", n, q)
        b = sp.MatrixSymbol("b", q, 1)
        matrix = model.matrix_mean() + Z * b
        if form is Form.SUM:
            raise FormArityError("conditional_mean requires an index for form='sum'")
        if form is Form.MATRIX:
            return matrix
        return DualForm(
            sum=EntryMap(lambda k: _conditional_mean_entry(model, k)),
            matrix=matrix,
        )
    entry = _conditional_mean_entry(model, i)
    if form is Form.SUM:
        return entry
    n, q = model.dimensions()
    Z = sp.MatrixSymbol("Z", n, q)
    b = sp.MatrixSymbol("b", q, 1)
    matrix = (model.matrix_mean() + Z * b)[i, 0]
    if form is Form.MATRIX:
        return matrix
    return DualForm(sum=entry, matrix=matrix)


def _conditional_mean_entry(model: Any, i: Any) -> Any:
    value = model.fixed.mean_at(model, i)
    for term_index, term in enumerate(model.random):
        z = term.z(model, i) if hasattr(term, "z") else sp.Matrix([1])
        base = f"b_{term.group}_{term_index}".replace(":", "_")
        level = model.group(i, term.group)
        for a in range(term.cov.dim):
            value += z[a] * sp.IndexedBase(base)[level, a]
    return value


def conditional_covariance(
    model: Any,
    i: Any | None = None,
    j: Any | None = None,
    form: str | Form | None = None,
) -> Any:
    """Return the conditional covariance Cov[y | b] = R."""
    form = require_form(form)
    if (i is None) != (j is None):
        raise FormArityError("conditional_covariance requires both indices or neither")
    R = sp.MatrixSymbol("R", model.n, model.n)
    if i is None:
        if form is Form.SUM:
            raise FormArityError("conditional_covariance requires indices for form='sum'")
        if form is Form.MATRIX:
            return R
        return DualForm(
            sum=EntryMap(lambda a, b: model.residual.entry(a, b, model)),
            matrix=R,
        )
    if form is Form.SUM:
        return model.residual.entry(i, j, model)
    if form is Form.MATRIX:
        return R[i, j]
    return DualForm(
        sum=model.residual.entry(i, j, model),
        matrix=R[i, j],
    )


def conditional_variance(
    model: Any,
    i: Any | None = None,
    form: str | Form | None = None,
) -> Any:
    """Return the conditional variance vector or entry."""
    form = require_form(form)
    R = sp.MatrixSymbol("R", model.n, model.n)
    if i is None:
        if form is Form.SUM:
            raise FormArityError("conditional_variance requires an index for form='sum'")
        vector = sp.matrices.expressions.diagonal.DiagonalOf(R)
        if form is Form.MATRIX:
            return vector
        return DualForm(
            sum=EntryMap(lambda k: model.residual.entry(k, k, model)),
            matrix=vector,
        )
    if form is Form.SUM:
        return model.residual.entry(i, i, model)
    if form is Form.MATRIX:
        return R[i, i]
    return DualForm(
        sum=model.residual.entry(i, i, model),
        matrix=R[i, i],
    )
