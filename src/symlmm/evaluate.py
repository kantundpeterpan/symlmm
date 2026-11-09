"""Substitution, numeric evaluation, and lambdification."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .covariances import CovarianceSpec, ResidualCovariance
from .errors import UnresolvedSymbols
from .forms import Form, require_form
from .specs import Fixed


def _substitute_value(value: Any, mapping: Mapping[Any, Any]) -> Any:
    """Return `value` with parameters replaced."""
    if isinstance(value, (sp.Expr, sp.MatrixBase)):
        return value.subs(mapping)
    if isinstance(value, (CovarianceSpec, ResidualCovariance)):
        return _substitute_object(value, mapping)
    if isinstance(value, tuple):
        return tuple(_substitute_value(item, mapping) for item in value)
    return value


def _substitute_object(value: Any, mapping: Mapping[Any, Any]) -> Any:
    """Return a shallow copy of `value` with substituted attributes."""
    clone = copy.copy(value)
    for name, attribute in vars(value).items():
        object.__setattr__(clone, name, _substitute_value(attribute, mapping))
    return clone


def _substitute_fixed(fixed: Fixed, mapping: Mapping[Any, Any]) -> Fixed:
    """Return a fixed-effect specification with substituted coefficients."""
    intercept = None if fixed.intercept is None else fixed.intercept.subs(mapping)
    slopes = {name: coefficient.subs(mapping) for name, coefficient in fixed.slopes.items()}
    return Fixed(intercept=intercept, slopes=slopes)


def subs(model: Any, mapping: Mapping[Any, Any]) -> Any:
    """Return a new model with substituted parameters."""
    clone = object.__new__(type(model))
    clone.__dict__ = model.__dict__.copy()
    clone._substitutions = {**getattr(model, "_substitutions", {}), **mapping}
    if hasattr(model, "fixed"):
        clone.fixed = _substitute_fixed(model.fixed, mapping)
    if hasattr(model, "random"):
        clone.random = tuple(_substitute_object(term, mapping) for term in model.random)
    if hasattr(model, "residual"):
        clone.residual = _substitute_object(model.residual, mapping)
    return clone


def evaluate(model: Any, mapping: Mapping[Any, Any]) -> Any:
    """Return a numeric SymPy object after substitution."""
    result = model.subs(mapping)
    expression = result.matrix_covariance()
    free = expression.free_symbols - set(mapping)
    if free:
        raise UnresolvedSymbols(f"unresolved symbols: {sorted(map(str, free))}")
    return expression


def lambdify(model: Any, args: Any, form: str | Form | None = "matrix") -> Any:
    """Return a callable over the declared arguments."""
    selected = require_form(form)
    if selected is Form.MATRIX:
        expression = model.matrix_covariance()
    else:
        expression = model.matrix_mean()
    return sp.lambdify(args, expression, modules="numpy")
