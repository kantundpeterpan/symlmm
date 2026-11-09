"""Notation selection and dual-form containers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import InvalidFormError, MissingFormError


class Form(str, Enum):
    """Supported notation modes."""

    SUM = "sum"
    MATRIX = "matrix"
    BOTH = "both"


def require_form(form: str | Form | None) -> Form:
    """Validate and normalize a required form argument."""
    if form is None:
        raise MissingFormError("form is required: use 'sum', 'matrix', or 'both'")
    try:
        return Form(form)
    except ValueError as exc:
        raise InvalidFormError(f"invalid form: {form!r}") from exc


@dataclass(frozen=True)
class EntryMap:
    """Lazy map from an index tuple to a symbolic entry."""

    func: Callable[..., sp.Expr]

    def __getitem__(self, key: tuple[sp.Expr, ...]) -> sp.Expr:
        if not isinstance(key, tuple):
            key = (key,)
        return self.func(*key)


@dataclass(frozen=True)
class DualForm:
    """Both notations for one quantity.

    `sum` is an expression for entry queries or an EntryMap for whole-matrix
    queries. `matrix` is a MatrixExpr or a MatrixElement.
    """

    sum: sp.Expr | EntryMap | None
    matrix: sp.MatrixExpr | sp.Expr | None


def matrix_to_sum(expr: Any, model: Any) -> EntryMap:
    """Return a lazy entry map for an abstract matrix expression."""
    return EntryMap(model.covariance_entry)


def sum_to_matrix(expr: Any, model: Any) -> Any:
    """Return the abstract matrix for a structured expression."""
    return model.matrix_covariance()
