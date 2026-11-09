"""Model specifications: grouping factors and designs."""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sympy as sp  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from .covariances import CovarianceSpec
    from .model import IndexedModel


@dataclass(frozen=True)
class Group:
    """A single grouping factor."""

    name: str

    @property
    def label(self) -> str:
        """Return the symbol label for this grouping factor."""
        return self.name


@dataclass(frozen=True)
class Nested:
    """A factor nested within another factor."""

    child: str
    within: str

    @property
    def name(self) -> str:
        """Return the factor name used in accessor lookups."""
        return self.child

    @property
    def label(self) -> str:
        """Return the composite symbol label for this grouping factor."""
        return f"{self.within}:{self.child}"


@dataclass(frozen=True)
class Crossed:
    """A crossed combination of factors."""

    left: str
    right: str

    @property
    def name(self) -> str:
        """Return the factor name used in accessor lookups."""
        return self.right

    @property
    def label(self) -> str:
        """Return the composite symbol label for this grouping factor."""
        return f"{self.left}:{self.right}"


@dataclass(frozen=True)
class Fixed:
    """Fixed-effect specification."""

    intercept: sp.Expr | None = None
    slopes: Mapping[str, sp.Expr] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slopes", types.MappingProxyType(dict(self.slopes or {})))

    def names(self) -> tuple[str, ...]:
        """Return fixed-effect names in design order."""
        names = [] if self.intercept is None else ["intercept"]
        return tuple(names) + tuple(self.slopes)

    def coefficients(self) -> tuple[sp.Expr, ...]:
        """Return fixed-effect coefficients in design order."""
        values = [] if self.intercept is None else [self.intercept]
        return tuple(values) + tuple(self.slopes.values())

    def mean_at(self, model: IndexedModel, i: sp.Expr) -> sp.Expr:
        """Return the fixed-effect mean for observation `i`."""
        value = sp.Integer(0) if self.intercept is None else self.intercept
        for name, coefficient in self.slopes.items():
            value = value + coefficient * model.covariate(i, name)
        return value


@dataclass(frozen=True)
class RandomIntercept:
    """Random-intercept term."""

    group: str
    cov: CovarianceSpec
