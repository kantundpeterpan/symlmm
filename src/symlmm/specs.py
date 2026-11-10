"""Model specifications: grouping factors and designs."""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import CovarianceConstructionError

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
class RandomSlope:
    """A random-slope column in a named coordinate; no covariance."""

    coordinate: str
    within: bool = True


@dataclass(frozen=True)
class RandomEffects:
    """One group's joint random-effect design and covariance."""

    group: str
    cov: CovarianceSpec
    intercept: bool = False
    slopes: tuple[RandomSlope, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slopes", tuple(self.slopes))
        if not self.intercept and not self.slopes:
            raise CovarianceConstructionError("RandomEffects requires an intercept or slope")
        if self.cov.dim != self.dim:
            raise CovarianceConstructionError(
                f"RandomEffects covariance must be {self.dim}x{self.dim}"
            )

    @property
    def dim(self) -> int:
        """Number of random coefficients in this block."""
        return int(self.intercept) + len(self.slopes)

    def coordinates(self) -> tuple[str, ...]:
        """Return the slope coordinate names in design order."""
        return tuple(slope.coordinate for slope in self.slopes)

    def z(self, model: IndexedModel, i: Any) -> sp.Matrix:
        """Return the random-effect design column for observation `i`."""
        columns: list[Any] = [sp.Integer(1)] if self.intercept else []
        columns.extend(
            model.coordinate(
                i,
                slope.coordinate,
                within=self.group if slope.within else None,
            )
            for slope in self.slopes
        )
        return sp.Matrix(columns)
