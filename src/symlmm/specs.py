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
    index: sp.Symbol | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            object.__setattr__(self, "index", sp.Symbol(self.label, integer=True))

    @property
    def label(self) -> str:
        """Return the symbol label for this grouping factor."""
        return self.name


@dataclass(frozen=True)
class Nested:
    """A factor nested within another factor."""

    child: str
    within: str
    index: sp.Symbol | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            object.__setattr__(self, "index", sp.Symbol(self.label, integer=True))

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
    index: sp.Symbol | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            object.__setattr__(self, "index", sp.Symbol(self.label, integer=True))

    @property
    def name(self) -> str:
        """Return the factor name used in accessor lookups."""
        return self.right

    @property
    def label(self) -> str:
        """Return the composite symbol label for this grouping factor."""
        return f"{self.left}:{self.right}"


@dataclass(frozen=True)
class Covariate:
    """A named covariate/coordinate definition with its index symbol."""

    name: str
    index: sp.Symbol | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            object.__setattr__(self, "index", sp.Symbol(self.name, integer=True))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Covariate):
            return self.name == other.name and self.index == other.index
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


def _coerce_group(value: str | Group) -> Group:
    """Normalize a grouping factor definition."""
    return Group(value) if isinstance(value, str) else value


def _coerce_covariate(value: str | Covariate) -> Covariate:
    """Normalize a covariate definition."""
    return Covariate(value) if isinstance(value, str) else value


@dataclass(frozen=True)
class Fixed:
    """Fixed-effect specification."""

    intercept: sp.Expr | None = None
    slopes: Mapping[Covariate | str, sp.Expr] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[Covariate, sp.Expr] = {}
        for key, value in (self.slopes or {}).items():
            normalized[_coerce_covariate(key)] = value
        object.__setattr__(self, "slopes", types.MappingProxyType(normalized))

    def _slope_items(self) -> tuple[tuple[Covariate, sp.Expr], ...]:
        """Return normalized slope definitions and coefficients in design order."""
        return tuple((_coerce_covariate(key), value) for key, value in self.slopes.items())

    def covariates(self) -> tuple[Covariate, ...]:
        """Return the covariate definitions in design order."""
        return tuple(covariate for covariate, _ in self._slope_items())

    def names(self) -> tuple[str, ...]:
        """Return fixed-effect names in design order."""
        names = [] if self.intercept is None else ["intercept"]
        return tuple(names) + tuple(covariate.name for covariate, _ in self._slope_items())

    def coefficients(self) -> tuple[sp.Expr, ...]:
        """Return fixed-effect coefficients in design order."""
        values = [] if self.intercept is None else [self.intercept]
        return tuple(values) + tuple(value for _, value in self._slope_items())

    def mean_at(self, model: IndexedModel, i: sp.Expr) -> sp.Expr:
        """Return the fixed-effect mean for observation `i`."""
        value = sp.Integer(0) if self.intercept is None else self.intercept
        for covariate, coefficient in self._slope_items():
            value = value + coefficient * model.covariate(i, covariate.name)
        return value


@dataclass(frozen=True, init=False)
class RandomSlope:
    """A random-slope column in a named coordinate; no covariance."""

    coordinate: Covariate
    within: bool = True

    def __init__(self, coordinate: str | Covariate, within: bool = True) -> None:
        """Build a random-slope design definition."""
        object.__setattr__(self, "coordinate", _coerce_covariate(coordinate))
        object.__setattr__(self, "within", within)


@dataclass(frozen=True, init=False)
class RandomEffects:
    """One group's joint random-effect design and covariance."""

    group: Group
    cov: CovarianceSpec
    intercept: bool = False
    slopes: tuple[RandomSlope, ...] = ()

    def __init__(
        self,
        group: str | Group,
        cov: CovarianceSpec,
        intercept: bool = False,
        slopes: tuple[RandomSlope, ...] = (),
    ) -> None:
        """Build one random-effects block and validate its covariance dimension."""
        object.__setattr__(self, "group", _coerce_group(group))
        object.__setattr__(self, "cov", cov)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "slopes", tuple(slopes))
        if not intercept and not self.slopes:
            raise CovarianceConstructionError("RandomEffects requires an intercept or slope")
        if cov.dim != self.dim:
            raise CovarianceConstructionError(
                f"RandomEffects covariance must be {self.dim}x{self.dim}"
            )

    @property
    def dim(self) -> int:
        """Number of random coefficients in this block."""
        return int(self.intercept) + len(self.slopes)

    def coordinates(self) -> tuple[str, ...]:
        """Return the slope coordinate names in design order."""
        return tuple(slope.coordinate.name for slope in self.slopes)

    def coordinate_definitions(self) -> tuple[Covariate, ...]:
        """Return the slope coordinate definitions in design order."""
        return tuple(slope.coordinate for slope in self.slopes)

    def z(self, model: IndexedModel, i: Any) -> sp.Matrix:
        """Return the random-effect design column for observation `i`."""
        columns: list[Any] = [sp.Integer(1)] if self.intercept else []
        columns.extend(
            model.coordinate(
                i,
                slope.coordinate.name,
                within=self.group.name if slope.within else None,
            )
            for slope in self.slopes
        )
        return sp.Matrix(columns)
