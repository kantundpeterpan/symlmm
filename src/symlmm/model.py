"""Model base classes and index accessors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import ModelError
from .specs import Crossed, Group, Nested
from .symbols import observation_index

GroupSpec = Group | Nested | Crossed


class IndexedModel:
    """Base class for models over a stacked observation vector."""

    def __init__(
        self,
        *,
        n: Any,
        groups: Sequence[GroupSpec] = (),
        time: str | None = None,
        covariates: Sequence[str] = (),
    ) -> None:
        self.n = n
        self.groups = tuple(groups)
        self.time_name = time
        self.covariates = tuple(covariates)
        self._i = observation_index("i")
        self._j = observation_index("j")

    def indices(self) -> tuple[Any, Any]:
        """Return the two observation indices used by entry queries."""
        return self._i, self._j

    def _group_spec(self, factor: str) -> GroupSpec:
        for spec in self.groups:
            if spec.name == factor:
                return spec
        raise ModelError(f"unknown grouping factor: {factor!r}")

    def group(self, i: Any, factor: str) -> Any:
        """Return the group label for observation `i` and factor `factor`."""
        spec = self._group_spec(factor)
        return sp.IndexedBase(spec.label, shape=(self.n,))[i]

    def time(self, i: Any, coordinate: str = "t") -> Any:
        """Return the time or coordinate value for observation `i`."""
        name = self.time_name if coordinate == "time" else coordinate
        if name is None:
            raise ModelError("model has no time coordinate")
        return sp.IndexedBase(name, shape=(self.n,))[i]

    def covariate(self, i: Any, name: str) -> Any:
        """Return a covariate value for observation `i`."""
        return sp.IndexedBase(name, shape=(self.n,))[i]

    def design(self, i: Any) -> Any:
        """Return the fixed-effect design row `[1, time, covariates...]`."""
        terms = [1]
        if self.time_name is not None:
            terms.append(self.time(i))
        terms.extend(self.covariate(i, name) for name in self.covariates if name != self.time_name)
        return sp.Matrix(terms)
