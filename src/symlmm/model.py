"""Model base classes and index accessors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import sympy as sp  # type: ignore[import-untyped]

from .covariances import IID, ResidualCovariance
from .errors import FormArityError, ModelError
from .specs import Crossed, Fixed, Group, Nested, RandomIntercept
from .symbols import observation_index

if TYPE_CHECKING:
    from .forms import Form

GroupSpec = Group | Nested | Crossed

__all__ = ["IID", "LMM", "IndexedModel"]


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


class LMM(IndexedModel):
    """Linear mixed model assembled from fixed, random, and residual terms."""

    def __init__(
        self,
        *,
        fixed: Fixed,
        random: Iterable[RandomIntercept],
        residual: ResidualCovariance,
        n: Any | None = None,
    ) -> None:
        self.fixed = fixed
        self.random = tuple(random)
        self.residual = residual
        groups = tuple(Group(term.group) for term in self.random)
        if hasattr(residual, "group"):
            groups = groups + (Group(residual.group),)
        time = getattr(residual, "time", None) or next(
            (getattr(term, "time", None) for term in self.random if getattr(term, "time", None)),
            None,
        )
        super().__init__(
            n=n if n is not None else sp.Symbol("n", positive=True, integer=True),
            groups=groups,
            time=time,
            covariates=tuple(fixed.slopes),
        )

    def levels(self, factor: str) -> Any:
        """Return the symbolic number of levels for a grouping factor."""
        return sp.Symbol(f"m_{factor}", positive=True, integer=True)

    def dimensions(self) -> tuple[Any, Any]:
        """Return observations and total random-coefficient dimension."""
        q = sum(self.levels(term.group) * term.cov.dim for term in self.random)
        return self.n, q

    def matrix_mean(self) -> Any:
        """Return the abstract matrix form of E[y]."""
        n = self.n
        p = len(self.fixed.coefficients())
        X = sp.MatrixSymbol("X", n, p)
        beta = sp.MatrixSymbol("beta", p, 1)
        return X * beta

    def matrix_covariance(self) -> Any:
        """Return the abstract matrix form of V = Z G Z' + R."""
        n, q = self.dimensions()
        Z = sp.MatrixSymbol("Z", n, q)
        G = sp.MatrixSymbol("G", q, q)
        R = sp.MatrixSymbol("R", n, n)
        return Z * G * Z.T + R

    def _random_term_entry(self, term: RandomIntercept, i: Any, j: Any) -> Any:
        """Return the structured contribution of one random term."""
        z_accessor = getattr(term, "z", None)
        zi = z_accessor(self, i) if z_accessor is not None else sp.Matrix([1])
        zj = z_accessor(self, j) if z_accessor is not None else sp.Matrix([1])
        value = (zi.T * term.cov.dense() * zj)[0, 0]
        return value * sp.KroneckerDelta(self.group(i, term.group), self.group(j, term.group))

    def covariance_entry(self, i: Any, j: Any) -> Any:
        """Return the structured sum-form covariance entry."""
        total = self.residual.entry(i, j, self)
        for term in self.random:
            total += self._random_term_entry(term, i, j)
        return total

    def mean(
        self,
        i: Any | None = None,
        j: Any | None = None,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the marginal mean in the requested form."""
        if j is not None:
            raise FormArityError("mean accepts at most one index")
        from .moments import mean

        return mean(self, i, form)

    def covariance(
        self,
        i: Any | None = None,
        j: Any | None = None,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the marginal covariance in the requested form."""
        from .moments import covariance

        return covariance(self, i, j, form)

    def variance(
        self,
        i: Any | None = None,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the marginal variance vector or entry in the requested form."""
        from .moments import variance

        return variance(self, i, form)

    def correlation(
        self,
        i: Any | None = None,
        j: Any | None = None,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the marginal correlation in the requested form."""
        from .moments import correlation

        return correlation(self, i, j, form)

    def conditional_mean(
        self,
        i: Any | None = None,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the conditional mean E[y | b] in the requested form."""
        from .moments import conditional_mean

        return conditional_mean(self, i, form)

    def icc(
        self,
        *,
        at: Mapping[str, Any] | None = None,
        form: str | Form | None = None,
    ) -> Any:
        """Return the conditional variance partition coefficient."""
        from .forms import DualForm, require_form

        form = require_form(form)
        random_part = sp.Integer(0)
        for term in self.random:
            z_accessor = getattr(term, "z", None)
            z = z_accessor(self, 0) if z_accessor is not None else sp.Matrix([1])
            term_time = getattr(term, "time", None)
            if at and term_time is not None:
                key = self.time(0, term_time)
                z = z.subs({key: at.get("time", key)})
            random_part += (z.T * term.cov.dense() * z)[0, 0]
        residual_part = self.residual.entry(0, 0, self)
        value = random_part / (random_part + residual_part)
        if form.value == "sum":
            return value
        return DualForm(sum=value, matrix=value)
