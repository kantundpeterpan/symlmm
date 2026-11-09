"""Model base classes and index accessors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import sympy as sp  # type: ignore[import-untyped]

from .covariances import IID, ResidualCovariance
from .errors import FormArityError, ModelError
from .specs import Crossed, Fixed, Group, Nested, RandomIntercept, RandomSlope
from .symbols import observation_index

if TYPE_CHECKING:
    from .forms import Form

GroupSpec = Group | Nested | Crossed

__all__ = ["IID", "LMM", "IndexedModel"]


class IndexedModel:
    """Base class for models over a stacked observation vector."""

    _dense: bool = False
    _observations: Mapping[str, Sequence[Any]]

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

    def _observation_column(self, name: str) -> Sequence[Any]:
        """Return the dense observation column named `name`."""
        try:
            return self._observations[name]
        except KeyError as exc:
            raise ModelError(f"dense mode missing observation column: {name!r}") from exc

    def _observation_value(self, i: Any, name: str) -> Any:
        """Return the dense value at observation `i` for column `name`."""
        return sp.sympify(self._observation_column(name)[int(i)])

    def group(self, i: Any, factor: str) -> Any:
        """Return the group label for observation `i` and factor `factor`."""
        if self._dense:
            return self._observation_value(i, factor)
        spec = self._group_spec(factor)
        return sp.IndexedBase(spec.label, shape=(self.n,))[i]

    def time(self, i: Any, coordinate: str | None = None) -> Any:
        """Return the time or coordinate value for observation `i`."""
        if coordinate is None:
            name = self.time_name if self.time_name is not None else "t"
        else:
            name = coordinate
        if self._dense:
            return self._observation_value(i, name)
        return sp.IndexedBase(name, shape=(self.n,))[i]

    def covariate(self, i: Any, name: str) -> Any:
        """Return a covariate value for observation `i`."""
        if self._dense:
            return self._observation_value(i, name)
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
        random: Iterable[RandomIntercept | RandomSlope],
        residual: ResidualCovariance,
        groups: Sequence[GroupSpec] | None = None,
        n: Any | None = None,
    ) -> None:
        self.fixed = fixed
        self.random = tuple(random)
        self.residual = residual
        registered: dict[str, GroupSpec] = {}
        for spec in groups or ():
            registered.setdefault(spec.name, spec)
        for term in self.random:
            registered.setdefault(term.group, Group(term.group))
        if hasattr(residual, "group"):
            registered.setdefault(residual.group, Group(residual.group))
        time = getattr(residual, "time", None) or next(
            (getattr(term, "time", None) for term in self.random if getattr(term, "time", None)),
            None,
        )
        super().__init__(
            n=n if n is not None else sp.Symbol("n", positive=True, integer=True),
            groups=tuple(registered.values()),
            time=time,
            covariates=tuple(fixed.slopes),
        )

    def levels(self, factor: str) -> Any:
        """Return the symbolic number of levels for a grouping factor."""
        return sp.Symbol(f"m_{factor}", positive=True, integer=True)

    def dimensions(self) -> tuple[Any, Any]:
        """Return observations and total random-coefficient dimension."""
        if self._dense:
            q = sum(
                len(set(self._observation_column(term.group))) * term.cov.dim
                for term in self.random
            )
            return self.n, q
        q = sum(self.levels(term.group) * term.cov.dim for term in self.random)
        return self.n, q

    def dense(
        self,
        observations: Mapping[str, Sequence[Any]] | None = None,
        **columns: Sequence[Any],
    ) -> LMM:
        """Return a concrete copy with observation data."""
        data: dict[str, Sequence[Any]] = dict(observations or {})
        data.update(columns)
        if not data:
            raise ModelError("dense mode requires observation columns")
        lengths = {len(values) for values in data.values()}
        if len(lengths) != 1:
            raise ModelError("dense mode observation columns must have equal length")
        dense: LMM = object.__new__(type(self))
        dense.__dict__ = self.__dict__.copy()
        dense._observations = data
        dense._dense = True
        dense.n = lengths.pop()
        return dense

    def _column(self, name: str) -> Any:
        """Return a dense observation column as a matrix."""
        return sp.Matrix(self._observation_column(name))

    def matrix_x(self) -> Any:
        """Return the fixed-effect design matrix."""
        if not self._dense:
            return self.matrix_mean()
        columns = []
        if self.fixed.intercept is not None:
            columns.append(sp.ones(self.n, 1))
        for name in self.fixed.slopes:
            columns.append(self._column(name))
        if not columns:
            return sp.zeros(self.n, 0)
        return sp.Matrix.hstack(*columns)

    def matrix_z(self) -> Any:
        """Return the random-effect design matrix."""
        if not self._dense:
            n, q = self.dimensions()
            return sp.MatrixSymbol("Z", n, q)
        columns = []
        for term in self.random:
            labels = list(self._observation_column(term.group))
            time_name = getattr(term, "time", None)
            if time_name is None:
                times = [0] * len(labels)
            else:
                times = list(self._observations.get(time_name, [0] * len(labels)))
            for level in sorted(set(labels)):
                for a in range(term.cov.dim):
                    column = []
                    for label, t in zip(labels, times):
                        if label != level:
                            column.append(0)
                        elif term.cov.dim == 1:
                            column.append(1)
                        else:
                            column.append(1 if a == 0 else t)
                    columns.append(sp.Matrix(column))
        if not columns:
            return sp.zeros(self.n, 0)
        return sp.Matrix.hstack(*columns)

    def matrix_g(self) -> Any:
        """Return the block-diagonal random-effect covariance matrix."""
        if not self._dense:
            _, q = self.dimensions()
            return sp.MatrixSymbol("G", q, q)
        from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix  # type: ignore[import-untyped]

        blocks = []
        for term in self.random:
            labels = sorted(set(self._observation_column(term.group)))
            blocks.extend([term.cov.dense()] * len(labels))
        if not blocks:
            return sp.zeros(0, 0)
        return BlockDiagMatrix(*blocks).as_explicit()

    def matrix_r(self) -> Any:
        """Return the residual covariance matrix."""
        if not self._dense:
            return sp.MatrixSymbol("R", self.n, self.n)
        return sp.Matrix(
            self.n,
            self.n,
            lambda a, b: self.residual.entry(sp.Integer(a), sp.Integer(b), self),
        )

    def matrix_mean(self) -> Any:
        """Return the matrix form of E[y]."""
        if not self._dense:
            n = self.n
            p = len(self.fixed.coefficients())
            X = sp.MatrixSymbol("X", n, p)
            beta = sp.MatrixSymbol("beta", p, 1)
            return X * beta
        return self.matrix_x() * sp.Matrix(self.fixed.coefficients())

    def matrix_covariance(self) -> Any:
        """Return the matrix form of V = Z G Z' + R."""
        if not self._dense:
            n, q = self.dimensions()
            Z = sp.MatrixSymbol("Z", n, q)
            G = sp.MatrixSymbol("G", q, q)
            R = sp.MatrixSymbol("R", n, n)
            return Z * G * Z.T + R
        n = self.n
        V = sp.zeros(n, n)
        for a in range(n):
            for b in range(n):
                V[a, b] = self.covariance_entry(sp.Integer(a), sp.Integer(b))
        return V

    def subs(self, mapping: Mapping[Any, Any]) -> LMM:
        """Return a new model with substituted parameters."""
        from .evaluate import subs

        result: LMM = subs(self, mapping)
        return result

    def evaluate(self, mapping: Mapping[Any, Any]) -> Any:
        """Return a numeric SymPy object after substitution."""
        from .evaluate import evaluate

        return evaluate(self, mapping)

    def lambdify(self, args: Any, form: str | Form | None = "matrix") -> Any:
        """Return a callable over the declared arguments."""
        from .evaluate import lambdify

        return lambdify(self, args, form)

    def _random_term_entry(self, term: RandomIntercept | RandomSlope, i: Any, j: Any) -> Any:
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

    def conditional_covariance(
        self,
        i: Any | None = None,
        j: Any | None = None,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the conditional covariance Cov[y | b] in the requested form."""
        from .moments import conditional_covariance

        return conditional_covariance(self, i, j, form)

    def conditional_variance(
        self,
        i: Any | None = None,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the conditional variance vector or entry in the requested form."""
        from .moments import conditional_variance

        return conditional_variance(self, i, form)

    def variogram(
        self,
        i: Any,
        j: Any,
        *,
        kind: str,
        form: str | Form | None = None,
    ) -> Any:
        """Return a raw, residual, or conditional variogram entry."""
        from .variogram import variogram

        return variogram(self, i, j, kind, form)

    def covariogram(
        self,
        i: Any,
        j: Any,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the covariogram entry Cov(y_i, y_j)."""
        from .variogram import covariogram

        return covariogram(self, i, j, form)

    def blup_mean(
        self,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the BLUP mean E[b | y] in the requested form."""
        from .blup import blup_mean

        return blup_mean(self, form)

    def blup_covariance(
        self,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the BLUP covariance Cov[b | y] in the requested form."""
        from .blup import blup_covariance

        return blup_covariance(self, form)

    def blup_cross_covariance(
        self,
        i: Any | None = None,
        *,
        form: str | Form | None = None,
    ) -> Any:
        """Return the BLUP cross-covariance Cov[y_i, b] in the requested form."""
        from .blup import blup_cross_covariance

        return blup_cross_covariance(self, i, form)

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
