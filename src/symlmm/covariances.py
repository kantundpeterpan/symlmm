"""Covariance structures for random effects and residuals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .constraints import Constraint, Relation, is_positive_definite
from .errors import CovarianceConstructionError


class CovarianceSpec(ABC):
    """Base class for covariance structures."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the covariance matrix."""

    @abstractmethod
    def entry(self, a: int, b: int) -> sp.Expr:
        """Return the (a, b) entry."""

    def dense(self) -> sp.Matrix:
        """Return the explicit matrix for this structure."""
        return sp.Matrix(self.dim, self.dim, lambda a, b: self.entry(a, b))

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return ()

    def is_positive_definite(self) -> bool | None:
        """Return a tri-state positive-definiteness verdict for this structure."""
        return is_positive_definite(self.dense())


RandomCovariance = CovarianceSpec


class Scalar(CovarianceSpec):
    """Scalar (1x1) covariance for a random intercept."""

    def __init__(self, var: sp.Expr) -> None:
        self.var = var

    @property
    def dim(self) -> int:
        """Dimension of the covariance matrix."""
        return 1

    def entry(self, a: int, b: int) -> sp.Expr:
        """Return the (a, b) entry."""
        return self.var if a == b == 0 else sp.Integer(0)

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return (Constraint(self.var, Relation.GT, "tau2 > 0"),)


class ResidualCovariance(ABC):
    """Base class for residual covariance structures."""

    @abstractmethod
    def entry(self, i: int, j: int, model: object) -> sp.Expr:
        """Return the (i, j) observation entry."""

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return ()


class IID(ResidualCovariance):
    """IID residual covariance."""

    def __init__(self, *, var: sp.Expr) -> None:
        self.var = var

    def entry(self, i: int, j: int, model: object) -> sp.Expr:
        """Return the (i, j) observation entry."""
        return self.var * sp.KroneckerDelta(i, j)

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return (Constraint(self.var, Relation.GT, "sigma2 > 0"),)


class CompoundSymmetry(ResidualCovariance):
    """Compound-symmetry residual covariance within groups."""

    def __init__(self, *, group: str, var: sp.Expr, rho: sp.Expr) -> None:
        """Build a compound-symmetry residual covariance."""
        self.group = group
        self.var = var
        self.rho = rho

    def entry(self, i: int, j: int, model: Any) -> sp.Expr:
        """Return the (i, j) observation entry."""
        if i == j:
            return self.var
        same_group = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
        return self.var * self.rho * same_group

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return (
            Constraint(self.rho + 1, Relation.GT, "rho > -1/(m_min - 1)"),
            Constraint(self.var, Relation.GT, "sigma2 > 0"),
        )


class Toeplitz(ResidualCovariance):
    """Toeplitz residual covariance with lag parameters."""

    def __init__(
        self,
        *,
        group: str,
        time: str,
        variances: Sequence[sp.Expr],
        correlations: Sequence[sp.Expr],
    ) -> None:
        """Build a Toeplitz residual covariance from variance and lag correlations."""
        self.group = group
        self.time = time
        self.variances = tuple(variances)
        self.correlations = tuple(correlations)

    def entry(self, i: int, j: int, model: Any) -> sp.Expr:
        """Return the (i, j) observation entry."""
        same_group = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
        lag = sp.Abs(model.time(i, self.time) - model.time(j, self.time))
        return self._lag_value(lag) * same_group

    def _lag_value(self, lag: sp.Expr) -> sp.Expr:
        """Return the covariance value for a time lag."""
        if lag.is_Integer:
            distance = int(lag)
            if distance == 0:
                return self.variances[0]
            if distance <= len(self.correlations):
                return self.variances[0] * self.correlations[distance - 1]
            return sp.Integer(0)
        return sp.Piecewise(
            *(
                (self.variances[0] * correlation, sp.Eq(lag, distance))
                for distance, correlation in enumerate(self.correlations, start=1)
            ),
            (sp.Integer(0), True),
        )

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return (Constraint(self.variances[0], Relation.GT, "sigma2 > 0"),)


class Heteroscedastic(ResidualCovariance):
    """Observation-specific residual variances."""

    def __init__(self, *, variances: Sequence[sp.Expr]) -> None:
        """Build a heteroscedastic residual covariance."""
        self.variances = tuple(variances)

    def entry(self, i: int, j: int, model: Any) -> sp.Expr:
        """Return the (i, j) observation entry."""
        if i == j:
            return self.variances[int(i)]
        return sp.Integer(0)

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return tuple(
            Constraint(value, Relation.GT, f"sigma2[{k}] > 0")
            for k, value in enumerate(self.variances)
        )


class AR1(ResidualCovariance):
    """AR(1) residual covariance within groups."""

    def __init__(self, *, group: str, time: str, var: sp.Expr, rho: sp.Expr) -> None:
        """Build an AR(1) residual covariance.

        ``time`` is the named time coordinate and is used by the covariance via
        ``model.time(i, self.time)``. With the default time resolution this is
        the same symbol as the canonical ``model.time(i)``.
        """
        self.group = group
        self.time = time
        self.var = var
        self.rho = rho

    def entry(self, i: int, j: int, model: Any) -> sp.Expr:
        """Return the (i, j) observation entry."""
        same_group = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
        lag = sp.Abs(model.time(i, self.time) - model.time(j, self.time))
        return self.var * self.rho**lag * same_group

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return (
            Constraint(self.rho + 1, Relation.GT, "-1 < rho"),
            Constraint(1 - self.rho, Relation.GT, "rho < 1"),
            Constraint(self.var, Relation.GT, "sigma2 > 0"),
        )


class Diagonal(CovarianceSpec):
    """Diagonal random-effect covariance."""

    def __init__(self, variances: Sequence[sp.Expr]) -> None:
        self.variances = tuple(variances)

    @property
    def dim(self) -> int:
        """Dimension of the covariance matrix."""
        return len(self.variances)

    def entry(self, a: int, b: int) -> sp.Expr:
        """Return the (a, b) entry."""
        if a != b:
            return sp.Integer(0)
        return self.variances[a]

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return tuple(
            Constraint(value, Relation.GT, f"d{a} > 0") for a, value in enumerate(self.variances)
        )


class Unstructured(CovarianceSpec):
    """Unstructured covariance, parameterized by SDs and correlations."""

    def __init__(self, matrix: sp.MatrixBase) -> None:
        if matrix.rows != matrix.cols:
            raise CovarianceConstructionError("covariance matrix must be square")
        if matrix != matrix.T:
            raise CovarianceConstructionError("covariance matrix must be symmetric")
        if is_positive_definite(matrix) is False:
            raise CovarianceConstructionError("covariance matrix must be positive definite")
        self.matrix = matrix
        self._corr: sp.Expr | None = None

    @classmethod
    def from_sd_corr(cls, sd: Sequence[sp.Expr], corr: sp.Expr) -> Unstructured:
        """Build a 2x2 covariance from two standard deviations and a correlation."""
        if len(sd) != 2:
            raise CovarianceConstructionError("v0.1 supports 2x2 SD+corr construction")
        s0, s1 = sd
        matrix = sp.Matrix([[s0**2, s0 * s1 * corr], [s0 * s1 * corr, s1**2]])
        instance = cls(matrix)
        instance._corr = corr
        return instance

    @classmethod
    def from_cholesky(cls, lower: sp.MatrixBase) -> Unstructured:
        """Build a covariance from a lower Cholesky factor."""
        return cls(lower * lower.T)

    @property
    def dim(self) -> int:
        """Dimension of the covariance matrix."""
        return int(self.matrix.rows)

    def entry(self, a: int, b: int) -> sp.Expr:
        """Return the (a, b) entry."""
        return self.matrix[a, b]

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        if self._corr is None:
            return ()
        return (
            Constraint(self._corr + 1, Relation.GT, "-1 < rho"),
            Constraint(1 - self._corr, Relation.GT, "rho < 1"),
        )
