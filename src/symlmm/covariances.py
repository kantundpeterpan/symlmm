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


class BlockDiag(CovarianceSpec):
    """Block-diagonal covariance assembled from explicit blocks.

    v0.1 accepts a finite, explicit sequence of blocks; symbolic block counts
    are not supported (design decision 9). ``group`` optionally records the
    grouping factor the blocks belong to.
    """

    def __init__(
        self,
        blocks: Sequence[CovarianceSpec | sp.MatrixBase],
        group: str | None = None,
    ) -> None:
        if not blocks:
            raise CovarianceConstructionError("BlockDiag requires at least one block")
        self.blocks = tuple(_as_covariance_block(block) for block in blocks)
        self.group = group

    @property
    def dim(self) -> int:
        """Dimension of the covariance matrix."""
        return sum(block.dim for block in self.blocks)

    def _block_entry(self, index: int) -> tuple[int, int]:
        """Return the block position and local index for a global index."""
        offset = 0
        for position, block in enumerate(self.blocks):
            if index < offset + block.dim:
                return position, index - offset
            offset += block.dim
        raise IndexError(f"covariance index out of range: {index}")

    def entry(self, a: int, b: int) -> sp.Expr:
        """Return the (a, b) entry."""
        block_a, row = self._block_entry(a)
        block_b, column = self._block_entry(b)
        if block_a != block_b:
            return sp.Integer(0)
        return self.blocks[block_a].entry(row, column)

    def dense(self) -> sp.Matrix:
        """Return the explicit block-diagonal matrix."""
        return sp.diag(*(block.dense() for block in self.blocks))

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints for every block."""
        return tuple(constraint for block in self.blocks for constraint in block.constraints())


def _as_covariance_block(block: CovarianceSpec | sp.MatrixBase) -> CovarianceSpec:
    """Coerce an explicit matrix block into a covariance structure."""
    if isinstance(block, CovarianceSpec):
        return block
    if isinstance(block, sp.MatrixBase):
        return Unstructured(block)
    raise CovarianceConstructionError(
        "BlockDiag blocks must be covariance structures or explicit matrices"
    )


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

    def __init__(self, *, group: str, var: sp.Expr, rho: sp.Expr, m_max: int = 2) -> None:
        """Build a compound-symmetry residual covariance."""
        if m_max < 2:
            raise CovarianceConstructionError("m_max must be at least 2")
        self.group = group
        self.var = var
        self.rho = rho
        self.m_max = m_max

    def entry(self, i: int, j: int, model: Any) -> sp.Expr:
        """Return the (i, j) observation entry."""
        same_group = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
        return sp.Piecewise(
            (self.var, sp.Eq(i, j)),
            (self.var * self.rho * same_group, True),
        )

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return (
            Constraint(
                (self.m_max - 1) * self.rho + 1,
                Relation.GT,
                f"rho > -1/({self.m_max}-1)",
            ),
            Constraint(1 - self.rho, Relation.GT, "rho < 1"),
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
        diagonal = sp.Piecewise(
            *((value, sp.Eq(i, k)) for k, value in enumerate(self.variances)),
            (sp.Integer(0), True),
        )
        return diagonal * sp.KroneckerDelta(i, j)

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


def _euclidean_distance(i: int, j: int, model: Any) -> sp.Expr:
    """Return the Euclidean distance between observations `i` and `j`."""
    dx = model.covariate(i, "x") - model.covariate(j, "x")
    dy = model.covariate(i, "y") - model.covariate(j, "y")
    return sp.sqrt(dx**2 + dy**2)


class Exponential(ResidualCovariance):
    """Exponential spatial covariance."""

    def __init__(self, *, var: sp.Expr, range: sp.Expr) -> None:
        self.var = var
        self.range = range

    def entry(self, i: int, j: int, model: Any) -> sp.Expr:
        """Return the (i, j) observation entry."""
        return self.entry_from_distance(_euclidean_distance(i, j, model))

    def entry_from_distance(self, distance: sp.Expr) -> sp.Expr:
        """Return the covariance entry for a distance."""
        return self.var * sp.exp(-distance / self.range)

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return (
            Constraint(self.var, Relation.GT, "sigma2 > 0"),
            Constraint(self.range, Relation.GT, "range > 0"),
        )


class Gaussian(ResidualCovariance):
    """Gaussian spatial covariance."""

    def __init__(self, *, var: sp.Expr, range: sp.Expr) -> None:
        self.var = var
        self.range = range

    def entry(self, i: int, j: int, model: Any) -> sp.Expr:
        """Return the (i, j) observation entry."""
        return self.entry_from_distance(_euclidean_distance(i, j, model))

    def entry_from_distance(self, distance: sp.Expr) -> sp.Expr:
        """Return the covariance entry for a distance."""
        return self.var * sp.exp(-(distance**2) / (2 * self.range**2))

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return (
            Constraint(self.var, Relation.GT, "sigma2 > 0"),
            Constraint(self.range, Relation.GT, "range > 0"),
        )


class Matern(ResidualCovariance):
    """Matern spatial covariance for nu in {1/2, 3/2, 5/2}."""

    def __init__(self, *, var: sp.Expr, range: sp.Expr, nu: sp.Expr) -> None:
        if nu not in (sp.Rational(1, 2), sp.Rational(3, 2), sp.Rational(5, 2)):
            raise CovarianceConstructionError(f"unsupported Matern nu: {nu}")
        self.var = var
        self.range = range
        self.nu = nu

    def entry(self, i: int, j: int, model: Any) -> sp.Expr:
        """Return the (i, j) observation entry."""
        return self.entry_from_distance(_euclidean_distance(i, j, model))

    def entry_from_distance(self, distance: sp.Expr) -> sp.Expr:
        """Return the covariance entry for a distance."""
        scaled = distance / self.range
        if self.nu == sp.Rational(1, 2):
            return self.var * sp.exp(-scaled)
        if self.nu == sp.Rational(3, 2):
            return self.var * (1 + sp.sqrt(3) * scaled) * sp.exp(-sp.sqrt(3) * scaled)
        if self.nu == sp.Rational(5, 2):
            return (
                self.var
                * (1 + sp.sqrt(5) * scaled + 5 * scaled**2 / 3)
                * sp.exp(-sp.sqrt(5) * scaled)
            )
        raise CovarianceConstructionError(f"unsupported Matern nu: {self.nu}")

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return (
            Constraint(self.var, Relation.GT, "sigma2 > 0"),
            Constraint(self.range, Relation.GT, "range > 0"),
        )
