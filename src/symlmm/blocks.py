"""Factor-level observation blocks and their moments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import ModelError, NotDefined
from .specs import Factor


def block_size_symbol(factor: Factor, level: Any) -> sp.Symbol:
    """Return the symbolic size of a block."""
    return sp.Symbol(f"n_{factor.label}:{level}", positive=True, integer=True)


def block_mean_symbol(factor: Factor, level: Any, p: int) -> sp.MatrixSymbol:
    """Return the fixed design matrix for a block."""
    return sp.MatrixSymbol(f"X_{factor.label}:{level}", block_size_symbol(factor, level), p)


def block_random_design(factor: Factor, level: Any, q: Any) -> sp.MatrixSymbol:
    """Return the random design matrix for a block."""
    return sp.MatrixSymbol(f"Z_{factor.label}:{level}", block_size_symbol(factor, level), q)


def block_term_design(factor: Factor, level: Any, term: Any) -> sp.MatrixSymbol:
    """Return the random design matrix for one term and block."""
    return sp.MatrixSymbol(
        f"Z_{factor.label}:{level}:{term.name}",
        block_size_symbol(factor, level),
        term.random.dim,
    )


def block_residual(factor: Factor, level: Any) -> sp.MatrixSymbol:
    """Return the residual covariance matrix for a block."""
    size = block_size_symbol(factor, level)
    return sp.MatrixSymbol(f"R_{factor.label}:{level}", size, size)


def block_cross_residual(
    left: Factor,
    left_level: Any,
    right: Factor,
    right_level: Any,
) -> sp.MatrixSymbol:
    """Return the residual cross-covariance between two blocks."""
    if left == right and left_level == right_level:
        return block_residual(left, left_level)
    return sp.MatrixSymbol(
        f"R_{left.label}:{left_level}|{right.label}:{right_level}",
        block_size_symbol(left, left_level),
        block_size_symbol(right, right_level),
    )


def block_coefficients(factor: Factor, level: Any, q: Any) -> sp.MatrixSymbol:
    """Return the random-effect vector for a block."""
    return sp.MatrixSymbol(f"b_{factor.label}:{level}", q, 1)


@dataclass(frozen=True)
class Block:
    """The observation set for one factor level."""

    model: Any
    factor: Factor
    level: Any

    def rows(self) -> list[int]:
        """Return dense observation rows for this block."""
        if not self.model._dense:
            raise NotDefined("block rows require dense mode")
        codes = self.model._dense_group_codes(self.factor)
        return [index for index, code in enumerate(codes) if code == self.level]

    def size(self) -> Any:
        """Return the block size."""
        if self.model._dense:
            return len(self.rows())
        return block_size_symbol(self.factor, self.level)

    def mean(self) -> Any:
        """Return E[Y_block]."""
        if self.model._dense:
            return self.model.matrix_mean()[self.rows(), :]
        return block_mean_symbol(
            self.factor,
            self.level,
            len(self.model.fixed.coefficients()),
        ) * sp.Matrix(self.model.fixed.coefficients())

    def covariance(self) -> Any:
        """Return Cov(Y_block)."""
        return self.cross_covariance(self)

    def cross_covariance(self, other: Block) -> Any:
        """Return Cov(Y_block, Y_other)."""
        if self.model is not other.model:
            raise ModelError("cross covariance requires blocks from the same model")
        if self.model._dense:
            rows = self.rows()
            columns = other.rows()
            matrix = self.model.matrix_covariance()
            return matrix[rows, :][:, columns]
        q = self.model.dimensions()[1]
        Z = block_random_design(self.factor, self.level, q)
        Z_other = block_random_design(other.factor, other.level, q)
        G = self.model.matrix_g()
        R = block_cross_residual(self.factor, self.level, other.factor, other.level)
        return Z * G * Z_other.T + R

    def design(self) -> Any:
        """Return the fixed design block."""
        if self.model._dense:
            return self.model.matrix_x()[self.rows(), :]
        return block_mean_symbol(
            self.factor,
            self.level,
            len(self.model.fixed.coefficients()),
        )

    def random_design(self, term: Any = None) -> Any:
        """Return the full or term-specific random design block."""
        if self.model._dense:
            matrix = self.model.matrix_z()
            if term is None:
                return matrix[self.rows(), :]
            from .decomposition import terms

            resolved = next(item for item in terms(self.model) if item.name == term)
            start = self.model._random_term_offset(resolved.random)
            stop = start + resolved.random.dim
            return matrix[self.rows(), start:stop]
        if term is None:
            q = self.model.dimensions()[1]
            return block_random_design(self.factor, self.level, q)
        from .decomposition import terms

        resolved = next(item for item in terms(self.model) if item.name == term)
        return block_term_design(self.factor, self.level, resolved)

    def covariance_terms(self) -> dict[str, Any]:
        """Return per-term contributions to Cov(Y_block)."""
        values: dict[str, Any] = {}
        from .decomposition import terms

        for term in terms(self.model):
            Z = self.random_design(term=term.name)
            values[term.name] = Z * term.random.cov.dense() * Z.T
        values["residual"] = self.residual()
        return values

    def partial_covariance(self, other: Block, *, exclude: tuple[str, ...] = ()) -> Any:
        """Return Cov(Y_block, Y_other) with named term contributions removed."""
        from .decomposition import terms, validate_exclusions

        resolved = terms(self.model)
        validate_exclusions(resolved, exclude)
        total = self.cross_covariance(other)
        for term in resolved:
            if term.name not in exclude:
                continue
            Z = self.random_design(term=term.name)
            Z_other = other.random_design(term=term.name)
            total -= Z * term.random.cov.dense() * Z_other.T
        if "residual" in exclude:
            if self.model._dense:
                total -= self.model.matrix_r()[self.rows(), :][:, other.rows()]
            else:
                total -= block_cross_residual(
                    self.factor,
                    self.level,
                    other.factor,
                    other.level,
                )
        return sp.simplify(total)

    def residual(self) -> Any:
        """Return the residual covariance block."""
        if self.model._dense:
            matrix = self.model.matrix_r()
            return matrix[self.rows(), :][:, self.rows()]
        return block_residual(self.factor, self.level)

    def random_effect(self) -> Any:
        """Return the random-effect vector for this block, if one exists."""
        random = next(
            (
                item
                for item in self.model.random
                if self.model.factor(item.group.name) == self.factor
            ),
            None,
        )
        if random is None:
            raise NotDefined(f"factor {self.factor.name!r} has no random-effect term")
        return block_coefficients(self.factor, self.level, random.dim)
