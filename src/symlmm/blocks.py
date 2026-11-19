"""Factor-level observation blocks and their moments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .blup import _inverse
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


def block_response_symbol(factor: Factor, level: Any, size: Any) -> sp.MatrixSymbol:
    """Return the symbolic response vector for one block."""
    return sp.MatrixSymbol(f"Y_{factor.label}:{level}", size, 1)


def _resolve_term(model: Any, name: Any) -> Any:
    """Return the term named `name`, or raise ModelError."""
    from .decomposition import terms

    for item in terms(model):
        if item.name == name:
            return item
    raise ModelError(f"unknown term name: {name!r}")


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

    def response(self) -> Any:
        """Return the symbolic response vector for this block."""
        return block_response_symbol(self.factor, self.level, self.size())

    def term(self, name: str | None = None) -> Any:
        """Return the random-effect term for this block.

        If more than one term belongs to the block's factor, `name` is required.
        """
        from .decomposition import terms

        candidates = [term for term in terms(self.model) if term.factor == self.factor]
        if name is None:
            if not candidates:
                raise NotDefined(f"factor {self.factor.name!r} has no random-effect term")
            if len(candidates) > 1:
                raise ModelError(
                    f"factor {self.factor.name!r} has multiple random-effect terms; "
                    "pass term=<name>"
                )
            return candidates[0]
        for term in candidates:
            if term.name == name:
                return term
        raise ModelError(f"unknown random-effect term: {name!r}")

    def mean(self) -> Any:
        """Return E[Y_block]."""
        if self.model._dense:
            return self.model.matrix_mean()[self.rows(), :]
        coefficients = sp.Matrix(self.model.fixed.coefficients())
        if coefficients.cols == 0:
            coefficients = sp.zeros(0, 1)
        return (
            block_mean_symbol(
                self.factor,
                self.level,
                coefficients.rows,
            )
            * coefficients
        )

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
        from .decomposition import terms

        total = block_cross_residual(self.factor, self.level, other.factor, other.level)
        for term in terms(self.model):
            Z = block_term_design(self.factor, self.level, term)
            Z_other = block_term_design(other.factor, other.level, term)
            total += Z * term.random.cov.dense() * Z_other.T
        return total

    def design(self) -> Any:
        """Return the fixed design block."""
        if self.model._dense:
            return self.model.matrix_x()[self.rows(), :]
        return block_mean_symbol(
            self.factor,
            self.level,
            len(self.model.fixed.coefficients()),
        )

    def _dense_term_bounds(self, term: Any) -> tuple[int, int]:
        """Return dense `matrix_z` column bounds for a term's full level layout."""
        start = self.model._random_term_offset(term.random)
        levels = len(set(self.model._dense_group_codes(term.factor)))
        return start, start + levels * term.random.dim

    def random_design(self, term: Any = None) -> Any:
        """Return the full or term-specific random design block.

        `term` is a term name string when given.
        """
        if self.model._dense:
            matrix = self.model.matrix_z()
            if term is None:
                return matrix[self.rows(), :]
            resolved = _resolve_term(self.model, term)
            start = self.model._random_term_offset(resolved.random)
            if resolved.factor == self.factor:
                labels = sorted(set(self.model._dense_group_codes(resolved.factor)))
                start += labels.index(self.level) * resolved.random.dim
            stop = start + resolved.random.dim
            return matrix[self.rows(), start:stop]
        if term is None:
            q = self.model.dimensions()[1]
            return block_random_design(self.factor, self.level, q)
        resolved = _resolve_term(self.model, term)
        return block_term_design(self.factor, self.level, resolved)

    def cross_residual(self, other: Block) -> Any:
        """Return the residual cross-covariance between this block and `other`."""
        if self.model is not other.model:
            raise ModelError("cross covariance requires blocks from the same model")
        if self.model._dense:
            matrix = self.model.matrix_r()
            return matrix[self.rows(), :][:, other.rows()]
        return block_cross_residual(self.factor, self.level, other.factor, other.level)

    def covariance_terms(self, other: Block | None = None) -> dict[str, Any]:
        """Return per-term contributions to Cov(Y_block, Y_other)."""
        from .decomposition import RESIDUAL, terms

        target = self if other is None else other
        if self.model is not target.model:
            raise ModelError("cross covariance requires blocks from the same model")
        values: dict[str, Any] = {}
        dense_z = self.model.matrix_z() if self.model._dense else None
        dense_g = self.model.matrix_g() if self.model._dense else None
        for term in terms(self.model):
            if dense_z is not None and dense_g is not None:
                start, stop = self._dense_term_bounds(term)
                Z = dense_z[self.rows(), start:stop]
                Z_other = dense_z[target.rows(), start:stop]
                G = dense_g[start:stop, start:stop]
                values[term.name] = Z * G * Z_other.T
            else:
                Z = self.random_design(term=term.name)
                Z_other = target.random_design(term=term.name)
                values[term.name] = Z * term.random.cov.dense() * Z_other.T
        values[RESIDUAL] = self.cross_residual(target)
        return values

    def partial_covariance(self, other: Block, *, exclude: tuple[str, ...] = ()) -> Any:
        """Return Cov(Y_block, Y_other) with named term contributions removed."""
        from .decomposition import RESIDUAL, terms, validate_exclusions

        resolved = terms(self.model)
        excluded = validate_exclusions(resolved, exclude)
        total = self.cross_covariance(other)
        if self.model._dense:
            values = self.covariance_terms(other)
            for name in excluded:
                total -= values[name]
            return sp.simplify(total)
        for term in resolved:
            if term.name not in excluded:
                continue
            Z = self.random_design(term=term.name)
            Z_other = other.random_design(term=term.name)
            total -= Z * term.random.cov.dense() * Z_other.T
        if RESIDUAL in excluded:
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

    def random_effect(self, term: str | None = None) -> Any:
        """Return the random-effect vector for one term of this block."""
        resolved = self.term(term)
        return block_coefficients(
            resolved.factor,
            self.level,
            resolved.random.dim,
        )

    def shrinkage(self, term: str | None = None) -> Any:
        """Return G_t Z_t' V_A^-1 for this block and random-effect term."""
        resolved = self.term(term)
        G = resolved.random.cov.dense()
        Z = self.random_design(term=resolved.name)
        V = self.covariance()
        return G * Z.T * _inverse(V)

    def blup_mean(self, term: str | None = None) -> Any:
        """Return E[b_A | Y_A] for this block and random-effect term."""
        resolved = self.term(term)
        residual = self.response() - self.mean()
        return self.shrinkage(resolved.name) * residual

    def blup_covariance(self, term: str | None = None) -> Any:
        """Return Cov(b_A | Y_A) for this block and random-effect term."""
        resolved = self.term(term)
        G = resolved.random.cov.dense()
        Z = self.random_design(term=resolved.name)
        S = self.shrinkage(resolved.name)
        return G - S * Z * G
