"""Ordered symbolic collections of factor-level blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .blocks import Block
from .blup import _inverse
from .errors import ModelError


def _assemble(model: Any, expression: Any) -> Any:
    """Return an explicit matrix for dense models, otherwise the expression."""
    if model._dense:
        return expression.as_explicit()
    return expression


@dataclass(frozen=True)
class BlockSet:
    """An ordered set of blocks from one model."""

    model: Any
    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        """Validate model membership, uniqueness, and non-emptiness."""
        if not self.blocks:
            raise ModelError("block set requires at least one block")
        if any(block.model is not self.model for block in self.blocks):
            raise ModelError("all blocks must belong to the same model")
        keys = [(block.factor, block.level) for block in self.blocks]
        if len(keys) != len(set(keys)):
            raise ModelError("duplicate block in block set")

    def size(self) -> Any:
        """Return the total number of observations in the set."""
        return sp.Add(*[block.size() for block in self.blocks])

    def response(self) -> Any:
        """Return the stacked response vector."""
        return _assemble(
            self.model,
            sp.BlockMatrix([[block.response()] for block in self.blocks]),
        )

    def mean(self) -> Any:
        """Return the stacked fixed-effect mean."""
        return _assemble(
            self.model,
            sp.BlockMatrix([[block.mean()] for block in self.blocks]),
        )

    def design(self) -> Any:
        """Return the stacked fixed-effect design matrix."""
        return _assemble(
            self.model,
            sp.BlockMatrix([[block.design()] for block in self.blocks]),
        )

    def _dense_term_design(self, block: Block, term: str) -> Any:
        """Return the dense design for one block's random-effect level."""
        resolved = block.term(term)
        matrix = block.model.matrix_z()
        start = block.model._random_term_offset(resolved.random)
        levels = sorted(set(block.model._dense_group_codes(block.factor)))
        try:
            level_index = {level: index for index, level in enumerate(levels)}[block.level]
        except KeyError as exc:
            raise ModelError("unknown block level") from exc
        start += level_index * resolved.random.dim
        return matrix[block.rows(), start : start + resolved.random.dim]

    def random_design(self, term: str | None = None) -> Any:
        """Return the block-diagonal design matrix for random-effect columns.

        `term` is required when the set contains blocks from multiple factors.
        """
        if term is None and len({block.factor for block in self.blocks}) > 1:
            raise ModelError("random_design requires a term for mixed-factor block sets")
        if self.model._dense and term is not None:
            return sp.BlockDiagMatrix(
                *[self._dense_term_design(block, term) for block in self.blocks]
            ).as_explicit()
        return _assemble(
            self.model,
            sp.BlockDiagMatrix(*[block.random_design(term=term) for block in self.blocks]),
        )

    def covariance(self) -> Any:
        """Return the joint block covariance matrix."""
        return _assemble(
            self.model,
            sp.BlockMatrix(
                [[left.cross_covariance(right) for right in self.blocks] for left in self.blocks]
            ),
        )

    def cross_covariance(self, other: BlockSet) -> Any:
        """Return Cov(Y_self, Y_other)."""
        if self.model is not other.model:
            raise ModelError("cross covariance requires block sets from the same model")
        return _assemble(
            self.model,
            sp.BlockMatrix(
                [[left.cross_covariance(right) for right in other.blocks] for left in self.blocks]
            ),
        )

    def _term(self, term: str | None = None) -> Any:
        """Resolve the shared random-effect term for this set."""
        factors = {block.factor for block in self.blocks}
        if len(factors) != 1:
            raise ModelError("block-set BLUP requires blocks from one factor")
        return self.blocks[0].term(term)

    def shrinkage(self, term: str | None = None) -> Any:
        """Return the block-set shrinkage operator."""
        resolved = self._term(term)
        G = sp.BlockDiagMatrix(*[resolved.random.cov.dense() for _ in self.blocks]).as_explicit()
        Z = self.random_design(term=resolved.name)
        V = self.covariance()
        return G * Z.T * _inverse(V)

    def blup_mean(self, term: str | None = None) -> Any:
        """Return the joint BLUP for the random effects in this set."""
        resolved = self._term(term)
        beta = sp.Matrix(self.model.fixed.coefficients())
        residual = self.response() - self.design() * beta
        return self.shrinkage(resolved.name) * residual

    def blup_covariance(self, term: str | None = None) -> Any:
        """Return the joint conditional covariance for this set."""
        resolved = self._term(term)
        G = sp.BlockDiagMatrix(*[resolved.random.cov.dense() for _ in self.blocks]).as_explicit()
        Z = self.random_design(term=resolved.name)
        S = self.shrinkage(resolved.name)
        return G - S * Z * G
