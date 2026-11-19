"""Ordered symbolic collections of factor-level blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .blocks import Block
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
        start += levels.index(block.level) * resolved.random.dim
        return matrix[block.rows(), start : start + resolved.random.dim]

    def random_design(self, term: str | None = None) -> Any:
        """Return the block-diagonal design matrix for random-effect columns."""
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
