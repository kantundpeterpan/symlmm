"""Ordered symbolic collections of factor-level blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .blocks import Block
from .errors import ModelError
from .structured import woodbury_covariance, woodbury_precision, woodbury_shrinkage


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

    def structured_random_design(self, term: str | None = None) -> Any:
        """Return the block-diagonal structured random design.

        `term` is required when the set contains blocks from multiple factors.
        When supplied, every block must belong to that term's factor.
        """
        if term is None:
            if len({block.factor for block in self.blocks}) > 1:
                raise ModelError(
                    "structured_random_design requires a term for mixed-factor block sets"
                )
        else:
            resolved = next(
                (item for item in self.model.terms() if item.name == term),
                None,
            )
            if resolved is None:
                raise ModelError(f"unknown random-effect term: {term!r}")
            if any(block.factor != resolved.factor for block in self.blocks):
                raise ModelError(
                    "structured_random_design requires blocks from the term's factor; "
                    f"term {term!r} belongs to factor {resolved.factor.name!r}"
                )
        return _assemble(
            self.model,
            sp.BlockDiagMatrix(
                *[block.structured_random_design(term=term) for block in self.blocks]
            ),
        )

    def residual_precision(self) -> Any:
        """Return the block-diagonal residual precision."""
        return _assemble(
            self.model,
            sp.BlockDiagMatrix(*[block.residual_precision() for block in self.blocks]),
        )

    def precision(self) -> Any:
        """Return the joint V^-1 using Woodbury.

        Single-term models only: structured block-set queries raise
        `NotDefined` when the model has more than one random-effect term.
        """
        resolved = self._term()
        G = sp.BlockDiagMatrix(*[resolved.random.cov.dense() for _ in self.blocks]).as_explicit()
        Z = self.structured_random_design(resolved.name)
        return woodbury_precision(G, Z, self.residual_precision())

    def _term(self, term: str | None = None) -> Any:
        """Resolve the shared random-effect term for this set."""
        factors = {block.factor for block in self.blocks}
        if len(factors) != 1:
            raise ModelError("block-set BLUP requires blocks from one factor")
        self.blocks[0]._require_single_term()
        return self.blocks[0].term(term)

    def shrinkage(self, term: str | None = None) -> Any:
        """Return the block-set shrinkage operator using Woodbury."""
        resolved = self._term(term)
        G = sp.BlockDiagMatrix(*[resolved.random.cov.dense() for _ in self.blocks]).as_explicit()
        Z = self.structured_random_design(resolved.name)
        return woodbury_shrinkage(G, Z, self.residual_precision())

    def blup_mean(self, term: str | None = None) -> Any:
        """Return the joint BLUP for the random effects in this set."""
        resolved = self._term(term)
        residual = self.response() - self.mean()
        return self.shrinkage(resolved.name) * residual

    def blup_covariance(self, term: str | None = None) -> Any:
        """Return the joint conditional covariance for this set using Woodbury."""
        resolved = self._term(term)
        G = sp.BlockDiagMatrix(*[resolved.random.cov.dense() for _ in self.blocks]).as_explicit()
        Z = self.structured_random_design(resolved.name)
        return woodbury_covariance(G, Z, self.residual_precision())
