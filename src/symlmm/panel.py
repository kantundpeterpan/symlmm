"""Panel view over one factor and one within-block position axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import sympy as sp  # type: ignore[import-untyped]

from .blocks import Block
from .errors import NotDefined
from .specs import Factor, Position


@dataclass(frozen=True)
class PanelGrid:
    """Resolved level and position codes for a panel."""

    model: Any
    factor: Factor
    position: Position | None
    _position_codes: list[int] | None = field(default=None, init=False, repr=False, compare=False)

    def level_codes(self) -> list[int]:
        """Return factor level codes for every observation."""
        if not self.model._dense:
            raise NotDefined("panel grids require dense mode")
        return cast(list[int], self.model._dense_group_codes(self.factor))

    def _require_position(self) -> Position:
        """Return the panel position or raise `NotDefined`."""
        if self.position is None:
            raise NotDefined("panel view requires a position axis")
        return self.position

    def position_codes(self) -> list[int]:
        """Return position codes for every observation."""
        position = self._require_position()
        if not self.model._dense:
            raise NotDefined("panel grids require dense mode")
        cached = self._position_codes
        if cached is None:
            cached = cast(
                list[int],
                self.model._dense_position_codes_for(self.factor, position),
            )
            object.__setattr__(self, "_position_codes", cached)
        return cached

    def levels(self) -> list[int]:
        """Return sorted level codes."""
        return sorted(set(self.level_codes()))

    def positions(self) -> list[int]:
        """Return sorted position codes."""
        return sorted(set(self.position_codes()))

    def rows_for_level(self, level: Any) -> list[int]:
        """Return block-local rows ordered by position code."""
        rows = [row for row, code in enumerate(self.level_codes()) if code == level]
        return sorted(rows, key=lambda row: self.position_codes()[row])

    def row(self, level: Any, position: Any) -> int:
        """Return the block-local row for a level and position."""
        ordered = self.rows_for_level(level)
        target = next(
            (row for row in ordered if self.position_codes()[row] == position),
            None,
        )
        if target is None:
            raise NotDefined(f"position {position!r} is not present for level {level!r}")
        block_rows = [row for row, code in enumerate(self.level_codes()) if code == level]
        return block_rows.index(target)

    def is_balanced(self) -> bool:
        """Return whether every level has the same position-code count."""
        level_codes = self.level_codes()
        position_codes = self.position_codes()
        per_level: dict[int, list[int]] = {}
        for level, position in zip(level_codes, position_codes, strict=True):
            per_level.setdefault(level, []).append(position)
        if not per_level:
            return True
        expected = set(next(iter(per_level.values())))
        for positions in per_level.values():
            if len(positions) != len(set(positions)):
                return False
            if set(positions) != expected:
                return False
        return True


class PanelView:
    """Panel-oriented moments for one factor and one position axis."""

    def __init__(
        self,
        model: Any,
        factor: Any,
        position: str | Position | None,
    ) -> None:
        self.model = model
        self.factor = model.factor(factor)
        if position is None:
            self.position: Position | None = None
        else:
            self.position = position if isinstance(position, Position) else Position(position)
        self.grid = PanelGrid(model, self.factor, self.position)

    def _require_position(self) -> Position:
        """Return the panel position or raise `NotDefined`."""
        if self.position is None:
            raise NotDefined("panel view requires a position axis")
        return self.position

    def level_count(self) -> Any:
        """Return the number of factor levels."""
        if self.model._dense:
            return len(self.grid.levels())
        return self.model.levels(self.factor)

    def position_count(self) -> Any:
        """Return the number of positions in a balanced panel."""
        self._require_position()
        if not self.model._dense:
            return sp.Symbol("T", positive=True, integer=True)
        if not self.grid.is_balanced():
            raise NotDefined("panel positions require a balanced panel")
        return len(self.grid.positions())

    def response_matrix(self) -> Any:
        """Return the balanced-panel response matrix Y."""
        self._require_position()
        return sp.MatrixSymbol("Y", self.level_count(), self.position_count())

    def response(self, level: Any) -> Any:
        """Return the response row for a factor level."""
        return self.response_matrix()[level, :]

    def response_column(self, position: Any) -> Any:
        """Return the response column for a position."""
        self._require_position()
        return self.response_matrix()[:, position]

    def position_code(self, i: Any) -> Any:
        """Return the position code for observation row `i`."""
        self._require_position()
        if self.model._dense:
            return self.grid.position_codes()[int(i)]
        return self.model._occasion_code(i, self.factor.name)

    def entry_mean(self, level: Any, position: Any) -> Any:
        """Return E[Y_level,position]."""
        self._require_position()
        row = self.grid.row(level, position)
        return Block(self.model, self.factor, level).mean()[row, 0]

    def entry_covariance(self, level: Any, p1: Any, p2: Any) -> Any:
        """Return Cov(Y_level,p1, Y_level,p2)."""
        self._require_position()
        row1 = self.grid.row(level, p1)
        row2 = self.grid.row(level, p2)
        return Block(self.model, self.factor, level).covariance()[row1, row2]

    def entry_cross_covariance(self, level1: Any, p1: Any, level2: Any, p2: Any) -> Any:
        """Return Cov(Y_level1,p1, Y_level2,p2)."""
        self._require_position()
        row1 = self.grid.row(level1, p1)
        row2 = self.grid.row(level2, p2)
        return Block(self.model, self.factor, level1).cross_covariance(
            Block(self.model, self.factor, level2)
        )[row1, row2]

    def position_mean(self, position: Any) -> Any:
        """Return E[Y_{.,position}]."""
        self._require_position()
        self.position_count()
        count = self.level_count()
        if not isinstance(count, int):
            raise NotDefined("position means require a concrete level count")
        return sp.Matrix([self.entry_mean(level, position) for level in range(count)])

    def position_covariance(self, p1: Any, p2: Any) -> Any:
        """Return Cov(Y_{.,p1}, Y_{.,p2})."""
        self._require_position()
        self.position_count()
        count = self.level_count()
        if not isinstance(count, int):
            raise NotDefined("position covariance requires a concrete level count")
        self.model._require_single_grouping_factor()
        return sp.diag(*[self.entry_covariance(level, p1, p2) for level in range(count)])

    def symbols(self) -> Any:
        """Return panel blackboard symbols."""
        return PanelAdapter(self).symbols()


class PanelAdapter:
    """Expose the legacy panel model surface used by blackboard.py."""

    def __init__(self, panel: PanelView) -> None:
        """Wrap a panel view for legacy consumers."""
        self.panel = panel

    def __getattr__(self, name: str) -> Any:
        """Delegate legacy attributes to the underlying model."""
        return getattr(self.panel.model, name)

    @property
    def group_index(self) -> Any:
        """Return the panel factor's index."""
        return self.panel.factor.index

    @property
    def occasion_index(self) -> Any:
        """Return the panel position index, if a position axis exists."""
        if self.panel.position is None:
            return None
        return self.panel.position.index

    @property
    def primary_group(self) -> str:
        """Return the panel factor's name."""
        return cast(str, self.panel.factor.name)

    def _primary_random_term(self) -> Any:
        """Return the random-effects term for this panel's factor, if any."""
        for term in self.panel.model.random:
            if self.panel.model.factor(term.group.name) == self.panel.factor:
                return term
        return None

    def group_count(self) -> Any:
        """Return the number of levels in this panel's factor."""
        return self.panel.level_count()

    def group_size(self, level: Any) -> Any:
        """Return the size of a level block in this panel's factor."""
        return self.panel.model.block(self.panel.factor, level).size()

    def response_matrix(self) -> Any:
        """Return the balanced-panel response matrix."""
        return self.panel.response_matrix()

    def response(self, level: Any, position: Any = None) -> Any:
        """Return a response row or a single response entry."""
        if position is None:
            return self.panel.response(level)
        return self.panel.response_matrix()[level, position]

    def time_response(self, position: Any) -> Any:
        """Return the response column at a position."""
        return self.panel.response_column(position)

    def time_mean(self, position: Any) -> Any:
        """Return the mean of the response column at a position."""
        return self.panel.position_mean(position)

    def time_covariance(self, p1: Any, p2: Any) -> Any:
        """Return the covariance between two position columns."""
        return self.panel.position_covariance(p1, p2)

    def entry_mean(self, level: Any, position: Any) -> Any:
        """Return the mean of one panel entry."""
        return self.panel.entry_mean(level, position)

    def entry_covariance(self, level: Any, p1: Any, p2: Any) -> Any:
        """Return the covariance between two entries of one level."""
        return self.panel.entry_covariance(level, p1, p2)

    def entry_cross_covariance(self, level1: Any, p1: Any, level2: Any, p2: Any) -> Any:
        """Return the covariance between entries of two levels."""
        return self.panel.entry_cross_covariance(level1, p1, level2, p2)

    def _entry_row(self, level: Any, position: Any) -> int:
        """Return the block-local row for a level and position."""
        return self.panel.grid.row(level, position)

    def group_mean(self, level: Any) -> Any:
        """Return the mean of a level block."""
        return self.panel.model.block(self.panel.factor, level).mean()

    def group_covariance(self, level: Any) -> Any:
        """Return the covariance of a level block."""
        return self.panel.model.block(self.panel.factor, level).covariance()

    def group_cross_covariance(self, left: Any, right: Any) -> Any:
        """Return the cross-covariance of two level blocks."""
        return self.panel.model.block(self.panel.factor, left).cross_covariance(
            self.panel.model.block(self.panel.factor, right)
        )

    def fixed_design(self, level: Any) -> Any:
        """Return the fixed-effect design of a level block."""
        return self.panel.model.block(self.panel.factor, level).design()

    def random_design(self, level: Any) -> Any:
        """Return the panel factor's random-effect design for a level block."""
        block = self.panel.model.block(self.panel.factor, level)
        term = next(
            (item for item in self.panel.model.terms() if item.factor == self.panel.factor),
            None,
        )
        if term is None:
            return block.random_design()
        return block.random_design(term=term.name)

    def residual_covariance(self, level: Any) -> Any:
        """Return the residual covariance of a level block."""
        return self.panel.model.block(self.panel.factor, level).residual()

    def random_effect(self, level: Any) -> Any:
        """Return the random-effect vector of a level block."""
        return self.panel.model.block(self.panel.factor, level).random_effect()

    def symbols(self) -> Any:
        """Return blackboard symbols bound to this panel."""
        from .blackboard import symbols

        return symbols(self)
