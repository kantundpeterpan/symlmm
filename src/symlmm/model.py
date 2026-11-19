"""Model base classes and index accessors.

Row-level accessors address the stacked observation vector. Blocks, panels, and
positions provide the factor-first views built on top of those rows.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import sympy as sp  # type: ignore[import-untyped]

from .covariances import IID, ResidualCovariance
from .errors import FormArityError, ModelError, NotDefined
from .factors import FactorResolver
from .specs import (
    Covariate,
    Crossed,
    Factor,
    FactorSpec,
    Fixed,
    Group,
    Nested,
    Position,
    RandomEffects,
)
from .symbols import observation_index

if TYPE_CHECKING:
    from .forms import Form

GroupSpec = FactorSpec

__all__ = ["IID", "LMM", "IndexedModel"]


def _residual_group(residual: ResidualCovariance) -> GroupSpec | None:
    """Return the residual covariance's grouping-factor definition, if any."""
    value = getattr(residual, "group", None)
    if value is None:
        return None
    if isinstance(value, (Group, Nested, Crossed)):
        return value
    if isinstance(value, str):
        return Group(value)
    return None


def _residual_covariate(residual: ResidualCovariance) -> Covariate | None:
    """Return the residual covariance's coordinate definition, if any."""
    value = getattr(residual, "coordinate", None)
    if value is None:
        return None
    if isinstance(value, Covariate):
        return value
    if isinstance(value, str):
        return Covariate(value)
    return None


def _occasion_sort_key(value: Any, position: int) -> tuple[Any, ...]:
    """Return a total ordering key for a derived position coordinate value."""
    try:
        return (0, sp.default_sort_key(sp.sympify(value)), position)
    except (sp.SympifyError, TypeError):  # fmt: skip
        return (1, str(value), position)


def _reject_free_symbols(values: Sequence[Any], column: str) -> None:
    """Reject derived coordinates whose values contain free symbols."""
    for value in values:
        if isinstance(value, sp.Basic) and value.free_symbols:
            raise ModelError(
                f"coordinate column {column!r} contains free symbols; "
                "substitute concrete values or supply an explicit position column"
            )


class IndexedModel:
    """Base class for models over a stacked observation vector."""

    _dense: bool = False
    _observations: Mapping[str, Sequence[Any]]

    def __init__(
        self,
        *,
        n: Any,
        groups: Sequence[GroupSpec] = (),
        coordinate: str | None = None,
        occasion: str | Mapping[str, str] | None = None,
        covariates: Sequence[str] = (),
    ) -> None:
        self.n = n
        self.groups = tuple(groups)
        self._factor_resolver = FactorResolver(self.groups)
        self._factors = self._factor_resolver.resolve_all()
        self.coordinate_name = coordinate
        self.occasion_names = occasion
        self.covariates = tuple(covariates)
        self._coordinate_cache: dict[tuple[str, str], dict[tuple[int, int], Any]] = {}
        self._group_cache: dict[str, list[int]] = {}
        self._occasion_cache: dict[str, list[int]] = {}
        self._within_symbols: dict[str, sp.Symbol] = {}
        self._i = observation_index("i")
        self._j = observation_index("j")

    def indices(self) -> tuple[Any, Any]:
        """Return the two observation indices used by entry queries."""
        return self._i, self._j

    def factor(self, ref: Any) -> Factor:
        """Return a resolved factor by name, path, declaration, or factor."""
        return self._factor_resolver.resolve(ref)

    def factors(self) -> tuple[Factor, ...]:
        """Return all resolved factors in declaration order."""
        return self._factors

    def level_code(self, factor: Any, i: Any) -> Any:
        """Return the factor level code for observation row `i`."""
        resolved = self.factor(factor)
        if self._dense:
            return self._dense_group_codes(resolved)[int(i)]
        label = resolved.label.replace(":", "_")
        return sp.IndexedBase(f"g_{label}", shape=(self.n,))[i]

    def _group_spec(self, factor: str) -> Factor:
        try:
            return self._factor_resolver.resolve(factor)
        except ModelError as exc:
            raise ModelError(f"unknown grouping factor: {factor!r}") from exc

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
        """Deprecated: use level_code(factor, i)."""
        warnings.warn(
            "group is deprecated; use level_code(factor, i)",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.level_code(factor, i)

    def _occasion_code(self, i: Any, factor: str) -> Any:
        """Return the within-group position index for row `i` and factor `factor`."""
        spec = self._group_spec(factor)
        if self._dense:
            return self._dense_occasion_codes(spec)[int(i)]
        label = spec.label.replace(":", "_")
        return sp.IndexedBase(f"k_{label}", shape=(self.n,))[i]

    def occasion(self, i: Any, factor: str) -> Any:
        """Deprecated: use panel(...).position_code(i)."""
        warnings.warn(
            "occasion is deprecated; use panel(...).position_code(i)",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._occasion_code(i, factor)

    def coordinate(
        self,
        i: Any,
        name: str | Covariate | None = None,
        within: str | None = None,
    ) -> Any:
        """Return a row coordinate, or a two-level ``X[level, position]`` when within."""
        selected = self.coordinate_name if name is None else name
        if selected is None:
            raise ModelError("no coordinate name; pass coordinate=... or coordinate(i, name)")
        selected_name = selected.name if isinstance(selected, Covariate) else selected
        if within is None:
            if self._dense:
                return self._observation_value(i, selected_name)
            return sp.IndexedBase(selected_name, shape=(self.n,))[i]
        spec = self._group_spec(within)
        if self._dense:
            return self._nested_coordinate_value(i, selected_name, spec)
        return sp.IndexedBase(selected_name)[
            self.level_code(spec.name, i), self._occasion_code(i, spec.name)
        ]

    def _dense_group_codes(self, factor: Factor) -> list[int]:
        """Return integer codes for a resolved factor's component path."""
        cached = self._group_cache.get(factor.label)
        if cached is not None:
            return cached
        raw: list[Any]
        composite = ":".join(factor.components)
        if composite in self._observations:
            raw = list(self._observation_column(composite))
        else:
            columns = [list(self._observation_column(name)) for name in factor.components]
            raw = [
                ":".join(str(value) for value in values) for values in zip(*columns, strict=True)
            ]
        codes = {label: code for code, label in enumerate(sorted(set(raw), key=str))}
        result = [codes[label] for label in raw]
        self._group_cache[factor.label] = result
        return result

    def _occasion_column(self, factor: str) -> str | None:
        """Return the explicit position column for `factor`, if configured."""
        if isinstance(self.occasion_names, Mapping):
            return self.occasion_names.get(factor)
        return self.occasion_names

    def _within_symbol(self, spec: Factor) -> sp.Symbol:
        """Return the cached common-level symbol for `spec`."""
        key = spec.label
        symbol = self._within_symbols.get(key)
        if symbol is None:
            symbol = sp.Symbol(f"g_{key.replace(':', '_')}", integer=True)
            self._within_symbols[key] = symbol
        return symbol

    def _dense_occasion_codes(self, spec: Factor) -> list[int]:
        """Return integer position codes within each group for `spec`.

        Derived position codes are ordered symbolically when possible and fall
        back to string ordering otherwise. Numeric ``1`` and ``1.0`` are
        distinct coordinate values for nesting validation.
        """
        cached = self._occasion_cache.get(spec.label)
        if cached is not None:
            return cached
        column = self._occasion_column(spec.name)
        groups = self._dense_group_codes(spec)
        result = [0] * self.n
        if column is not None:
            raw = list(self._observation_column(column))
            per_group: dict[int, dict[Any, int]] = {}
            for position, group_code in enumerate(groups):
                mapping = per_group.setdefault(group_code, {})
                value = raw[position]
                if value not in mapping:
                    mapping[value] = len(mapping)
                result[position] = mapping[value]
        else:
            if self.coordinate_name is not None:
                raw = list(self._observation_column(self.coordinate_name))
            else:
                raw = list(range(self.n))
            _reject_free_symbols(raw, self.coordinate_name or "position")
            for group_code in sorted(set(groups)):
                positions = [k for k, value in enumerate(groups) if value == group_code]
                ordered = sorted(positions, key=lambda k: _occasion_sort_key(raw[k], k))
                code = -1
                previous: Any = object()
                for position in ordered:
                    value = raw[position]
                    if value != previous:
                        code += 1
                        previous = value
                    result[position] = code
        self._occasion_cache[spec.label] = result
        return result

    def _dense_position_codes_for(self, factor: Any, position: Any) -> list[int]:
        """Return dense position codes for a factor and position definition."""
        resolved = self.factor(factor)
        raw = list(self._observation_column(position.column or position.name))
        groups = self._dense_group_codes(resolved)
        result = [0] * self.n
        if position.column is None:
            _reject_free_symbols(raw, position.name)
        if position.column is not None:
            per_group: dict[int, dict[Any, int]] = {}
            for row, group_code in enumerate(groups):
                mapping = per_group.setdefault(group_code, {})
                value = raw[row]
                if value not in mapping:
                    mapping[value] = len(mapping)
                result[row] = mapping[value]
            return result
        for group_code in sorted(set(groups)):
            rows = [row for row, code in enumerate(groups) if code == group_code]
            ordered = sorted(rows, key=lambda row: _occasion_sort_key(raw[row], row))
            code = -1
            previous: Any = object()
            for row in ordered:
                value = raw[row]
                if value != previous:
                    code += 1
                    previous = value
                result[row] = code
        return result

    def _nested_coordinate_value(self, i: Any, name: str, spec: Factor) -> Any:
        """Return a dense coordinate validated as constant within group/position."""
        key = (name, spec.label)
        mapping = self._coordinate_cache.get(key)
        if mapping is None:
            mapping = {}
            groups = self._dense_group_codes(spec)
            occasions = self._dense_occasion_codes(spec)
            for k in range(self.n):
                pair = (groups[k], occasions[k])
                value = self._observation_value(k, name)
                previous = mapping.get(pair, value)
                if previous != value:
                    raise ModelError(
                        f"coordinate {name!r} is not constant within {spec.label!r}: "
                        f"({pair[0]}, {pair[1]}) has {previous!r} and {value!r}"
                    )
                mapping[pair] = value
            self._coordinate_cache[key] = mapping
        return mapping[(self.level_code(spec.name, i), self._occasion_code(i, spec.name))]

    def covariate(self, i: Any, name: str | Covariate) -> Any:
        """Return a covariate value for observation `i`."""
        selected = name.name if isinstance(name, Covariate) else name
        if self._dense:
            return self._observation_value(i, selected)
        return sp.IndexedBase(selected, shape=(self.n,))[i]

    def design(self, i: Any) -> Any:
        """Return the fixed-effect design row `[coordinate, covariates...]`."""
        terms = [] if self.coordinate_name is None else [self.coordinate(i)]
        terms.extend(
            self.covariate(i, name) for name in self.covariates if name != self.coordinate_name
        )
        return sp.Matrix(terms) if terms else sp.zeros(0, 1)


class LMM(IndexedModel):
    """Linear mixed model assembled from fixed, random, and residual terms."""

    def __init__(
        self,
        *,
        fixed: Fixed,
        random: Iterable[RandomEffects],
        residual: ResidualCovariance,
        groups: Sequence[GroupSpec] | None = None,
        n: Any | None = None,
        coordinate: str | None = None,
        occasion: str | Mapping[str, str] | None = None,
    ) -> None:
        self.fixed = fixed
        self.random = tuple(random)
        self.residual = residual
        seen_groups: set[str] = set()
        for term in self.random:
            group_name = term.group.name
            if group_name in seen_groups:
                raise ModelError(
                    f"multiple RandomEffects blocks for group {group_name!r}; "
                    "combine intercepts and slopes into one block"
                )
            seen_groups.add(group_name)
        specs: list[FactorSpec] = []
        declared: dict[str, FactorSpec] = {}
        for spec in groups or ():
            existing = declared.get(spec.name)
            if existing is None:
                declared[spec.name] = spec
                specs.append(spec)
            elif existing != spec:
                specs.append(spec)

        def add_implicit(spec: FactorSpec) -> None:
            existing = declared.get(spec.name)
            if existing is None:
                declared[spec.name] = spec
                specs.append(spec)
            elif not isinstance(spec, Group) and existing != spec:
                specs.append(spec)

        for term in self.random:
            add_implicit(term.group)
        residual_group = _residual_group(residual)
        if residual_group is not None:
            add_implicit(residual_group)
        residual_covariate = _residual_covariate(residual)
        residual_coordinate = residual_covariate.name if residual_covariate is not None else None
        slope_coordinates = tuple(
            dict.fromkeys(slope.coordinate.name for term in self.random for slope in term.slopes)
        )
        if coordinate is not None:
            if residual_coordinate is not None and residual_coordinate != coordinate:
                raise ModelError(
                    f"coordinate {coordinate!r} conflicts with residual coordinate "
                    f"{residual_coordinate!r}"
                )
            coordinate_name = coordinate
        elif residual_coordinate is not None:
            coordinate_name = residual_coordinate
        elif len(slope_coordinates) == 1:
            coordinate_name = slope_coordinates[0]
        else:
            coordinate_name = None

        super().__init__(
            n=n if n is not None else sp.Symbol("n", positive=True, integer=True),
            groups=tuple(specs),
            coordinate=coordinate_name,
            occasion=occasion,
            covariates=tuple(covariate.name for covariate in fixed.covariates()),
        )

        primary: Factor | None
        if self.random:
            primary = self.factor(self.random[0].group.name)
        elif residual_group is not None:
            primary = self.factor(residual_group.name)
        elif self._factors:
            primary = self._factors[0]
        else:
            primary = None

        group_index: sp.Symbol | None = None if primary is None else primary.index
        occasion_definition = self._canonical_covariate(coordinate_name, residual_covariate)
        occasion_index: sp.Symbol | None = (
            None if occasion_definition is None else occasion_definition.index
        )
        if group_index is not None and occasion_index is not None and group_index == occasion_index:
            raise ModelError("group and position indices must be distinct")

        self.primary_group_definition = self._declaration(primary)
        self.occasion_definition = occasion_definition
        self.group_index = group_index
        self.occasion_index = occasion_index

    def _declaration(self, factor: Factor | None) -> FactorSpec | Factor | None:
        """Return the declaration backing a resolved factor, when present."""
        if factor is None:
            return None
        for spec in self.groups:
            if spec.name == factor.name:
                return spec
        return factor

    def _canonical_covariate(
        self,
        coordinate_name: str | None,
        residual_covariate: Covariate | None,
    ) -> Covariate | None:
        """Return the covariate definition for the model's coordinate."""
        if coordinate_name is None:
            return None
        if residual_covariate is not None and residual_covariate.name == coordinate_name:
            return residual_covariate
        for covariate in self.fixed.covariates():
            if covariate.name == coordinate_name:
                return covariate
        for term in self.random:
            for slope in term.slopes:
                if slope.coordinate.name == coordinate_name:
                    return slope.coordinate
        return Covariate(coordinate_name)

    def design(self, i: Any) -> Any:
        """Return the fixed-effect design row in ``Fixed.names()`` order."""
        terms = [] if self.fixed.intercept is None else [sp.Integer(1)]
        terms.extend(self.covariate(i, covariate.name) for covariate in self.fixed.covariates())
        return sp.Matrix(terms) if terms else sp.zeros(0, 1)

    def levels(self, factor: str) -> Any:
        """Return the number of distinct labels for a grouping factor."""
        spec = self._group_spec(factor)
        if self._dense:
            return len(set(self._dense_group_codes(spec)))
        label = spec.label.replace(":", "_")
        return sp.Symbol(f"m_{label}", positive=True, integer=True)

    def dimensions(self) -> tuple[Any, Any]:
        """Return observations and total random-coefficient dimension."""
        q = sum(self.levels(term.group.name) * term.dim for term in self.random)
        return self.n, q

    @property
    def primary_group(self) -> str | None:
        """Return the primary grouping factor for group indexing."""
        definition = self.primary_group_definition
        if definition is None:
            return None
        return definition.name

    def _require_group_index(self) -> str:
        """Return the primary grouping-factor name or raise `NotDefined`."""
        factor = self.primary_group
        if self.group_index is None or factor is None:
            raise NotDefined("group indexing requires a grouping factor")
        return factor

    def _require_occasion_index(self) -> Covariate:
        """Return the position definition or raise `NotDefined`."""
        definition = self.occasion_definition
        if self.occasion_index is None or definition is None:
            raise NotDefined("position indexing requires a coordinate")
        return definition

    def _group_rows(self, group: Any) -> list[int]:
        """Return dense observation rows belonging to `group` in position order."""
        factor = self._require_group_index()
        if not self._dense:
            raise ModelError("group rows are available only in dense mode")
        rows = [k for k in range(self.n) if self.level_code(factor, k) == group]
        if self.occasion_index is None:
            return rows
        codes = self._dense_occasion_codes(self._group_spec(factor))
        return sorted(rows, key=lambda row: codes[row])

    def _primary_random_term(self) -> RandomEffects | None:
        """Return the primary grouping factor's random-effects term, if any."""
        factor = self.primary_group
        if factor is None:
            return None
        for term in self.random:
            if term.group.name == factor:
                return term
        return None

    def _random_term_offset(self, term: RandomEffects) -> int:
        """Return the dense `matrix_z` column offset for `term`."""
        offset = 0
        for candidate in self.random:
            if candidate is term:
                return offset
            labels = set(self._dense_group_codes(self._group_spec(candidate.group.name)))
            offset += len(labels) * candidate.dim
        return offset

    def _require_single_grouping_factor(self) -> None:
        """Raise `NotDefined` when random terms use more than one factor."""
        groups = {term.group for term in self.random}
        if len(groups) > 1:
            raise NotDefined("group blocks require a single grouping factor")

    def _entry_row(self, group: Any, occasion: Any) -> Any:
        """Return the group-block row for a position within `group`."""
        if not self._dense:
            return occasion
        factor = self._require_group_index()
        codes = self._dense_occasion_codes(self._group_spec(factor))
        for position, row in enumerate(self._group_rows(group)):
            if codes[row] == occasion:
                return position
        raise NotDefined(f"position {occasion!r} is not present in group {group!r}")

    def terms(self) -> Any:
        """Return named random-effect terms in model order."""
        from .decomposition import terms

        return terms(self)

    def covariance_terms(self, a: Any, b: Any) -> Any:
        """Return per-term contributions to Cov(Y_a, Y_b)."""
        from .blocks import Block
        from .decomposition import covariance_terms

        if isinstance(a, Block) and isinstance(b, Block):
            return a.covariance_terms(b)
        if isinstance(a, Block) or isinstance(b, Block):
            raise ModelError("covariance_terms requires two rows or two blocks")
        return covariance_terms(self, a, b)

    def partial_covariance(
        self,
        a: Any,
        b: Any,
        *,
        exclude: tuple[str, ...] = (),
    ) -> Any:
        """Return covariance with named term contributions removed."""
        from .blocks import Block
        from .decomposition import partial_covariance

        if isinstance(a, Block) and isinstance(b, Block):
            return a.partial_covariance(b, exclude=exclude)
        if isinstance(a, Block) or isinstance(b, Block):
            raise ModelError("partial_covariance requires two rows or two blocks")
        return partial_covariance(self, a, b, exclude=exclude)

    def partial_correlation(
        self,
        a: Any,
        b: Any,
        *,
        exclude: tuple[str, ...] = (),
    ) -> Any:
        """Return partial correlation for scalar entries."""
        from .blocks import Block
        from .decomposition import partial_correlation

        if isinstance(a, Block) or isinstance(b, Block):
            raise ModelError("partial_correlation is defined for scalar entries only")
        return partial_correlation(self, a, b, exclude=exclude)

    def block(self, factor: Any, level: Any) -> Any:
        """Return the observation block for a factor level."""
        from .blocks import Block

        return Block(self, self.factor(factor), level)

    def block_mean(self, factor: Any, level: Any) -> Any:
        """Return E[Y_block]."""
        return self.block(factor, level).mean()

    def block_covariance(self, factor: Any, level: Any) -> Any:
        """Return Cov(Y_block)."""
        return self.block(factor, level).covariance()

    def block_cross_covariance(
        self,
        factor1: Any,
        level1: Any,
        factor2: Any,
        level2: Any,
    ) -> Any:
        """Return Cov(Y_block1, Y_block2)."""
        return self.block(factor1, level1).cross_covariance(self.block(factor2, level2))

    def panel(self, factor: Any = None, position: Any = None) -> Any:
        """Return a panel view for a factor and optional position axis."""
        from .panel import PanelView

        selected_factor = factor if factor is not None else self.primary_group
        if selected_factor is None:
            raise NotDefined("panel view requires a factor")
        selected_position: Any
        if position is None:
            if self.coordinate_name is None:
                selected_position = None
            else:
                selected_position = Position(
                    self.coordinate_name,
                    index=self.occasion_index,
                )
        elif isinstance(position, str) and position == self.coordinate_name:
            selected_position = Position(position, index=self.occasion_index)
        else:
            selected_position = position
        return PanelView(self, selected_factor, selected_position)

    def _deprecated(self, name: str) -> Any:
        """Return the default panel view and emit a deprecation warning."""
        warnings.warn(
            f"{name} is deprecated; use panel(...)",
            DeprecationWarning,
            stacklevel=3,
        )
        return self.panel()

    def response_matrix(self) -> Any:
        """Deprecated: use panel(...).response_matrix()."""
        return self._deprecated("response_matrix").response_matrix()

    def response(self, group: Any, occasion: Any = None) -> Any:
        """Deprecated: use panel(...).response(...)."""
        panel = self._deprecated("response")
        if occasion is None:
            return panel.response(group)
        return panel.response_matrix()[group, occasion]

    def group_count(self) -> Any:
        """Deprecated: use panel(...).level_count()."""
        return self._deprecated("group_count").level_count()

    def occasion_count(self) -> Any:
        """Deprecated: use panel(...).position_count()."""
        return self._deprecated("occasion_count").position_count()

    def group_size(self, group: Any) -> Any:
        """Deprecated: use block(...).size()."""
        panel = self._deprecated("group_size")
        return panel.model.block(panel.factor, group).size()

    def group_mean(self, group: Any) -> Any:
        """Deprecated: use block(...).mean()."""
        panel = self._deprecated("group_mean")
        return panel.model.block(panel.factor, group).mean()

    def group_covariance(self, group: Any) -> Any:
        """Deprecated: use block(...).covariance()."""
        panel = self._deprecated("group_covariance")
        return panel.model.block(panel.factor, group).covariance()

    def group_cross_covariance(self, group: Any, other: Any) -> Any:
        """Deprecated: use block(...).cross_covariance(...)."""
        panel = self._deprecated("group_cross_covariance")
        return panel.model.block(panel.factor, group).cross_covariance(
            panel.model.block(panel.factor, other)
        )

    def entry_mean(self, group: Any, occasion: Any) -> Any:
        """Deprecated: use panel(...).entry_mean(...)."""
        return self._deprecated("entry_mean").entry_mean(group, occasion)

    def entry_covariance(self, group: Any, t1: Any, t2: Any) -> Any:
        """Deprecated: use panel(...).entry_covariance(...)."""
        return self._deprecated("entry_covariance").entry_covariance(group, t1, t2)

    def entry_cross_covariance(self, group: Any, t1: Any, other: Any, t2: Any) -> Any:
        """Deprecated: use panel(...).entry_cross_covariance(...)."""
        return self._deprecated("entry_cross_covariance").entry_cross_covariance(
            group, t1, other, t2
        )

    def time_response(self, position: Any) -> Any:
        """Deprecated: use panel(...).response_column(...)."""
        return self._deprecated("time_response").response_column(position)

    def time_mean(self, position: Any) -> Any:
        """Deprecated: use panel(...).position_mean(...)."""
        return self._deprecated("time_mean").position_mean(position)

    def time_covariance(self, p1: Any, p2: Any) -> Any:
        """Deprecated: use panel(...).position_covariance(...)."""
        return self._deprecated("time_covariance").position_covariance(p1, p2)

    def fixed_design(self, group: Any) -> Any:
        """Deprecated: use block(...).design()."""
        panel = self._deprecated("fixed_design")
        return panel.model.block(panel.factor, group).design()

    def random_design(self, group: Any) -> Any:
        """Deprecated: use block(...).random_design()."""
        panel = self._deprecated("random_design")
        return panel.model.block(panel.factor, group).random_design()

    def residual_covariance(self, group: Any) -> Any:
        """Deprecated: use block(...).residual()."""
        panel = self._deprecated("residual_covariance")
        return panel.model.block(panel.factor, group).residual()

    def random_effect(self, group: Any) -> Any:
        """Deprecated: use block(...).random_effect()."""
        panel = self._deprecated("random_effect")
        return panel.model.block(panel.factor, group).random_effect()

    def symbols(self) -> Any:
        """Deprecated: use panel(...).symbols()."""
        warnings.warn("symbols is deprecated; use panel(...)", DeprecationWarning, stacklevel=2)
        return self.panel().symbols()

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
        dense._coordinate_cache = {}
        dense._group_cache = {}
        dense._occasion_cache = {}
        dense._within_symbols = {}
        dense.n = lengths.pop()
        return dense

    def _column(self, name: str) -> Any:
        """Return a dense observation column as a matrix."""
        return sp.Matrix(self._observation_column(name))

    def matrix_x(self) -> Any:
        """Return the fixed-effect design matrix."""
        if not self._dense:
            return sp.MatrixSymbol("X", self.n, len(self.fixed.coefficients()))
        columns = []
        if self.fixed.intercept is not None:
            columns.append(sp.ones(self.n, 1))
        for covariate in self.fixed.covariates():
            columns.append(self._column(covariate.name))
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
            labels = self._dense_group_codes(self._group_spec(term.group.name))
            designs = [term.z(self, k) for k in range(self.n)]
            for level in sorted(set(labels)):
                for a in range(term.dim):
                    columns.append(
                        sp.Matrix(
                            [designs[k][a] if labels[k] == level else 0 for k in range(self.n)]
                        )
                    )
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
            labels = sorted(set(self._dense_group_codes(self._group_spec(term.group.name))))
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
        coefficients = sp.Matrix(self.fixed.coefficients())
        if coefficients.cols == 0:
            coefficients = sp.zeros(0, 1)
        return self.matrix_x() * coefficients

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

    def _random_term_entry(self, term: RandomEffects, i: Any, j: Any) -> Any:
        """Return the structured contribution of one random-effects block."""
        zi = term.z(self, i)
        zj = term.z(self, j)
        value = (zi.T * term.cov.dense() * zj)[0, 0]
        return value * sp.KroneckerDelta(
            self.level_code(term.group.name, i), self.level_code(term.group.name, j)
        )

    def _apply_within(self, expr: Any, i: Any, j: Any, within: str) -> Any:
        """Resolve a factor's delta by placing both rows in one common level."""
        spec = self._group_spec(within)
        common = self._within_symbol(spec)
        return expr.subs(
            {
                self.level_code(within, i): common,
                self.level_code(within, j): common,
            }
        )

    def covariance_entry(self, i: Any, j: Any, within: str | None = None) -> Any:
        """Return the structured sum-form covariance entry.

        ``within`` validates and resolves the named factor's delta in symbolic
        mode. Dense models keep their actual concrete level codes; ``within``
        cannot override them.
        """
        if within is not None:
            self._group_spec(within)
        total = self.residual.entry(i, j, self)
        for term in self.random:
            total += self._random_term_entry(term, i, j)
        if within is not None and not self._dense:
            total = self._apply_within(total, i, j, within)
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
        within: str | None = None,
    ) -> Any:
        """Return the marginal covariance in the requested form.

        ``within`` resolves a grouping factor's delta and is only supported for
        ``form='sum'`` entries.
        """
        from .moments import covariance

        return covariance(self, i, j, form, within)

    def variance(
        self,
        i: Any | None = None,
        *,
        form: str | Form | None = None,
        within: str | None = None,
    ) -> Any:
        """Return the marginal variance vector or entry in the requested form.

        ``within`` resolves a grouping factor's delta and is only supported for
        ``form='sum'`` entries.
        """
        from .moments import variance

        return variance(self, i, form, within)

    def correlation(
        self,
        i: Any | None = None,
        j: Any | None = None,
        *,
        form: str | Form | None = None,
        within: str | None = None,
    ) -> Any:
        """Return the marginal correlation in the requested form.

        ``within`` resolves a grouping factor's delta and is only supported for
        ``form='sum'`` entries.
        """
        from .moments import correlation

        return correlation(self, i, j, form, within)

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
        within: str | None = None,
    ) -> Any:
        """Return a raw, residual, or conditional variogram entry.

        ``within`` resolves a grouping factor's delta and is only supported for
        ``form='sum'`` entries.
        """
        from .variogram import variogram

        return variogram(self, i, j, kind, form, within)

    def covariogram(
        self,
        i: Any,
        j: Any,
        *,
        form: str | Form | None = None,
        within: str | None = None,
    ) -> Any:
        """Return the covariogram entry Cov(y_i, y_j).

        ``within`` resolves a grouping factor's delta and is only supported for
        ``form='sum'`` entries.
        """
        from .variogram import covariogram

        return covariogram(self, i, j, form, within)

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
        coordinate_slopes = [(term, slope) for term in self.random for slope in term.slopes]
        if coordinate_slopes and at is None:
            raise ModelError("icc requires at={'coordinate': value} or coordinate mapping")
        values: dict[str, Any] = {}
        if at is not None:
            coordinate_value = at.get("coordinate")
            if coordinate_value is not None and self.coordinate_name is not None:
                values[self.coordinate_name] = coordinate_value
            values.update(
                {
                    key: value
                    for key, value in at.items()
                    if key != "coordinate" and value is not None
                }
            )
        random_part = sp.Integer(0)
        for term in self.random:
            if self._dense:
                columns: list[Any] = [sp.Integer(1)] if term.intercept else []
                for slope in term.slopes:
                    coordinate_name = slope.coordinate.name
                    if coordinate_name not in values:
                        raise ModelError(f"icc requires a value for coordinate {coordinate_name!r}")
                    columns.append(values[coordinate_name])
                z = sp.Matrix(columns)
            else:
                z = term.z(self, 0)
                for slope in term.slopes:
                    coordinate_name = slope.coordinate.name
                    if coordinate_name not in values:
                        raise ModelError(f"icc requires a value for coordinate {coordinate_name!r}")
                    key = self.coordinate(
                        0,
                        coordinate_name,
                        within=term.group.name if slope.within else None,
                    )
                    z = z.subs({key: values[coordinate_name]})
            random_part += (z.T * term.cov.dense() * z)[0, 0]
        residual_part = self.residual.entry(0, 0, self)
        canonical = self.coordinate_name
        if canonical is not None and canonical in values and not self._dense:
            residual_part = residual_part.subs({self.coordinate(0, canonical): values[canonical]})
        value = random_part / (random_part + residual_part)
        if form.value == "sum":
            return value
        return DualForm(sum=value, matrix=value)
