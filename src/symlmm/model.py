"""Model base classes and index accessors.

Index model A (temporary): ``i`` is only the observation-row cursor. Group
levels ``g`` and within-group occasions ``k`` are distinct structural index
spaces; coordinates live on ``G x K`` as ``X[g, k]``. The full two-level
``(group, occasion)`` response index redesign is tracked in issue #1:
https://github.com/kantundpeterpan/symlmm/issues/1
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import sympy as sp  # type: ignore[import-untyped]

from .covariances import IID, ResidualCovariance
from .errors import FormArityError, ModelError, NotDefined
from .specs import Covariate, Crossed, Fixed, Group, Nested, RandomEffects
from .symbols import observation_index

if TYPE_CHECKING:
    from .forms import Form

GroupSpec = Group | Nested | Crossed

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


def _primary_definition(
    definition: GroupSpec,
    registered: Mapping[str, GroupSpec],
) -> GroupSpec:
    """Return the registered definition for `definition`, when present."""
    return registered.get(definition.name, definition)


def _occasion_sort_key(value: Any, position: int) -> tuple[Any, ...]:
    """Return a total ordering key for a derived-occasion coordinate value."""
    try:
        return (0, sp.default_sort_key(sp.sympify(value)), position)
    except (sp.SympifyError, TypeError):  # fmt: skip
        return (1, str(value), position)


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
        """Return the group-level index for row `i` and factor `factor`."""
        spec = self._group_spec(factor)
        if self._dense:
            return self._dense_group_codes(spec)[int(i)]
        label = spec.label.replace(":", "_")
        return sp.IndexedBase(f"g_{label}", shape=(self.n,))[i]

    def occasion(self, i: Any, factor: str) -> Any:
        """Return the within-group occasion index for row `i` and factor `factor`."""
        spec = self._group_spec(factor)
        if self._dense:
            return self._dense_occasion_codes(spec)[int(i)]
        label = spec.label.replace(":", "_")
        return sp.IndexedBase(f"k_{label}", shape=(self.n,))[i]

    def coordinate(
        self,
        i: Any,
        name: str | Covariate | None = None,
        within: str | None = None,
    ) -> Any:
        """Return a row coordinate, or ``X[group(i), occasion(i)]`` when within."""
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
        return sp.IndexedBase(selected_name)[self.group(i, within), self.occasion(i, within)]

    def _dense_group_codes(self, spec: GroupSpec) -> list[int]:
        """Return integer codes for the composite label of `spec`."""
        cached = self._group_cache.get(spec.label)
        if cached is not None:
            return cached
        raw: list[Any]
        if isinstance(spec, Nested):
            if spec.label in self._observations:
                raw = list(self._observation_column(spec.label))
            else:
                left = self._observation_column(spec.within)
                right = self._observation_column(spec.child)
                raw = [f"{a}:{b}" for a, b in zip(left, right, strict=True)]
        elif isinstance(spec, Crossed):
            if spec.label in self._observations:
                raw = list(self._observation_column(spec.label))
            else:
                left = self._observation_column(spec.left)
                right = self._observation_column(spec.right)
                raw = [f"{a}:{b}" for a, b in zip(left, right, strict=True)]
        else:
            raw = list(self._observation_column(spec.name))
        codes = {label: code for code, label in enumerate(sorted(set(raw), key=str))}
        result = [codes[label] for label in raw]
        self._group_cache[spec.label] = result
        return result

    def _occasion_column(self, factor: str) -> str | None:
        """Return the explicit occasion column for `factor`, if configured."""
        if isinstance(self.occasion_names, Mapping):
            return self.occasion_names.get(factor)
        return self.occasion_names

    def _within_symbol(self, spec: GroupSpec) -> sp.Symbol:
        """Return the cached common-level symbol for `spec`."""
        key = spec.label
        symbol = self._within_symbols.get(key)
        if symbol is None:
            symbol = sp.Symbol(f"g_{key.replace(':', '_')}", integer=True)
            self._within_symbols[key] = symbol
        return symbol

    def _dense_occasion_codes(self, spec: GroupSpec) -> list[int]:
        """Return integer occasion codes within each group for `spec`.

        Derived occasion codes are ordered symbolically when possible and fall
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

    def _nested_coordinate_value(self, i: Any, name: str, spec: GroupSpec) -> Any:
        """Return a dense coordinate validated as constant within group/occasion."""
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
        return mapping[(self.group(i, spec.name), self.occasion(i, spec.name))]

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
        registered: dict[str, GroupSpec] = {}

        def register(spec: GroupSpec) -> None:
            existing = registered.get(spec.name)
            if existing is None:
                registered[spec.name] = spec
            elif existing != spec:
                raise ModelError(
                    f"conflicting grouping-factor specs for {spec.name!r}: "
                    f"{existing!r} and {spec!r}"
                )

        for spec in groups or ():
            register(spec)
        for term in self.random:
            registered.setdefault(term.group.name, term.group)
        residual_group = _residual_group(residual)
        if residual_group is not None:
            registered.setdefault(residual_group.name, residual_group)
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

        primary: GroupSpec | None
        if self.random:
            primary = _primary_definition(self.random[0].group, registered)
        elif residual_group is not None:
            primary = _primary_definition(residual_group, registered)
        elif registered:
            primary = next(iter(registered.values()))
        else:
            primary = None

        group_index: sp.Symbol | None = None if primary is None else primary.index
        occasion_definition = self._canonical_covariate(coordinate_name, residual_covariate)
        occasion_index: sp.Symbol | None = (
            None if occasion_definition is None else occasion_definition.index
        )
        if group_index is not None and occasion_index is not None and group_index == occasion_index:
            raise ModelError("group and occasion indices must be distinct")

        self.primary_group_definition = primary
        self.occasion_definition = occasion_definition
        self.group_index = group_index
        self.occasion_index = occasion_index

        super().__init__(
            n=n if n is not None else sp.Symbol("n", positive=True, integer=True),
            groups=tuple(registered.values()),
            coordinate=coordinate_name,
            occasion=occasion,
            covariates=tuple(covariate.name for covariate in fixed.covariates()),
        )

    def _canonical_covariate(
        self,
        coordinate_name: str | None,
        residual_covariate: Covariate | None,
    ) -> Covariate | None:
        """Return the covariate definition for the model's canonical coordinate."""
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
        """Return the primary grouping factor for canonical group indexing."""
        definition = self.primary_group_definition
        if definition is None:
            return None
        return definition.name

    def _require_group_index(self) -> str:
        """Return the primary grouping-factor name or raise `NotDefined`."""
        factor = self.primary_group
        if self.group_index is None or factor is None:
            raise NotDefined("canonical group indexing requires a grouping factor")
        return factor

    def _require_occasion_index(self) -> None:
        """Raise `NotDefined` when canonical occasion indexing is unavailable."""
        if self.occasion_index is None:
            raise NotDefined("canonical occasion indexing requires a coordinate")

    def _group_rows(self, group: Any) -> list[int]:
        """Return dense observation rows belonging to `group`."""
        factor = self._require_group_index()
        if not self._dense:
            raise ModelError("group rows are available only in dense mode")
        return [k for k in range(self.n) if self.group(k, factor) == group]

    def group_count(self) -> Any:
        """Return the number of groups (symbolic or dense)."""
        factor = self._require_group_index()
        if self._dense:
            return len(set(self._dense_group_codes(self._group_spec(factor))))
        return sp.Symbol("m", positive=True, integer=True)

    def occasion_count(self) -> Any:
        """Return the number of occasions for a balanced panel."""
        self._require_occasion_index()
        factor = self._require_group_index()
        if self._dense:
            groups = set(self._dense_group_codes(self._group_spec(factor)))
            counts = {len(self._group_rows(group)) for group in groups}
            if len(counts) != 1:
                raise NotDefined("occasion vectors require a balanced panel")
            return counts.pop()
        return sp.Symbol("T", positive=True, integer=True)

    def group_size(self, group: Any) -> Any:
        """Return the number of observations in `group`."""
        self._require_group_index()
        if self._dense:
            return len(self._group_rows(group))
        from .canonical import group_size_symbol

        return group_size_symbol(group)

    def response(self, group: Any, occasion: Any = None) -> Any:
        """Return Y_group (vector) or Y_group,occasion (scalar)."""
        self._require_group_index()
        if occasion is not None:
            self._require_occasion_index()
        from .canonical import group_entry, group_response

        vector = group_response(group, self.group_size(group))
        if occasion is None:
            return vector
        return group_entry(group, self.group_size(group), occasion)

    def response_matrix(self) -> Any:
        """Return the balanced-panel response matrix Y (groups x occasions)."""
        self._require_group_index()
        self._require_occasion_index()
        return sp.MatrixSymbol("Y", self.group_count(), self.occasion_count())

    def time_response(self, occasion: Any) -> Any:
        """Return the occasion column vector Y[:, occasion]."""
        return self.response_matrix()[:, occasion]

    def fixed_design(self, group: Any) -> Any:
        """Return X_group."""
        self._require_group_index()
        from .canonical import group_mean

        if self._dense:
            rows = self._group_rows(group)
            return self.matrix_x()[rows, :]
        return group_mean(group, self.group_size(group), len(self.fixed.coefficients()))

    def random_design(self, group: Any) -> Any:
        """Return Z_group."""
        self._require_group_index()
        from .canonical import group_random_design

        q = sum(term.dim for term in self.random)
        if self._dense:
            rows = self._group_rows(group)
            return self.matrix_z()[rows, :]
        return group_random_design(group, self.group_size(group), q)

    def residual_covariance(self, group: Any) -> Any:
        """Return R_group."""
        self._require_group_index()
        from .canonical import group_residual

        if self._dense:
            rows = self._group_rows(group)
            return self.matrix_r()[rows, :][:, rows]
        return group_residual(group, self.group_size(group))

    def random_effect(self, group: Any) -> Any:
        """Return b_group."""
        self._require_group_index()
        from .canonical import group_coefficients

        q = sum(term.dim for term in self.random)
        return group_coefficients(group, q)

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

    def _random_term_entry(self, term: RandomEffects, i: Any, j: Any) -> Any:
        """Return the structured contribution of one random-effects block."""
        zi = term.z(self, i)
        zj = term.z(self, j)
        value = (zi.T * term.cov.dense() * zj)[0, 0]
        return value * sp.KroneckerDelta(
            self.group(i, term.group.name), self.group(j, term.group.name)
        )

    def _apply_within(self, expr: Any, i: Any, j: Any, within: str) -> Any:
        """Resolve a factor's delta by placing both rows in one common level."""
        spec = self._group_spec(within)
        common = self._within_symbol(spec)
        return expr.subs({self.group(i, within): common, self.group(j, within): common})

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
