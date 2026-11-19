# Factor-First Model API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the primary-group/occasion panel core with a factor-first model API, then layer `PanelView`, term decomposition, and partial covariance on top.

**Architecture:** Resolve `Group` / `Nested` / `Crossed` declarations into immutable `Factor` objects with full ancestry. Address observation sets as `Block(model, factor, level)`. Decompose covariance into named `Term` contributions. Keep panel behavior in a separate `PanelView` adapter; keep legacy `LMM` methods as deprecated wrappers.

**Tech Stack:** Python 3.14, SymPy, pytest, mypy (strict), ruff.

**Spec:** `docs/superpowers/specs/2026-11-18-symlmm-factor-first-design.md`

## Global Constraints

- Python 3.14; runtime dependency remains `sympy>=1.14.0`; no new runtime dependencies.
- Public functions/classes have docstrings.
- No new hardcoded `"time"` or `"occasion"` identifiers in the new API.
- Use `factor`, `level`, `block`, `node`, `position`, `coordinate`, `term`, `panel`.
- Every PR: description, test evidence, and spec link.
- CI commands:
  - `uv run pytest --cov=symlmm --cov-fail-under=90`
  - `uv run mypy src/symlmm`
  - `uv run ruff check`
  - `uv run ruff format --check`
- Branch naming: `feat/factor-foundation`, `feat/term-decomposition`, `feat/block-moments`, `feat/panel-view`, `feat/panel-symbols`, `chore/factor-first-vocab`.
- No direct commits to `main`; every workstream opens a PR.
- **Concurrency rule:** one implementer per worktree. Parallel workstreams use separate worktrees and branches. Never run two implementers in the same worktree. Integration merges are serialized.

## Object Dependency Graph

This graph is object-level, not file-level. An edge means the consumer needs the producer’s interface to be frozen.

```mermaid
graph TD
  FS[FactorSpec: Group | Nested | Crossed] --> FR[FactorResolver]
  FR --> F[Factor]
  F --> REG[FactorRegistry]
  F --> RS[RowSelector]
  RS --> B[Block]
  RE[RandomEffects + name] --> T[Term]
  F --> T
  T --> TC[TermContribution]
  B --> TC
  TC --> MA[MomentAggregate]
  B --> MA
  F --> PG[PanelGrid]
  P[Position] --> PG
  PG --> PV[PanelView]
  MA --> PM[PanelMoments]
  PV --> PM
  PV --> PS[PanelSymbols]
  B --> LA[LegacyAdapter]
  PV --> LA
```

### Interface ownership

| Object | Owner | Consumers | Frozen surface |
|--------|-------|-----------|----------------|
| `Factor` | Task 1 | Tasks 2–6 | `name`, `label`, `components`, `index`, `parent`, `kind` |
| `FactorResolver` | Task 1 | Task 3, Task 4 | `resolve(ref) -> Factor`, `resolve_all() -> tuple[Factor, ...]` |
| `Term` | Task 2 | Task 3b, Task 5 | `name`, `factor`, `random`, `index` |
| `Block` | Task 3a | Task 3b, Task 4, Task 5 | `model`, `factor`, `level`; `rows`, `size`, `mean`, `covariance`, `cross_covariance`, `design`, `random_design`, `residual`, `random_effect` |
| `Block` decomposition | Task 3b | Task 5 | `covariance_terms()`, `partial_covariance(other, exclude)` |
| `PanelGrid` / `PanelView` | Task 4 | Task 5 | `level_count`, `position_count`, `response_matrix`, `response`, `response_column`, `entry_*`, `position_*`, `position_code` |
| `PanelSymbols` | Task 5 | docs/tests | `PanelView.symbols()` via `PanelAdapter` |
| `LegacyAdapter` | Task 5 | tests | old `LMM` methods delegate with `DeprecationWarning` |

## Concurrency Waves

| Wave | Workstream | Objects | Depends on | Worktree |
|------|------------|---------|------------|----------|
| 0 | Foundation | `FactorSpec`, `Factor`, `FactorResolver`, `RandomEffects.name`, dense codes | — | `feat/factor-foundation` |
| 1 | Term algebra | `Term`, `TermContribution`, row-level decomposition | Wave 0 | `feat/term-decomposition` |
| 1 | Block core | `Block`, basic block moments | Wave 0 | `feat/block-moments` |
| 2 | Block decomposition | `Block.covariance_terms`, `Block.partial_covariance` | Term algebra + Block core | `feat/block-moments` (continue) |
| 2 | Panel view | `PanelGrid`, `PanelView` | Block core | `feat/panel-view` |
| 3 | Public wiring and legacy | `LMM.block`, `LMM.panel`, deprecated wrappers, blackboard | Block decomposition + Panel view | `feat/panel-symbols` |
| 4 | Vocabulary and docs | remove `canonical`, rename modules, docs | Public wiring | `chore/factor-first-vocab` |
| 5 | Integration | full suite, type check, lint | all | `main` merge queue |

Maximum safe concurrency: **two implementers** in wave 1, **two** in wave 2. Wave 3 is serial because it edits `model.py` and `blackboard.py` together.

---

## Task 1: Factor Foundation and Ancestry Resolution

**Owner:** foundation worktree
**Branch:** `feat/factor-foundation`

**Files:**
- Modify: `src/symlmm/specs.py`
- Create: `src/symlmm/factors.py`
- Modify: `src/symlmm/model.py:90-235` (dense codes and factor accessors)
- Test: `tests/unit/test_factors.py`

**Interfaces:**
- Consumes: existing `Group`, `Nested`, `Crossed`, `RandomEffects`, `CovarianceSpec`, `ModelError`.
- Produces:
  - `FactorSpec = Group | Nested | Crossed`
  - `Factor(name, label, components, index, parent, kind)`
  - `Position(name, column=None, index=None)`
  - `RandomEffects(..., name=None)`
  - `FactorResolver(specs, observations=None).resolve(ref) -> Factor`
  - `FactorResolver.resolve_all() -> tuple[Factor, ...]`
  - `LMM.factor(ref)`, `LMM.factors()`, `LMM.level_code(factor, i)`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_factors.py`:

```python
import pytest
import sympy as sp

import symlmm as sl
from symlmm.covariances import IID, Scalar
from symlmm.errors import ModelError
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, Nested, RandomEffects


def _model(groups, random_group, n=4):
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group=random_group, intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=groups,
        n=n,
    )


def test_nested_string_resolves_registered_ancestry():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", "G")
    model = _model([F, G], H)
    factor = model.factor("H")
    assert factor.components == ("F", "G", "H")
    assert factor.label == "F:G:H"
    assert model.factor(("F", "G", "H")) is factor


def test_nested_explicit_spec_carries_ancestry():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", G)
    model = _model([F, G], H)
    assert model.factor("H").components == ("F", "G", "H")


def test_unregistered_string_is_legacy_shallow():
    H = Nested("H", "missing")
    model = _model([], H)
    assert model.factor("H").components == ("missing", "H")


def test_dense_codes_use_full_ancestry():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", "G")
    model = _model([F, G], H)
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    assert [dense.level_code("H", k) for k in range(4)] == [0, 1, 2, 3]


def test_repeated_child_labels_across_ancestors_do_not_collide():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", "G")
    model = _model([F, G], H)
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    assert dense.covariance_entry(0, 2) == 0


def test_duplicate_factor_names_with_different_paths_raise():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", "G")
    with pytest.raises(ModelError, match="conflicting"):
        _model([F, G, Group("H")], H)


def test_cycle_raises():
    with pytest.raises(ModelError, match="cycle"):
        _model([Nested("F", "G"), Nested("G", "F")], Group("F"))


def test_crossed_accepts_multiple_components():
    model = _model([], sl.Crossed("F", "G", "H"))
    factor = model.factor("H")
    assert factor.components == ("F", "G", "H")
    assert factor.label == "F:G:H"


def test_random_effects_name_is_stored():
    term = RandomEffects(
        group="F",
        intercept=True,
        cov=Scalar(sp.Symbol("tau2", positive=True)),
        name="custom",
    )
    assert term.name == "custom"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_factors.py -v`

Expected: collection errors or assertion failures because `Factor`, `Position`, `model.factor`, `level_code`, and `RandomEffects.name` do not exist.

- [ ] **Step 3: Implement declarations and resolution**

In `src/symlmm/specs.py`, add `FactorSpec`, `Factor`, and `Position` after the existing factor declarations. Extend `Nested` and `Crossed`:

```python
from collections.abc import Sequence


@dataclass(frozen=True)
class Factor:
    """A resolved grouping dimension."""

    name: str
    label: str
    components: tuple[str, ...]
    index: sp.Symbol
    parent: Factor | None = None
    kind: str = "group"


@dataclass(frozen=True)
class Position:
    """An optional within-block position axis."""

    name: str
    column: str | None = None
    index: sp.Symbol | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            object.__setattr__(self, "index", sp.Symbol(self.name, integer=True))


```

Change `Nested`:

```python
@dataclass(frozen=True)
class Nested:
    """A factor nested within another factor."""

    child: str
    within: str | FactorSpec
    index: sp.Symbol | None = None

    def __post_init__(self) -> None:
        if isinstance(self.within, str) and not self.within:
            raise ModelError("Nested.within must be a non-empty name or factor definition")
        if self.index is None:
            object.__setattr__(self, "index", sp.Symbol(self.label, integer=True))

    @property
    def name(self) -> str:
        """Return the factor name used in accessor lookups."""
        return self.child

    @property
    def label(self) -> str:
        """Return the declaration-level composite label."""
        parent = self.within if not isinstance(self.within, str) else None
        prefix = parent.label if parent is not None else self.within
        return f"{prefix}:{self.child}"
```

Change `Crossed` to accept multiple components while preserving `left`/`right`:

```python
@dataclass(frozen=True, init=False)
class Crossed:
    """A crossed combination of two or more factors."""

    components: tuple[str | FactorSpec, ...]
    index: sp.Symbol | None
    explicit_name: str | None

    def __init__(
        self,
        *components: str | FactorSpec,
        index: sp.Symbol | None = None,
        name: str | None = None,
    ) -> None:
        if len(components) < 2:
            raise ModelError("Crossed requires at least two components")
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "explicit_name", name)
        if index is None:
            object.__setattr__(self, "index", sp.Symbol(self.label, integer=True))

    @property
    def left(self) -> str | FactorSpec:
        """Return the first component."""
        return self.components[0]

    @property
    def right(self) -> str | FactorSpec:
        """Return the final component."""
        return self.components[-1]

    @property
    def name(self) -> str:
        """Return the explicit or legacy lookup name."""
        if self.explicit_name is not None:
            return self.explicit_name
        right = self.right
        return right.name if not isinstance(right, str) else right

    @property
    def label(self) -> str:
        """Return the declaration-level composite label."""
        parts = []
        for component in self.components:
            parts.append(component if isinstance(component, str) else component.label)
        return ":".join(parts)


FactorSpec = Group | Nested | Crossed
```

Update `_coerce_group` to `str | FactorSpec -> FactorSpec`.

Add `name` to `RandomEffects`:

```python
@dataclass(frozen=True, init=False)
class RandomEffects:
    group: FactorSpec
    cov: CovarianceSpec
    intercept: bool = False
    slopes: tuple[RandomSlope, ...] = ()
    name: str | None = None

    def __init__(
        self,
        group: str | FactorSpec,
        cov: CovarianceSpec,
        intercept: bool = False,
        slopes: tuple[RandomSlope, ...] = (),
        name: str | None = None,
    ) -> None:
        object.__setattr__(self, "group", _coerce_group(group))
        object.__setattr__(self, "cov", cov)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "slopes", tuple(slopes))
        object.__setattr__(self, "name", name)
        if not intercept and not self.slopes:
            raise CovarianceConstructionError("RandomEffects requires an intercept or slope")
        if cov.dim != self.dim:
            raise CovarianceConstructionError(
                f"RandomEffects covariance must be {self.dim}x{self.dim}"
            )
```

- [ ] **Step 4: Implement `FactorResolver`**

Create `src/symlmm/factors.py`:

```python
"""Resolve factor declarations into immutable factors with full ancestry."""

from __future__ import annotations

from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import ModelError
from .specs import Crossed, Factor, FactorSpec, Group, Nested


class FactorResolver:
    """Resolve factor declarations and string references to `Factor` objects."""

    def __init__(self, specs: tuple[FactorSpec, ...]) -> None:
        self.specs = specs
        self._resolved: dict[str, Factor] = {}
        self._by_label: dict[str, Factor] = {}

    def resolve_all(self) -> tuple[Factor, ...]:
        """Resolve every declaration and return factors in declaration order."""
        for spec in self.specs:
            self._resolve(spec, ())
        return tuple(self._resolved.values())

    def resolve(self, ref: str | FactorSpec | Factor | tuple[str, ...]) -> Factor:
        """Resolve a name, component path, declaration, or factor."""
        if isinstance(ref, Factor):
            return ref
        if isinstance(ref, str):
            try:
                return self._resolved[ref]
            except KeyError as exc:
                raise ModelError(f"unknown factor: {ref!r}") from exc
        if isinstance(ref, tuple):
            for factor in self._resolved.values():
                if factor.components == ref:
                    return factor
            raise ModelError(f"unknown factor path: {ref!r}")
        return self._resolve(ref, ())

    def _resolve(self, spec: FactorSpec, stack: tuple[str, ...]) -> Factor:
        if isinstance(spec, Group):
            return self._register(
                Factor(
                    name=spec.name,
                    label=spec.name,
                    components=(spec.name,),
                    index=spec.index,
                    parent=None,
                    kind="group",
                )
            )
        if isinstance(spec, Nested):
            if spec.child in stack:
                raise ModelError(f"cycle in nested factors: {' -> '.join(stack + (spec.child,))}")
            parent = self._resolve_reference(spec.within, stack + (spec.child,))
            components = (*(parent.components if parent else ()), spec.child)
            return self._register(
                Factor(
                    name=spec.child,
                    label=":".join(components),
                    components=components,
                    index=self._resolved_index(spec, components),
                    parent=parent,
                    kind="nested",
                )
            )
        if isinstance(spec, Crossed):
            components: list[str] = []
            for component in spec.components:
                resolved = self._resolve_reference(component, stack)
                if resolved is None:
                    if not isinstance(component, str):
                        raise ModelError("Crossed component could not be resolved")
                    components.append(component)
                else:
                    components.extend(resolved.components)
            factor = Factor(
                name=spec.name,
                label=":".join(components),
                components=tuple(components),
                index=self._resolved_index(spec, tuple(components)),
                parent=None,
                kind="crossed",
            )
            return self._register(factor)
        raise ModelError(f"unsupported factor declaration: {spec!r}")

    def _resolved_index(self, spec: FactorSpec, components: tuple[str, ...]) -> Any:
        """Return the explicit index, or a default index built from full ancestry."""
        default = sp.Symbol(spec.label, integer=True)
        if spec.index == default:
            return sp.Symbol(":".join(components), integer=True)
        return spec.index

    def _resolve_reference(
        self,
        ref: str | FactorSpec,
        stack: tuple[str, ...],
    ) -> Factor | None:
        if not isinstance(ref, str):
            return self._resolve(ref, stack)
        if ref in self._resolved:
            return self._resolved[ref]
        for spec in self.specs:
            if spec.name == ref:
                return self._resolve(spec, stack)
        return None

    def _register(self, factor: Factor) -> Factor:
        existing = self._resolved.get(factor.name)
        if existing is not None:
            if existing.components != factor.components:
                raise ModelError(
                    f"conflicting factor definitions for {factor.name!r}: "
                    f"{existing.label!r} and {factor.label!r}"
                )
            return existing
        existing_label = self._by_label.get(factor.label)
        if existing_label is not None:
            raise ModelError(
                f"conflicting factor label {factor.label!r}: "
                f"{existing_label.name!r} and {factor.name!r}"
            )
        self._resolved[factor.name] = factor
        self._by_label[factor.label] = factor
        return factor
```

- [ ] **Step 5: Integrate resolution into `LMM`**

In `src/symlmm/model.py`:

1. Import `Factor`, `FactorResolver` at module scope.
2. In `LMM.__init__`, replace the `registered` dict and `register` closure with a `FactorResolver`.
3. Resolve all declarations after collecting `groups`, `random` groups, and the residual group:

```python
specs = list(groups or ())
specs.extend(term.group for term in self.random)
if residual_group is not None:
    specs.append(residual_group)
self._factor_resolver = FactorResolver(tuple(specs))
self._factors = self._factor_resolver.resolve_all()
```

4. Store `self._factor_resolver` and `self._factors`; pass `groups=self._factors` to `super().__init__`.
5. Replace `_group_spec` to return a resolved `Factor` by name.
6. Replace primary-group selection with resolved factors:

```python
if self.random:
    primary = self.factor(self.random[0].group.name)
elif residual_group is not None:
    primary = self.factor(residual_group.name)
elif self._factors:
    primary = self._factors[0]
else:
    primary = None
```

7. Add:

```python
def factor(self, ref: Any) -> Any:
    """Return a resolved factor by name, path, declaration, or factor."""
    return self._factor_resolver.resolve(ref)

def factors(self) -> tuple[Any, ...]:
    """Return all resolved factors in declaration order."""
    return self._factors

def level_code(self, factor: Any, i: Any) -> Any:
    """Return the factor level code for observation row `i`."""
    resolved = self.factor(factor)
    if self._dense:
        return self._dense_group_codes(resolved)[int(i)]
    return sp.IndexedBase(f"g_{resolved.label.replace(':', '_')}", shape=(self.n,))[i]
```

8. Change `_dense_group_codes(factor: Factor)` to use `factor.components`:

```python
def _dense_group_codes(self, factor: Any) -> list[int]:
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
        raw = [":".join(str(value) for value in values) for values in zip(*columns, strict=True)]
    codes = {label: code for code, label in enumerate(sorted(set(raw), key=str))}
    result = [codes[label] for label in raw]
    self._group_cache[factor.label] = result
    return result
```

9. Keep `group(i, factor)` as a deprecated alias to `level_code`.

- [ ] **Step 6: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_factors.py tests/unit/test_indices.py -v
```

Expected: all new tests pass; old `test_indices` passes except any test asserting shallow nested labels, which must be updated to expect the resolved path.

- [ ] **Step 7: Commit**

```bash
git add src/symlmm/specs.py src/symlmm/factors.py src/symlmm/model.py tests/unit/test_factors.py tests/unit/test_indices.py
git commit -m "feat: resolve factor ancestry and dense codes"
```

---

## Task 2: Term Identity and Row-Level Decomposition

**Owner:** term worktree
**Branch:** `feat/term-decomposition`
**Depends on:** Task 1

**Files:**
- Create: `src/symlmm/decomposition.py`
- Test: `tests/unit/test_decomposition.py`

**Interfaces:**
- Consumes: `Factor`, `RandomEffects.name`, `LMM._random_term_entry`, `LMM.residual`.
- Produces:
  - `Term(name, factor, random, index)`
  - `terms(model) -> tuple[Term, ...]`
  - `covariance_terms(model, a, b) -> dict[str, Any]`
  - `partial_covariance(model, a, b, exclude=()) -> Any`
  - `partial_correlation(model, a, b, exclude=()) -> Any`
- This workstream does not edit `model.py`; Task 5 adds the `LMM` wrappers.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_decomposition.py`:

```python
import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.decomposition import (
    covariance_terms,
    partial_correlation,
    partial_covariance,
    terms,
)
from symlmm.errors import ModelError
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, RandomEffects


def _model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="F", intercept=True, cov=Scalar(t2), name="f"),
            RandomEffects(group="G", intercept=True, cov=Scalar(t2), name="g"),
        ],
        residual=IID(var=s2),
        n=2,
    )


def test_terms_are_named_and_ordered():
    model = _model()
    assert [term.name for term in terms(model)] == ["f", "g"]


def test_covariance_terms_sum_to_total():
    model = _model()
    total = model.covariance_entry(0, 1)
    values = covariance_terms(model, 0, 1)
    assert sum(values.values()) == total
    assert list(values)[-1] == "residual"


def test_partial_covariance_excludes_term():
    model = _model()
    total = model.covariance_entry(0, 1)
    values = covariance_terms(model, 0, 1)
    assert partial_covariance(model, 0, 1, exclude=("f",)) == total - values["f"]


def test_partial_covariance_excludes_everything():
    model = _model()
    assert partial_covariance(model, 0, 1, exclude=("f", "g", "residual")) == 0


def test_partial_correlation_diagonal_is_one():
    model = _model()
    assert partial_correlation(model, 0, 0, exclude=("f",)) == 1


def test_unknown_exclusion_raises():
    model = _model()
    with pytest.raises(ModelError, match="unknown term"):
        partial_covariance(model, 0, 1, exclude=("missing",))


def test_residual_name_is_reserved():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2), name="residual")],
        residual=IID(var=s2),
        n=2,
    )
    with pytest.raises(ModelError, match="reserved"):
        terms(model)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_decomposition.py -v`

Expected: failures for missing `terms`, `covariance_terms`, `partial_covariance`, `partial_correlation`.

- [ ] **Step 3: Implement `Term` and decomposition helpers**

Create `src/symlmm/decomposition.py`:

```python
"""Term-level covariance decomposition and partial covariance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import ModelError, NotDefined
from .specs import Factor, RandomEffects

RESIDUAL = "residual"


@dataclass(frozen=True)
class Term:
    """A named random-effect component."""

    name: str
    factor: Factor
    random: RandomEffects
    index: int


def terms(model: Any) -> tuple[Term, ...]:
    """Return named random-effect terms in model order."""
    resolved = []
    for index, random in enumerate(model.random):
        factor = model.factor(random.group.name)
        name = random.name or factor.label
        resolved.append(Term(name=name, factor=factor, random=random, index=index))
    names = [term.name for term in resolved]
    if len(names) != len(set(names)):
        raise ModelError("random-effect term names must be unique")
    if RESIDUAL in names:
        raise ModelError("'residual' is reserved for the residual term")
    return tuple(resolved)


def validate_exclusions(terms: tuple[Term, ...], exclude: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and de-duplicate public exclusion names."""
    names = {term.name for term in terms}
    names.add(RESIDUAL)
    unknown = sorted(set(exclude) - names)
    if unknown:
        raise ModelError(f"unknown term names: {unknown!r}")
    return tuple(dict.fromkeys(exclude))


def term_value(model: Any, term: Term, a: Any, b: Any) -> Any:
    """Return one term's contribution to Cov(Y_a, Y_b)."""
    return model._random_term_entry(term.random, a, b)


def covariance_terms(model: Any, a: Any, b: Any) -> dict[str, Any]:
    """Return per-term contributions to Cov(Y_a, Y_b)."""
    resolved = terms(model)
    values = {term.name: term_value(model, term, a, b) for term in resolved}
    values[RESIDUAL] = model.residual.entry(a, b, model)
    return values


def partial_covariance(
    model: Any,
    a: Any,
    b: Any,
    exclude: tuple[str, ...] = (),
) -> Any:
    """Return Cov(Y_a, Y_b) with named term contributions removed."""
    resolved = terms(model)
    validate_exclusions(resolved, exclude)
    total = model.covariance_entry(a, b)
    for term in resolved:
        if term.name in exclude:
            total -= term_value(model, term, a, b)
    if RESIDUAL in exclude:
        total -= model.residual.entry(a, b, model)
    return sp.simplify(total)


def partial_correlation(
    model: Any,
    a: Any,
    b: Any,
    exclude: tuple[str, ...] = (),
) -> Any:
    """Return the partial correlation between two scalar entries."""
    covariance = partial_covariance(model, a, b, exclude=exclude)
    variance_a = partial_covariance(model, a, a, exclude=exclude)
    variance_b = partial_covariance(model, b, b, exclude=exclude)
    denominator = sp.sqrt(variance_a * variance_b)
    if denominator == 0:
        raise NotDefined("partial correlation requires nonzero partial variances")
    return sp.simplify(covariance / denominator)
```

- [ ] **Step 4: Export the module API**

Add `__all__ = ["RESIDUAL", "Term", "covariance_terms", "partial_correlation", "partial_covariance", "terms", "validate_exclusions"]` to `decomposition.py` after the definitions.

Do not edit `model.py` in this workstream. Task 5 owns the `LMM` wrappers.

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_decomposition.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/decomposition.py tests/unit/test_decomposition.py
git commit -m "feat: add term decomposition and partial covariance"
```

---

## Task 3a: Block Core and Basic Moments

**Owner:** block worktree
**Branch:** `feat/block-moments`
**Depends on:** Task 1
**Parallel with:** Task 2

**Files:**
- Create: `src/symlmm/blocks.py`
- Test: `tests/unit/test_blocks.py`

**Interfaces:**
- Consumes: `Factor`, `LMM.matrix_covariance`, `matrix_mean`, `matrix_x`, `matrix_z`, `matrix_r`, `_dense_group_codes`.
- Produces:
  - `Block(model, factor, level)`
  - `Block.rows() -> list[int]`
  - `Block.size() -> Any`
  - `Block.mean() -> Any`
  - `Block.covariance() -> Any`
  - `Block.cross_covariance(other) -> Any`
  - `Block.design() -> Any`
  - `Block.random_design() -> Any`
  - `Block.residual() -> Any`
  - `Block.random_effect() -> Any`
  - Symbolic helpers `block_size_symbol`, `block_mean_symbol`, `block_random_design`, `block_residual`, `block_coefficients`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_blocks.py`:

```python
import sympy as sp

from symlmm.blocks import Block
from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, Nested, RandomEffects


def _hierarchical():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="F", intercept=True, cov=Scalar(t2)),
            RandomEffects(group=Nested("H", "G"), intercept=True, cov=Scalar(t2)),
        ],
        residual=IID(var=s2),
        groups=[Group("F"), Nested("G", "F")],
        n=4,
    )


def test_block_dense_covariance_matches_submatrix():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    rows = block.rows()
    assert rows == [0, 1]
    assert block.covariance() == dense.matrix_covariance()[rows, :][:, rows]


def test_block_cross_covariance_matches_submatrix():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    left = Block(dense, dense.factor("H"), 0)
    right = Block(dense, dense.factor("H"), 1)
    assert left.cross_covariance(right) == dense.matrix_covariance()[left.rows(), :][:, right.rows()]


def test_block_mean_and_design():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    assert block.mean() == dense.matrix_mean()[block.rows(), :]
    assert block.design() == dense.matrix_x()[block.rows(), :]


def test_symbolic_block_covariance_shape():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    assert block.covariance().shape == (block.size(), block.size())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_blocks.py -v`

Expected: failures for missing `Block` and block methods.

- [ ] **Step 3: Implement `Block`**

Create `src/symlmm/blocks.py`:

```python
"""Factor-level observation blocks and their moments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import NotDefined
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

    def random_design(self) -> Any:
        """Return the full random design block."""
        if self.model._dense:
            return self.model.matrix_z()[self.rows(), :]
        q = self.model.dimensions()[1]
        return block_random_design(self.factor, self.level, q)

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
```

- [ ] **Step 4: Do not edit `model.py` in this workstream**

The public `LMM.block` factory is added in Task 5. Task 3a tests construct `Block` directly.

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_blocks.py tests/unit/test_factors.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/blocks.py tests/unit/test_blocks.py
git commit -m "feat: add factor-level block moments"
```

---

## Task 3b: Block Term Decomposition and Partial Covariance

**Owner:** block worktree (continue after Task 2 merges)
**Branch:** `feat/block-moments`
**Depends on:** Task 2 + Task 3a

**Files:**
- Modify: `src/symlmm/blocks.py`
- Test: `tests/unit/test_blocks.py`

**Interfaces:**
- Consumes: `Term`, `terms(model)`, `validate_exclusions`, `Block.covariance`, `Block.random_design`.
- Produces:
  - `Block.random_design(term=...) -> Any`
  - `Block.covariance_terms() -> dict[str, Any]`
  - `Block.partial_covariance(other, exclude=()) -> Any`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_blocks.py`:

```python
import pytest

from symlmm.errors import ModelError


def test_block_covariance_terms_sum_to_total():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    terms = block.covariance_terms()
    assert sum(terms.values()) == block.covariance()
    assert list(terms)[-1] == "residual"


def test_block_partial_covariance_excludes_term():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    terms = block.covariance_terms()
    excluded = list(terms)[0]
    assert block.partial_covariance(block, exclude=[excluded]) == (
        block.covariance() - terms[excluded]
    )


def test_block_unknown_exclusion_raises():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    with pytest.raises(ModelError, match="unknown term"):
        block.partial_covariance(block, exclude=["missing"])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_blocks.py -k "terms or partial" -v`

Expected: failures for missing methods.

- [ ] **Step 3: Implement block decomposition**

In `src/symlmm/blocks.py`, add the term-design helper:

```python
def block_term_design(factor: Factor, level: Any, term: Any) -> sp.MatrixSymbol:
    """Return the random design matrix for one term and block."""
    return sp.MatrixSymbol(
        f"Z_{factor.label}:{level}:{term.name}",
        block_size_symbol(factor, level),
        term.random.dim,
    )
```

Replace `Block.random_design` with a term-aware version:

```python
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
```

Append the decomposition methods:

```python
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
            total -= block_cross_residual(
                self.factor,
                self.level,
                other.factor,
                other.level,
            )
        return sp.simplify(total)
```

- [ ] **Step 4: Do not edit `model.py` in this workstream**

The `LMM.covariance_terms` and `LMM.partial_*` block dispatch is wired in Task 5. Task 3b tests call `Block` methods directly.

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_blocks.py tests/unit/test_decomposition.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/blocks.py tests/unit/test_blocks.py
git commit -m "feat: add block term decomposition and partial covariance"
```

---

## Task 4: PanelGrid and PanelView

**Owner:** panel worktree
**Branch:** `feat/panel-view`
**Depends on:** Task 1 + Task 3a

**Files:**
- Create: `src/symlmm/panel.py`
- Test: `tests/unit/test_panel.py`

**Interfaces:**
  - Consumes: `Factor`, `Position`, `Block`, `LMM._dense_position_codes_for`, `LMM._dense_group_codes`, `LMM.matrix_mean`, `LMM.matrix_covariance`.
- Produces:
  - `PanelGrid(model, factor, position)`
  - `PanelGrid.level_codes()`, `position_codes()`, `is_balanced()`, `row(level, position)`
  - `PanelView(model, factor, position)`
  - `PanelView.level_count()`, `position_count()`, `response_matrix()`, `response(level)`, `response_column(position)`, `entry_mean`, `entry_covariance`, `entry_cross_covariance`, `position_mean`, `position_covariance`, `position_code(i)`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_panel.py`:

```python
import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import NotDefined
from symlmm.model import LMM
from symlmm.panel import PanelView
from symlmm.specs import Fixed, Group, Position, RandomEffects


def _model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
        n=4,
    )


def test_panel_uses_position_codes_not_raw_values():
    dense = _model().dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    assert panel.position_count() == 2
    assert panel.response_matrix().shape == (2, 2)


def test_panel_explicit_position_column():
    dense = _model().dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21], visit=["a", "b", "c", "d"])
    panel = PanelView(dense, dense.factor("F"), Position("p", column="visit"))
    assert panel.position_count() == 2


def test_panel_rejects_unbalanced_blocks():
    dense = _model().dense(F=[0, 0, 1], p=[10, 11, 20])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    with pytest.raises(NotDefined, match="balanced"):
        panel.position_count()


def test_panel_entry_moments():
    dense = _model().dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    assert panel.entry_mean(0, 0) == dense.matrix_mean()[0, 0]
    assert panel.entry_covariance(0, 0, 1) == dense.matrix_covariance()[0, 1]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_panel.py -v`

Expected: failures for missing `symlmm.panel`, `Position`, `PanelView`.

- [ ] **Step 3: Implement PanelGrid and PanelView**

Create `src/symlmm/panel.py`:

```python
"""Panel view over one factor and one within-block position axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .blocks import Block
from .errors import NotDefined
from .specs import Factor, Position


def _position_sort_key(value: Any, position: int) -> tuple[Any, ...]:
    """Return a total ordering key for a position value."""
    try:
        return (0, sp.default_sort_key(sp.sympify(value)), position)
    except (sp.SympifyError, TypeError):  # fmt: skip
        return (1, str(value), position)


@dataclass(frozen=True)
class PanelGrid:
    """Resolved level and position codes for a panel."""

    model: Any
    factor: Factor
    position: Position

    def level_codes(self) -> list[int]:
        """Return factor level codes for every observation."""
        if not self.model._dense:
            raise NotDefined("panel grids require dense mode")
        return self.model._dense_group_codes(self.factor)

    def position_codes(self) -> list[int]:
        """Return position codes for every observation."""
        if not self.model._dense:
            raise NotDefined("panel grids require dense mode")
        return self.model._dense_position_codes_for(self.factor, self.position)

    def levels(self) -> list[int]:
        """Return sorted level codes."""
        return sorted(set(self.level_codes()))

    def positions(self) -> list[int]:
        """Return sorted position codes."""
        return sorted(set(self.position_codes()))

    def rows_for_level(self, level: Any) -> list[int]:
        """Return block-local rows ordered by position code."""
        rows = [
            row
            for row, code in enumerate(self.level_codes())
            if code == level
        ]
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
        block_rows = [
            row
            for row, code in enumerate(self.level_codes())
            if code == level
        ]
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

    def __init__(self, model: Any, factor: Any, position: str | Position) -> None:
        self.model = model
        self.factor = model.factor(factor)
        self.position = position if isinstance(position, Position) else Position(position)
        self.grid = PanelGrid(model, self.factor, self.position)

    def level_count(self) -> Any:
        """Return the number of factor levels."""
        if self.model._dense:
            return len(self.grid.levels())
        return self.model.levels(self.factor)

    def position_count(self) -> Any:
        """Return the number of positions in a balanced panel."""
        if not self.model._dense:
            return sp.Symbol("T", positive=True, integer=True)
        if not self.grid.is_balanced():
            raise NotDefined("panel positions require a balanced panel")
        return len(self.grid.positions())

    def response_matrix(self) -> Any:
        """Return the balanced-panel response matrix Y."""
        return sp.MatrixSymbol("Y", self.level_count(), self.position_count())

    def response(self, level: Any) -> Any:
        """Return the response row for a factor level."""
        return self.response_matrix()[level, :]

    def response_column(self, position: Any) -> Any:
        """Return the response column for a position."""
        return self.response_matrix()[:, position]

    def position_code(self, i: Any) -> Any:
        """Return the position code for observation row `i`."""
        if self.model._dense:
            return self.grid.position_codes()[int(i)]
        return sp.IndexedBase(f"k_{self.factor.label.replace(':', '_')}", shape=(self.model.n,))[i]

    def entry_mean(self, level: Any, position: Any) -> Any:
        """Return E[Y_level,position]."""
        row = self.grid.row(level, position)
        return Block(self.model, self.factor, level).mean()[row, 0]

    def entry_covariance(self, level: Any, p1: Any, p2: Any) -> Any:
        """Return Cov(Y_level,p1, Y_level,p2)."""
        row1 = self.grid.row(level, p1)
        row2 = self.grid.row(level, p2)
        return Block(self.model, self.factor, level).covariance()[row1, row2]

    def entry_cross_covariance(
        self, level1: Any, p1: Any, level2: Any, p2: Any
    ) -> Any:
        """Return Cov(Y_level1,p1, Y_level2,p2)."""
        row1 = self.grid.row(level1, p1)
        row2 = self.grid.row(level2, p2)
        return Block(self.model, self.factor, level1).cross_covariance(
            Block(self.model, self.factor, level2)
        )[row1, row2]

    def position_mean(self, position: Any) -> Any:
        """Return E[Y_{.,position}]."""
        count = self.level_count()
        if not isinstance(count, int):
            raise NotDefined("position means require a concrete level count")
        return sp.Matrix(
            [self.entry_mean(level, position) for level in range(count)]
        )

    def position_covariance(self, p1: Any, p2: Any) -> Any:
        """Return Cov(Y_{.,p1}, Y_{.,p2})."""
        count = self.level_count()
        if not isinstance(count, int):
            raise NotDefined("position covariance requires a concrete level count")
        return sp.diag(
            *[self.entry_covariance(level, p1, p2) for level in range(count)]
        )
```

- [ ] **Step 4: Add dense position-code helper**

In `src/symlmm/model.py`, add:

```python
def _dense_position_codes_for(self, factor: Any, position: Any) -> list[int]:
    """Return dense position codes for a factor and position definition."""
    resolved = self.factor(factor)
    raw = list(self._observation_column(position.column or position.name))
    groups = self._dense_group_codes(resolved)
    result = [0] * self.n
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
        ordered = sorted(rows, key=lambda row: _position_sort_key(raw[row], row))
        code = -1
        previous: Any = object()
        for row in ordered:
            value = raw[row]
            if value != previous:
                code += 1
                previous = value
            result[row] = code
    return result
```

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_panel.py tests/unit/test_blocks.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/panel.py src/symlmm/model.py tests/unit/test_panel.py
git commit -m "feat: add factor-first panel view"
```

---

## Task 5: Public Wiring, Blackboard, and Legacy Wrappers

**Owner:** integration worktree
**Branch:** `feat/panel-symbols`
**Depends on:** Task 3b + Task 4

**Files:**
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/blackboard.py`
- Test: `tests/unit/test_panel_symbols.py`
- Test: `tests/unit/test_deprecation.py`

**Interfaces:**
- Consumes: `PanelView`, `Block`, `ModelSymbols`.
- Produces:
  - `LMM.panel(factor, position) -> PanelView`
  - `PanelView.symbols()`
  - deprecated `LMM.symbols`, `response_matrix`, `group_*`, `entry_*`, `time_*`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_panel_symbols.py`:

```python
import sympy as sp

from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects
from symlmm.covariances import IID, Scalar


def _panel_model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
        n=4,
    )
    return model.dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])


def test_panel_symbols_bind_to_panel_view():
    panel = _panel_model().panel("F", "p")
    syms = panel.symbols()
    assert syms.Y[:, :].shape == panel.response_matrix().shape
    assert syms.t == panel.position.index


def test_model_panel_returns_panel_view():
    model = _panel_model()
    panel = model.panel("F", "p")
    assert panel.response_matrix().shape == (2, 2)


def test_model_wires_blocks_and_decomposition():
    model = _panel_model()
    block = model.block("F", 0)
    other = model.block("F", 1)
    assert block.size() == 2
    assert model.block_mean("F", 0) == block.mean()
    assert model.block_covariance("F", 0) == block.covariance()
    assert model.block_cross_covariance("F", 0, "F", 1) == block.cross_covariance(other)
    values = model.covariance_terms(0, 1)
    assert sum(values.values()) == model.covariance_entry(0, 1)
    assert model.partial_covariance(
        block, other, exclude=("F",)
    ) == block.partial_covariance(other, exclude=("F",))
```

Create `tests/unit/test_deprecation.py`:

```python
import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


def _model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
        n=2,
    ).dense(F=[0, 0], p=[0, 1])


def test_time_mean_warns():
    model = _model()
    with pytest.warns(DeprecationWarning, match="time_mean"):
        model.time_mean(0)


def test_response_matrix_warns():
    model = _model()
    with pytest.warns(DeprecationWarning, match="response_matrix"):
        model.response_matrix()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_panel_symbols.py tests/unit/test_deprecation.py -v
```

Expected: failures for missing `LMM.panel`, `panel.symbols`, and warnings.

- [ ] **Step 3: Implement `LMM.panel` and deprecated wrappers**

Add public wiring to `src/symlmm/model.py`:

```python
def terms(self) -> Any:
    """Return named random-effect terms in model order."""
    from .decomposition import terms

    return terms(self)

def covariance_terms(self, a: Any, b: Any) -> Any:
    """Return per-term contributions to Cov(Y_a, Y_b)."""
    from .blocks import Block
    from .decomposition import covariance_terms

    if isinstance(a, Block) and isinstance(b, Block):
        return a.covariance_terms()
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
    """Return a panel view for a factor and position axis."""
    from .panel import PanelView

    selected_factor = factor if factor is not None else self.primary_group
    if selected_factor is None:
        raise NotDefined("panel view requires a factor")
    if position is None:
        if self.coordinate_name is None:
            raise NotDefined("panel view requires a position axis")
        position = self.coordinate_name
    return PanelView(self, selected_factor, position)
```

Replace the existing panel/group method bodies with these deprecated wrappers; do not leave duplicate method definitions.

```python
def _deprecated(self, name: str) -> Any:
    """Return the default panel view and emit a deprecation warning."""
    import warnings

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
    return self._deprecated("symbols").symbols()
```

- [ ] **Step 4: Bind blackboard through `PanelAdapter`**

Add `PanelAdapter` and `PanelView.symbols()` to `src/symlmm/panel.py`:

```python
class PanelAdapter:
    """Expose the legacy panel model surface used by blackboard.py."""

    def __init__(self, panel: PanelView) -> None:
        self.panel = panel

    def __getattr__(self, name: str) -> Any:
        return getattr(self.panel.model, name)

    @property
    def group_index(self) -> Any:
        return self.panel.factor.index

    @property
    def occasion_index(self) -> Any:
        return self.panel.position.index

    @property
    def primary_group(self) -> str:
        return self.panel.factor.name

    def response_matrix(self) -> Any:
        return self.panel.response_matrix()

    def response(self, level: Any, position: Any = None) -> Any:
        if position is None:
            return self.panel.response(level)
        return self.panel.response_matrix()[level, position]

    def time_response(self, position: Any) -> Any:
        return self.panel.response_column(position)

    def time_mean(self, position: Any) -> Any:
        return self.panel.position_mean(position)

    def time_covariance(self, p1: Any, p2: Any) -> Any:
        return self.panel.position_covariance(p1, p2)

    def entry_mean(self, level: Any, position: Any) -> Any:
        return self.panel.entry_mean(level, position)

    def entry_covariance(self, level: Any, p1: Any, p2: Any) -> Any:
        return self.panel.entry_covariance(level, p1, p2)

    def entry_cross_covariance(
        self, level1: Any, p1: Any, level2: Any, p2: Any
    ) -> Any:
        return self.panel.entry_cross_covariance(level1, p1, level2, p2)

    def _entry_row(self, level: Any, position: Any) -> int:
        return self.panel.grid.row(level, position)

    def group_mean(self, level: Any) -> Any:
        return self.panel.model.block(self.panel.factor, level).mean()

    def group_covariance(self, level: Any) -> Any:
        return self.panel.model.block(self.panel.factor, level).covariance()

    def group_cross_covariance(self, left: Any, right: Any) -> Any:
        return self.panel.model.block(self.panel.factor, left).cross_covariance(
            self.panel.model.block(self.panel.factor, right)
        )

    def fixed_design(self, level: Any) -> Any:
        return self.panel.model.block(self.panel.factor, level).design()

    def random_design(self, level: Any) -> Any:
        return self.panel.model.block(self.panel.factor, level).random_design()

    def residual_covariance(self, level: Any) -> Any:
        return self.panel.model.block(self.panel.factor, level).residual()

    def random_effect(self, level: Any) -> Any:
        return self.panel.model.block(self.panel.factor, level).random_effect()

    def symbols(self) -> Any:
        from .blackboard import symbols

        return symbols(self)
```

Add to `PanelView`:

```python
    def symbols(self) -> Any:
        """Return panel blackboard symbols."""
        return PanelAdapter(self).symbols()
```

`blackboard.py` does not need its internal time calls rewritten: it receives a `PanelAdapter`, whose `time_*`, `entry_*`, `group_*`, `_entry_row`, `X`, `Z`, `R`, and `b` methods are panel-specific. Keep `symbols(model)` unchanged except for the `LMM` type annotation widening to `Any`. Do not register `LMM.symbols` in `blackboard.py`; add the deprecated method in `model.py`:

```python
def symbols(self) -> Any:
    """Deprecated: use panel(...).symbols()."""
    import warnings

    warnings.warn("symbols is deprecated; use panel(...)", DeprecationWarning, stacklevel=2)
    return self.panel().symbols()
```

Remove the lazy `LMM.symbols` registration at the bottom of `blackboard.py`; the method now lives on `LMM`.

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_panel_symbols.py tests/unit/test_deprecation.py tests/unit/test_blackboard.py -v
```

Expected: PASS; old blackboard tests either pass through wrappers or are updated to use `panel.symbols()`.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/model.py src/symlmm/blackboard.py src/symlmm/panel.py tests/unit/test_panel_symbols.py tests/unit/test_deprecation.py
git commit -m "feat: wire factor-first panel and legacy adapters"
```

---

## Task 6: Vocabulary Cleanup, Documentation, and Exports

**Owner:** docs worktree
**Branch:** `chore/factor-first-vocab`
**Depends on:** Task 5

**Files:**
- Delete: `src/symlmm/canonical.py`
- Modify: `src/symlmm/blocks.py` (remove dead canonical helpers)
- Modify: `src/symlmm/__init__.py`
- Modify: `CHANGELOG.md`, `README.md`, `docs/pedagogy/blackboard.md`
- Modify: `tests/unit/test_canonical.py` (update imports to `symlmm.blocks`)
- Test: `tests/unit/test_package.py`

**Interfaces:**
- Consumes: `blocks.py` from Task 3a/3b.
- Produces: no `symlmm.canonical` module; `symlmm.blocks` exports block helpers.

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_package.py`:

```python
import importlib.util


def test_canonical_module_is_removed():
    assert importlib.util.find_spec("symlmm.canonical") is None


def test_blocks_module_is_public():
    import symlmm.blocks as blocks

    assert hasattr(blocks, "Block")
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_package.py -v`

Expected: `canonical` module still exists or `blocks` missing.

- [ ] **Step 3: Remove the legacy module**

`blocks.py` already owns the block helpers created in Task 3a. Do not duplicate them.

1. Update every remaining import from `.canonical` to `.blocks`.
2. Confirm `canonical.py` has no unique public helpers left.
3. Delete `src/symlmm/canonical.py`.
4. Update `tests/unit/test_canonical.py` to import from `symlmm.blocks`.
5. Export `Factor`, `FactorSpec`, `Position`, `Block`, `PanelView`, `Term`, and block helpers from `__init__.py`.

- [ ] **Step 4: Update docs**

- `CHANGELOG.md`: add an Unreleased entry describing factor-first API and deprecations.
- `README.md`: replace “scaffolded; implementation has not started” with the current package scope and a factor-first example.
- `docs/pedagogy/blackboard.md`: change panel examples to `model.panel(...)` and `panel.symbols()`.
- Remove “canonical” from new prose; keep it only in migration notes.

- [ ] **Step 5: Run tests and checks**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_package.py tests/unit/test_blocks.py tests/unit/test_panel.py tests/unit/test_canonical.py -v
uv run ruff check
uv run ruff format --check
uv run mypy src/symlmm
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/__init__.py src/symlmm/blocks.py src/symlmm/canonical.py CHANGELOG.md README.md docs/pedagogy/blackboard.md tests/unit/test_package.py tests/unit/test_canonical.py
git commit -m "chore: remove canonical vocabulary"
```

---

## Task 7: Integration, Regression Tests, and Release Checks

**Owner:** supervisor / integration worktree
**Branch:** `feat/factor-first-integration`
**Depends on:** Tasks 1–6

**Files:**
- Modify: existing tests as needed (`tests/unit/test_indices.py`, `test_blackboard.py`, `test_dense.py`)
- Test: `tests/integration/test_hierarchy.py`
- Test: `tests/integration/test_panel_view.py`

**Interfaces:**
- Consumes: all public APIs from Tasks 1–6.
- Produces: final merged tree.

- [ ] **Step 1: Add integration tests**

Create `tests/integration/test_hierarchy.py`:

```python
import sympy as sp

from symlmm.blocks import Block
from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, Nested, RandomEffects


def test_three_level_hierarchy_covariance():
    b0 = sp.Symbol("b0")
    s2_e, s2_f, s2_g, s2 = sp.symbols("s2_e s2_f s2_g s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group=Group("E"), intercept=True, cov=Scalar(s2_e)),
            RandomEffects(group=Nested("F", "E"), intercept=True, cov=Scalar(s2_f)),
            RandomEffects(group=Nested("G", "F"), intercept=True, cov=Scalar(s2_g)),
        ],
        residual=IID(var=s2),
        groups=[Group("E"), Nested("F", "E")],
        n=4,
    )
    dense = model.dense(E=[0, 0, 1, 1], F=[0, 0, 0, 0], G=[0, 1, 0, 1])
    assert dense.covariance_entry(0, 0) == s2_e + s2_f + s2_g + s2
    assert dense.covariance_entry(0, 1) == s2_e + s2_f
    assert dense.covariance_entry(0, 2) == 0
    assert dense.covariance_entry(0, 3) == 0
```

Create `tests/integration/test_panel_view.py`:

```python
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


def test_panel_view_balanced_with_different_raw_coordinates():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
        n=4,
    ).dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])
    panel = model.panel("F", "p")
    assert panel.position_count() == 2
    assert panel.response_matrix().shape == (2, 2)
```

- [ ] **Step 2: Merge order**

Merge in this order:

1. `feat/factor-foundation`
2. `feat/term-decomposition`
3. `feat/block-moments`
4. `feat/panel-view`
5. `feat/panel-symbols`
6. `chore/factor-first-vocab`

After each merge, run:

```bash
uv run pytest tests/unit tests/integration
```

Fix conflicts in the integration worktree; never resolve conflicts by reverting another workstream’s public interface.

- [ ] **Step 3: Update old tests**

- `tests/unit/test_indices.py`: replace `model.group(...)`/`model.occasion(...)` with `model.level_code(...)`/`panel.position_code(...)` where the test is about the new API; keep one deprecation test for aliases.
- `tests/unit/test_canonical.py`: move panel-specific assertions to `tests/unit/test_panel.py`; keep math assertions for block months in `test_blocks.py`.
- `tests/unit/test_blackboard.py`: use `panel.symbols()`; keep one `model.symbols()` warning test.
- `tests/unit/test_dense.py`: update calls that relied on raw-coordinate balance; assert position-code balance instead.

- [ ] **Step 4: Run full verification**

```bash
uv run pytest --cov=symlmm --cov-fail-under=90
uv run mypy src/symlmm
uv run ruff check
uv run ruff format --check
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests src/symlmm CHANGELOG.md README.md docs
git commit -m "test: integrate factor-first hierarchy and panel APIs"
```

---

## Review Gates

Each workstream PR must pass:

1. **Spec gate:** does the change match the frozen object interfaces?
2. **Test gate:** new tests fail before implementation and pass after.
3. **Compatibility gate:** old public methods either work with warnings or are explicitly removed in Task 6.
4. **Concurrency gate:** no shared file edited by two worktrees without a merge note.

## Rollback

- Task 1 is a standalone commit; revert it to restore the old dense-code behavior.
- Task 2 is isolated in `decomposition.py`; revert it without touching block or panel APIs.
- Task 3a/3b are isolated in `blocks.py`; revert them to restore the single-factor `group_covariance`.
- Task 4 is isolated in `panel.py`; revert it to restore the old panel methods.
- Task 5 is the only compatibility layer; revert it last.
