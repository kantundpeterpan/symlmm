# RandomEffects and Deferred-Minor Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace covariance-bearing `RandomIntercept`/`RandomSlope` terms with one joint `RandomEffects` block per group, then clear the remaining M2.6 minors.

**Architecture:** `RandomSlope` becomes a pure design definition. `RandomEffects` owns a group, intercept flag, slope definitions, and the joint covariance. `LMM` treats each `RandomEffects` as one block: `Z` columns per level, `G` block per level, joint conditional covariance. This is a clean 0.3.0 break.

**Tech Stack:** Python 3.14, SymPy 1.14, pytest 9, ruff 0.17, mypy 2.4, uv.

## Global Constraints

- Every public query method requires `form`; missing values raise `MissingFormError`.
- Exceptions raised by library code come from `symlmm.errors`.
- Internal indices are 0-based; rendered documentation is 1-based.
- Do not call `as_explicit()` outside dense mode.
- `python_version = "3.14"`, ruff `target-version = "py314"`, line length 100.
- `src/symlmm/__init__.py` is controller-owned; implementers must not edit it.
- Public functions/classes keep docstrings.
- Full gate: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`.

---

### Task 1: RandomEffects core and LMM assembly

**Files:**
- Modify: `src/symlmm/specs.py`
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/moments.py`
- Test: `tests/unit/test_random_effects.py` (new)
- Test: all files currently importing `RandomIntercept`/`RandomSlope`

**Interfaces:**
- Consumes: `CovarianceSpec`, `ModelError`, `CovarianceConstructionError`, `Group`, `Fixed`.
- Produces: `RandomSlope(coordinate, within=True)` as a pure definition; `RandomEffects(group, cov, intercept=False, slopes=())`; `LMM(random: Iterable[RandomEffects])`.

- [ ] **Step 1: Write the failing RandomEffects tests**

Create `tests/unit/test_random_effects.py`:

```python
"""Joint random-effects block tests."""

import pytest
import sympy as sp

from symlmm.covariances import Diagonal, Scalar, Unstructured
from symlmm.errors import CovarianceConstructionError, ModelError
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects, RandomSlope


def test_intercept_only_effects_have_dim_one():
    tau2 = sp.Symbol("tau2", positive=True)
    effects = RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))
    assert effects.dim == 1


def test_joint_intercept_slope_effects_keep_covariance():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    effects = RandomEffects(
        group="subject",
        intercept=True,
        slopes=(RandomSlope("time"),),
        cov=Unstructured.from_sd_corr([s0, s1], rho),
    )
    assert effects.dim == 2
    assert effects.cov.dim == 2


def test_slopes_only_effects_are_supported():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    effects = RandomEffects(
        group="subject",
        slopes=(RandomSlope("time"), RandomSlope("dose")),
        cov=Diagonal([s0, s1]),
    )
    assert effects.dim == 2
    assert effects.coordinates() == ("time", "dose")


def test_effects_require_design_and_matching_covariance():
    tau2 = sp.Symbol("tau2", positive=True)
    with pytest.raises(CovarianceConstructionError, match="intercept or slope"):
        RandomEffects(group="subject", cov=Scalar(tau2))
    with pytest.raises(CovarianceConstructionError, match="2x2"):
        RandomEffects(
            group="subject",
            intercept=True,
            slopes=(RandomSlope("time"),),
            cov=Scalar(tau2),
        )


def test_duplicate_group_blocks_raise():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    with pytest.raises(ModelError, match="multiple RandomEffects"):
        LMM(
            fixed=Fixed(intercept=sp.Symbol("b0")),
            random=[
                RandomEffects(group="subject", intercept=True, cov=Scalar(tau2)),
                RandomEffects(group="subject", intercept=True, cov=Scalar(tau2)),
            ],
            residual=__import__("symlmm").IID(var=sigma2),
        )


def test_joint_covariance_entry_uses_one_block():
    b0 = sp.Symbol("b0")
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    effects = RandomEffects(
        group="subject",
        intercept=True,
        slopes=(RandomSlope("time"),),
        cov=Unstructured.from_sd_corr([s0, s1], rho),
    )
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[effects],
        residual=__import__("symlmm").IID(var=sigma2),
    )
    i, j = model.indices()
    ti = model.coordinate(i, "time", within="subject")
    tj = model.coordinate(j, "time", within="subject")
    expected = (s0**2 + s0 * s1 * rho * (ti + tj) + s1**2 * ti * tj) * sp.KroneckerDelta(
        model.group(i, "subject"), model.group(j, "subject")
    ) + sigma2 * sp.KroneckerDelta(i, j)
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0
```

Use the normal `from symlmm.covariances import IID` import instead of
`__import__` when writing the file.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/test_random_effects.py -q`
Expected: FAIL on missing `RandomEffects`, changed `RandomSlope`, and duplicate-group validation.

- [ ] **Step 3: Implement the specs**

Replace the old `RandomIntercept`/`RandomSlope` classes in `src/symlmm/specs.py` with:

```python
@dataclass(frozen=True)
class RandomSlope:
    """A random-slope column in a named coordinate; no covariance."""

    coordinate: str
    within: bool = True


@dataclass(frozen=True)
class RandomEffects:
    """One group's joint random-effect design and covariance."""

    group: str
    cov: CovarianceSpec
    intercept: bool = False
    slopes: tuple[RandomSlope, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slopes", tuple(self.slopes))
        if not self.intercept and not self.slopes:
            raise CovarianceConstructionError("RandomEffects requires an intercept or slope")
        if self.cov.dim != self.dim:
            raise CovarianceConstructionError(
                f"RandomEffects covariance must be {self.dim}x{self.dim}"
            )

    @property
    def dim(self) -> int:
        """Number of random coefficients in this block."""
        return int(self.intercept) + len(self.slopes)

    def coordinates(self) -> tuple[str, ...]:
        """Return the slope coordinate names in design order."""
        return tuple(slope.coordinate for slope in self.slopes)

    def z(self, model: IndexedModel, i: Any) -> sp.Matrix:
        """Return the random-effect design column for observation `i`."""
        columns: list[Any] = [sp.Integer(1)] if self.intercept else []
        columns.extend(
            model.coordinate(
                i,
                slope.coordinate,
                within=self.group if slope.within else None,
            )
            for slope in self.slopes
        )
        return sp.Matrix(columns)
```

- [ ] **Step 4: Migrate `LMM` assembly**

In `src/symlmm/model.py`:

- Import `RandomEffects` instead of `RandomIntercept, RandomSlope`.
- `random: Iterable[RandomEffects]`.
- After `self.random = tuple(random)`, reject duplicate groups:
  ```python
  seen_groups: set[str] = set()
  for term in self.random:
      if term.group in seen_groups:
          raise ModelError(
              f"multiple RandomEffects blocks for group {term.group!r}; "
              "combine intercepts and slopes into one block"
          )
      seen_groups.add(term.group)
  ```
- Group registration: `registered.setdefault(term.group, Group(term.group))`.
- `dimensions`: `q = sum(self.levels(term.group) * term.dim for term in self.random)`.
- `_random_term_entry`:
  ```python
  def _random_term_entry(self, term: RandomEffects, i: Any, j: Any) -> Any:
      zi = term.z(self, i)
      zj = term.z(self, j)
      value = (zi.T * term.cov.dense() * zj)[0, 0]
      return value * sp.KroneckerDelta(
          self.group(i, term.group), self.group(j, term.group)
      )
  ```
- `matrix_z` dense:
  ```python
  for term in self.random:
      labels = self._dense_group_codes(self._group_spec(term.group))
      designs = [term.z(self, k) for k in range(self.n)]
      for level in sorted(set(labels)):
          for a in range(term.dim):
              columns.append(
                  sp.Matrix(
                      [
                          designs[k][a] if labels[k] == level else 0
                          for k in range(self.n)
                      ]
                  )
              )
  ```
- `matrix_g` dense: `blocks.extend([term.cov.dense()] * len(labels))`.

In `src/symlmm/moments.py`, `_conditional_mean_entry`:

```python
for term_index, term in enumerate(model.random):
    z = term.z(model, i)
    base = f"b_{term.group}_{term_index}".replace(":", "_")
    level = model.group(i, term.group)
    for a in range(term.dim):
        value += z[a] * sp.IndexedBase(base)[level, a]
```

- [ ] **Step 5: Migrate existing tests and examples mechanically**

Apply these mappings everywhere:

```text
RandomIntercept(group=G, cov=C)
    -> RandomEffects(group=G, intercept=True, cov=C)

RandomSlope(group=G, coordinate=X, cov=C)
    -> RandomEffects(group=G, intercept=True, slopes=(RandomSlope(X),), cov=C)

RandomSlope(group=G, coordinate=X, cov=C, within=W)
    -> RandomEffects(group=G, intercept=True, slopes=(RandomSlope(X, within=W),), cov=C)
```

When a test previously had both `RandomIntercept` and `RandomSlope` on the same
group and expected independent blocks, either:

- combine them with a joint covariance if the test is about the general model; or
- preserve independence explicitly with
  `cov=BlockDiag([Scalar(tau2), old_slope_cov])`.

Update imports from `RandomIntercept, RandomSlope` to
`RandomEffects, RandomSlope`; remove `RandomIntercept` imports.

- [ ] **Step 6: Run the full task gate and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add src/symlmm/specs.py src/symlmm/model.py src/symlmm/moments.py tests
git commit -m "feat: add joint RandomEffects blocks"
```

---

### Task 2: Coordinates, ICC, and BLUP with RandomEffects

**Files:**
- Modify: `src/symlmm/model.py`
- Test: `tests/unit/test_indices.py`
- Test: `tests/unit/test_closed_forms.py`
- Test: `tests/unit/test_blup.py`
- Test: `tests/integration/test_blup_dense.py`
- Test: `tests/unit/test_random_effects.py`

**Interfaces:**
- Consumes: `RandomEffects.dim`, `.z`, `.coordinates`, `.cov`.
- Produces: multi-coordinate `LMM` random slopes; `icc(at=...)` keyed by coordinate name; BLUP over joint blocks.

- [ ] **Step 1: Write the failing coordinate/ICC tests**

Add to `tests/unit/test_random_effects.py`:

```python
def test_multiple_slope_coordinates_are_allowed():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=sp.Symbol("b0")),
        random=[
            RandomEffects(
                group="subject",
                slopes=(RandomSlope("time"), RandomSlope("dose", within=False)),
                cov=Diagonal([s0, s1]),
            )
        ],
        residual=IID(var=sigma2),
    )
    assert model.coordinate_name is None


def test_icc_accepts_coordinate_mapping_for_multiple_slopes():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=sp.Symbol("b0")),
        random=[
            RandomEffects(
                group="subject",
                slopes=(RandomSlope("time"), RandomSlope("dose")),
                cov=Diagonal([s0, s1]),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )
    value = model.icc(at={"time": 2, "dose": 3}, form="sum")
    assert not value.has(sp.Indexed)
```

- [ ] **Step 2: Implement canonical-coordinate relaxation**

In `LMM.__init__`:

```python
residual_coordinate = getattr(residual, "coordinate", None)
slope_coordinates = tuple(
    dict.fromkeys(
        slope.coordinate
        for term in self.random
        for slope in term.slopes
    )
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
```

Do not raise for multiple slope coordinates.

- [ ] **Step 3: Implement multi-coordinate ICC**

Replace `LMM.icc` time-term handling with:

```python
coordinate_slopes = [
    (term, slope)
    for term in self.random
    for slope in term.slopes
]
if coordinate_slopes and at is None:
    raise ModelError("icc requires at={'coordinate': value} or coordinate mapping")
values: dict[str, Any] = {}
if at is not None:
    if "coordinate" in at and self.coordinate_name is not None:
        values[self.coordinate_name] = at["coordinate"]
    values.update({key: value for key, value in at.items() if key != "coordinate"})
for term in self.random:
    z = term.z(self, 0)
    if self._dense:
        columns: list[Any] = [sp.Integer(1)] if term.intercept else []
        for slope in term.slopes:
            if slope.coordinate not in values:
                raise ModelError(
                    f"icc requires a value for coordinate {slope.coordinate!r}"
                )
            columns.append(values[slope.coordinate])
        z = sp.Matrix(columns)
    else:
        for slope in term.slopes:
            if slope.coordinate not in values:
                raise ModelError(
                    f"icc requires a value for coordinate {slope.coordinate!r}"
                )
            key = self.coordinate(
                0,
                slope.coordinate,
                within=term.group if slope.within else None,
            )
            z = z.subs({key: values[slope.coordinate]})
    random_part += (z.T * term.cov.dense() * z)[0, 0]
```

Keep residual substitution for a declared canonical coordinate.

- [ ] **Step 4: Add BLUP and dense checks**

Add to `tests/unit/test_blup.py` and `tests/integration/test_blup_dense.py` a model with
`RandomEffects(intercept=True, slopes=(RandomSlope("time"),), cov=...)` and assert
`blup_mean`, `blup_covariance`, and `blup_cross_covariance` shapes/values match
the joint `matrix_z`/`matrix_g` assembly.

- [ ] **Step 5: Run the task gate and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add src/symlmm/model.py tests/unit/test_indices.py tests/unit/test_closed_forms.py tests/unit/test_blup.py tests/integration/test_blup_dense.py tests/unit/test_random_effects.py
git commit -m "feat: support multi-coordinate random effects and ICC"
```

---

### Task 3: Deferred M2.6 minors

**Files:**
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/covariances.py`
- Test: `tests/unit/test_indices.py`
- Test: `tests/unit/test_dense.py`
- Test: `tests/unit/test_spatial_covariances.py`
- Test: `tests/unit/test_residual_covariances.py`

**Interfaces:**
- Consumes: existing model/covariance APIs.
- Produces: no API change; fixes correctness/coverage/doc gaps.

- [ ] **Step 1: Fix base design double-counting**

In `IndexedModel.design`:

```python
terms = [] if self.coordinate_name is None else [self.coordinate(i)]
terms.extend(
    self.covariate(i, name)
    for name in self.covariates
    if name != self.coordinate_name
)
return sp.Matrix(terms) if terms else sp.zeros(0, 1)
```

Add a test where `coordinate="time"` and `covariates=("time", "x")`; assert
`design(i)` has two entries, not three.

- [ ] **Step 2: Consolidate nugget constraints and `_ZERO`**

Add:

```python
def _append_nugget_constraint(
    constraints: list[Constraint], nugget: sp.Expr
) -> None:
    """Append the non-negativity constraint for a nonzero nugget."""
    if nugget != _ZERO:
        constraints.append(_nonnegative(nugget))
```

Use it in `AR1`, `Toeplitz`, `CompoundSymmetry`, `Exponential`, `Gaussian`,
`Matern`. Make `_nugget_entry` return `_ZERO` for zero.

- [ ] **Step 3: Fix `_apply_within` symbol churn**

Add `self._within_symbols: dict[str, sp.Symbol] = {}` in `IndexedModel.__init__`
and a helper:

```python
def _within_symbol(self, spec: GroupSpec) -> sp.Symbol:
    key = spec.label
    symbol = self._within_symbols.get(key)
    if symbol is None:
        symbol = sp.Symbol(f"g_{key.replace(':', '_')}", integer=True)
        self._within_symbols[key] = symbol
    return symbol
```

Use it in `_apply_within`.

- [ ] **Step 4: Make derived occasion sorting robust**

In `_dense_occasion_codes`, replace the raw sort key with:

```python
def _occasion_sort_key(value: Any, position: int) -> tuple[Any, ...]:
    try:
        return (0, sp.default_sort_key(sp.sympify(value)), position)
    except (sp.SympifyError, TypeError):
        return (1, str(value), position)
```

Use `key=lambda k: _occasion_sort_key(raw[k], k)`. Document that numeric
`1` and `1.0` are distinct coordinate values for nesting validation.

- [ ] **Step 5: Add missing coverage and docstrings**

- Dense grouped-spatial cross-group zero:
  `Exponential(group="subject")` with different dense subjects must return `0`.
- Per-factor `occasion` mapping: `occasion={"subject": "visit", "item": "occasion"}`.
- Covariance docstrings: document `coordinate`, `nugget`, `group`, and `coords`
  on `AR1`, `Toeplitz`, `CompoundSymmetry`, `Exponential`, `Gaussian`, `Matern`.

- [ ] **Step 6: Run the task gate and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add src/symlmm/model.py src/symlmm/covariances.py tests/unit/test_indices.py tests/unit/test_dense.py tests/unit/test_spatial_covariances.py tests/unit/test_residual_covariances.py
git commit -m "fix: clear deferred M2.6 minors"
```

---

### Task 4: Docs, exports, version

**Files:**
- Modify: `PLAN.md`
- Modify: `docs/superpowers/plans/2026-11-09-symlmm-marginal-gee.md`
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml`
- Controller: `src/symlmm/__init__.py`

**Interfaces:**
- Produces: public exports `RandomEffects`, `RandomSlope`; remove `RandomIntercept`;
  version 0.3.0; docs and M3 plan aligned.

- [ ] **Step 1: Update docs**

- `PLAN.md` §3.1/§6.1: replace the `RandomIntercept`/`RandomSlope` description
  with `RandomEffects(group, cov, intercept, slopes)`.
- M3 plan: replace `RandomIntercept` constructions with `RandomEffects`.
- Changelog 0.3.0: breaking removal of `RandomIntercept` and covariance-bearing
  `RandomSlope`; joint `RandomEffects`; migration mapping; deferred minors.
- `pyproject.toml`: version `0.3.0`.
- Run `uv lock` after the version bump and include `uv.lock` if it changes.

- [ ] **Step 2: Controller export update**

After implementation, update `src/symlmm/__init__.py`:

```python
from .specs import Crossed, Fixed, Group, Nested, RandomEffects, RandomSlope
```

Remove `RandomIntercept` from imports and `__all__`; add `RandomEffects`.
Run `ruff --fix` to sort `__all__`.

- [ ] **Step 3: Verify and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add PLAN.md docs/superpowers/plans/2026-11-09-symlmm-marginal-gee.md CHANGELOG.md pyproject.toml uv.lock src/symlmm/__init__.py
git commit -m "docs: document RandomEffects and bump to 0.3.0"
```

---

## Final Gate

- [ ] `uv run pytest -q`
- [ ] `uv run ruff check`
- [ ] `uv run ruff format --check`
- [ ] `uv run mypy src/symlmm`
- [ ] Whole-branch review on the most capable model.
- [ ] Merge to `main` with `--no-ff`, push, clean up the worktree.
