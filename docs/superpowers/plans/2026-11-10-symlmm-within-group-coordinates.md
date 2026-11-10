# Within-Group Coordinates, Delta Resolution, and Nugget Effects — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate observation rows from group levels, within-group occasions, and coordinates; resolve deltas correctly for same-group queries; rename `time` to `coordinate`; add grouped spatial covariance and nugget effects.

**Architecture:** Index model A keeps `i` as the row cursor only. `group(i, factor)` returns a level index, `occasion(i, factor)` returns a within-group occasion, and `coordinate(i, name, within=factor)` returns `X[group, occasion]`. General expressions retain `KroneckerDelta`; a `within=` query resolves it to `1` for one factor while the default query keeps the cross-group expression. Spatial structures gain an optional group gate and generic coordinate names; residual structures gain additive nuggets.

**Tech Stack:** Python 3.14, SymPy 1.14, pytest 9, ruff 0.17, mypy 2.4, uv.

## Global Constraints

- Every public query method requires `form`; missing values raise `MissingFormError`.
- Exceptions raised by library code come from `symlmm.errors`.
- Internal indices are 0-based; rendered documentation is 1-based.
- Do not call `as_explicit()` outside dense mode.
- `python_version = "3.14"`, ruff `target-version = "py314"`, line length 100.
- `src/symlmm/__init__.py` is controller-owned; implementers must not edit it.
- Public functions/classes keep docstrings.
- Index model A is a containment choice. `src/symlmm/model.py` must say so and
  point at issue #1 for the full two-level `(group, occasion)` response index.
- Full gate: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`.

---

### Task 1: Index model A and coordinate rename

**Files:**
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/specs.py`
- Test: `tests/unit/test_indices.py`
- Test: `tests/unit/test_lmm_matrix.py`
- Test: `tests/unit/test_dense.py`
- Test: `tests/unit/test_random_slopes.py`
- Test: `tests/unit/test_closed_forms.py`
- Test: `tests/unit/test_moments.py`

**Interfaces:**
- Consumes: `Group`, `Nested`, `Crossed`, `Fixed`, `RandomIntercept`, `RandomSlope`, `ModelError`.
- Produces: `IndexedModel(coordinate=..., occasion=...)`, `coordinate_name`, `group(i, factor)` as a level index, `occasion(i, factor)`, `coordinate(i, name=None, within=None)`, `RandomSlope(group, coordinate, cov, within=True)`, `icc(at={"coordinate": ...})`.

- [ ] **Step 1: Write the failing accessor tests**

Replace the body of `tests/unit/test_indices.py` accessor expectations with:

```python
def test_observation_accessors_are_distinct_index_spaces():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), coordinate="coordinate")
    i, _ = model.indices()

    assert model.group(i, "subject") == sp.IndexedBase("g_subject", shape=(n,))[i]
    assert model.occasion(i, "subject") == sp.IndexedBase("k_subject", shape=(n,))[i]
    assert model.coordinate(i) == sp.IndexedBase("coordinate", shape=(n,))[i]


def test_within_coordinate_is_two_level():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), coordinate="visit_time")
    i, _ = model.indices()

    inside = model.coordinate(i, "visit_time", within="subject")
    assert inside == sp.IndexedBase("visit_time")[
        model.group(i, "subject"), model.occasion(i, "subject")
    ]


def test_missing_canonical_coordinate_raises():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),))
    i, _ = model.indices()

    with pytest.raises(ModelError, match="no coordinate name"):
        model.coordinate(i)
```

Update `test_nested_label_is_composite` and `test_crossed_label_is_composite`:

```python
assert label.base == sp.IndexedBase("g_subject_visit", shape=(n,))
assert label.base == sp.IndexedBase("g_subject_item", shape=(n,))
```

- [ ] **Step 2: Run the accessor tests to verify they fail**

Run: `uv run pytest tests/unit/test_indices.py -q`
Expected: FAIL on `g_*`, `k_*`, `coordinate`, and the missing-coordinate error.

- [ ] **Step 3: Implement index model A accessors**

At the top of `src/symlmm/model.py`, add:

```python
"""Model base classes and index accessors.

Index model A (temporary): ``i`` is only the observation-row cursor. Group
levels ``g`` and within-group occasions ``k`` are distinct structural index
spaces; coordinates live on ``G x K`` as ``X[g, k]``. The full two-level
response index redesign is tracked in issue #1.
"""
```

Change `IndexedModel.__init__` to:

```python
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
    self._i = observation_index("i")
    self._j = observation_index("j")
```

Replace `group` and add `occasion`/`coordinate`:

```python
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
    name: str | None = None,
    within: str | None = None,
) -> Any:
    """Return a row coordinate, or ``X[group(i), occasion(i)]`` when within."""
    selected = self.coordinate_name if name is None else name
    if selected is None:
        raise ModelError("no coordinate name; pass coordinate=... or coordinate(i, name)")
    if within is None:
        if self._dense:
            return self._observation_value(i, selected)
        return sp.IndexedBase(selected, shape=(self.n,))[i]
    spec = self._group_spec(within)
    if self._dense:
        return self._nested_coordinate_value(i, selected, spec)
    return sp.IndexedBase(selected)[self.group(i, within), self.occasion(i, within)]
```

Add dense occasion derivation and nested-coordinate validation. Cache
`_dense_group_codes` by `spec.label` too; repeated `group()` calls from
`_nested_coordinate_value` and `matrix_z` must not rebuilt the code list.

```python
def _occasion_column(self, factor: str) -> str | None:
    """Return the explicit occasion column for `factor`, if configured."""
    if isinstance(self.occasion_names, Mapping):
        return self.occasion_names.get(factor)
    return self.occasion_names

def _dense_occasion_codes(self, spec: GroupSpec) -> list[int]:
    """Return within-group integer occasion codes for `spec`."""
    cached = self._occasion_cache.get(spec.label)
    if cached is not None:
        return cached
    groups = self._dense_group_codes(spec)
    column = self._occasion_column(spec.name)
    if column is not None:
        raw = list(self._observation_column(column))
        mapping: dict[int, dict[Any, int]] = {}
        result = [0] * self.n
        for position, (group_code, value) in enumerate(zip(groups, raw, strict=True)):
            per_group = mapping.setdefault(group_code, {})
            if value not in per_group:
                per_group[value] = len(per_group)
            result[position] = per_group[value]
        self._occasion_cache[spec.label] = result
        return result
    if self.coordinate_name is not None:
        raw = list(self._observation_column(self.coordinate_name))
    else:
        raw = list(range(self.n))
    result = [0] * self.n
    for group_code in sorted(set(groups)):
        positions = [k for k, value in enumerate(groups) if value == group_code]
        ordered = sorted(positions, key=lambda k: (raw[k], k))
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
        for k in range(self.n):
            group = self.group(k, spec.name)
            occasion = self.occasion(k, spec.name)
            value = self._observation_value(k, name)
            previous = mapping.get((group, occasion), value)
            if previous != value:
                raise ModelError(
                    f"coordinate {name!r} is not constant within {spec.label!r}: "
                    f"({group}, {occasion}) has {previous!r} and {value!r}"
                )
            mapping[(group, occasion)] = value
        self._coordinate_cache[key] = mapping
    return mapping[(self.group(i, spec.name), self.occasion(i, spec.name))]
```

Explicit occasion codes are assigned per group: globally unique labels still
start at `0` within each group. Derived occasion codes sort unique coordinate
values within each group.

Update `design` in `IndexedModel`:

```python
terms = [] if self.coordinate_name is None else [self.coordinate(i)]
terms.extend(self.covariate(i, name) for name in self.covariates)
return sp.Matrix(terms) if terms else sp.zeros(0, 1)
```

In `dense()`, after copying state, reset all caches:

```python
dense._coordinate_cache = {}
dense._group_cache = {}
dense._occasion_cache = {}
```

- [ ] **Step 4: Rename `time` to `coordinate` in the model and specs**

In `src/symlmm/model.py`:

- `LMM.__init__(..., coordinate: str | None = None, occasion: str | Mapping[str, str] | None = None)`.
- Collect declared coordinates with `getattr(residual, "coordinate", None)` and
  `getattr(term, "coordinate", None)`.
- Error messages use "coordinate".
- `super().__init__(..., coordinate=coordinate_name, occasion=occasion, ...)`.
- `matrix_z` dense uses `term.coordinate` and `term.within`:
  ```python
  coordinate_name = getattr(term, "coordinate", None)
  within = getattr(term, "within", False)
  values = (
      [
          self.coordinate(
              k, coordinate_name, within=term.group if within else None
          )
          for k in range(self.n)
      ]
      if coordinate_name is not None
      else [0] * self.n
  )
  ```
- `icc` uses `term.coordinate`, `at.get("coordinate")`, then
  `at.get(self.coordinate_name)`, and substitution key
  `self.coordinate(0, term.coordinate, within=term.group if getattr(term, "within", False) else None)`.

In `src/symlmm/specs.py`:

```python
@dataclass(frozen=True)
class RandomSlope:
    """Random slope term in a named coordinate."""

    group: str
    coordinate: str
    cov: CovarianceSpec
    within: bool = True

    def __post_init__(self) -> None:
        if self.cov.dim != 2:
            raise CovarianceConstructionError("RandomSlope covariance must be 2x2")

    def z(self, model: IndexedModel, i: Any) -> sp.Matrix:
        """Return the random-effect design row for observation `i`."""
        value = model.coordinate(
            i, self.coordinate, within=self.group if self.within else None
        )
        return sp.Matrix([1, value])
```

- [ ] **Step 5: Write the failing dense accessor tests**

Append to `tests/unit/test_dense.py`:

```python
def test_dense_occasion_codes_are_within_group():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="visit_time",
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], visit_time=[10, 11, 10, 11])
    assert [dense.occasion(k, "subject") for k in range(4)] == [0, 1, 0, 1]
    assert dense.coordinate(0, "visit_time", within="subject") == 10
    assert dense.coordinate(1, "visit_time", within="subject") == 11


def test_dense_explicit_occurrence_codes_are_within_group():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="visit_time",
        occasion="occasion",
        n=4,
    )
    dense = model.dense(
        subject=[0, 0, 1, 1],
        occasion=["s1v1", "s1v2", "s2v1", "s2v2"],
        visit_time=[10, 11, 20, 21],
    )
    assert [dense.occasion(k, "subject") for k in range(4)] == [0, 1, 0, 1]


def test_dense_nested_coordinate_rejects_conflict():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="visit_time",
        occasion="occasion",
        n=4,
    )
    dense = model.dense(
        subject=[0, 0, 1, 1],
        occasion=[0, 0, 1, 1],
        visit_time=[10, 10, 11, 11],
    )
    conflicting = model.dense(
        subject=[0, 0, 1, 1],
        occasion=[0, 0, 1, 1],
        visit_time=[10, 11, 12, 13],
    )
    assert dense.coordinate(0, "visit_time", within="subject") == 10
    with pytest.raises(ModelError, match="not constant within"):
        conflicting.coordinate(1, "visit_time", within="subject")
```

- [ ] **Step 6: Run the dense tests**

Run: `uv run pytest tests/unit/test_indices.py tests/unit/test_dense.py tests/unit/test_random_slopes.py tests/unit/test_lmm_matrix.py -q`
Expected: PASS after updating remaining `time` names in these tests.

- [ ] **Step 7: Commit Task 1**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add src/symlmm/model.py src/symlmm/specs.py tests/unit/test_indices.py tests/unit/test_lmm_matrix.py tests/unit/test_dense.py tests/unit/test_random_slopes.py tests/unit/test_closed_forms.py tests/unit/test_moments.py
git commit -m "feat: separate row, group, occasion, and coordinate indices"
```

---

### Task 2: Within-group covariance structures, grouped spatial covariance, and nuggets

**Files:**
- Modify: `src/symlmm/covariances.py`
- Modify: `src/symlmm/model.py` (remove the transitional `time` alias and residual fallback)
- Test: `tests/unit/test_residual_covariances.py`
- Test: `tests/unit/test_spatial_covariances.py`
- Test: `tests/unit/test_closed_forms.py`
- Test: `tests/unit/test_random_slopes.py`
- Test: `tests/unit/test_dense.py`

**Interfaces:**
- Consumes: `model.coordinate(i, name, within=factor)`, `model.group(i, factor)`, `Constraint`, `Relation`.
- Produces: `AR1(group, coordinate, var, rho, nugget=0)`, `Toeplitz(group, coordinate, variances, correlations, nugget=0)`, `CompoundSymmetry(group, var, rho, m_max=2, nugget=0)`, `Exponential/Gaussian/Matern(var, range, group=None, coords=("x","y"), nugget=0)`, and no public `model.time` alias or residual `.time` fallback.

- [ ] **Step 1: Write the failing residual tests**

Add to `tests/unit/test_closed_forms.py`:

```python
def test_ar1_uses_two_level_within_coordinate():
    b0 = sp.Symbol("b0")
    sigma2, rho = sp.symbols("sigma2 rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
    )
    i, j = model.indices()
    inside = model.coordinate(i, "visit_time", within="subject")
    entry = model.residual.entry(i, j, model)
    assert inside in entry.atoms(sp.Indexed)
    assert sp.KroneckerDelta(
        model.group(i, "subject"), model.group(j, "subject")
    ) in entry.atoms(sp.KroneckerDelta)


def test_ar1_nugget_at_zero_lag_only():
    b0 = sp.Symbol("b0")
    sigma2, rho, omega2 = sp.symbols("sigma2 rho omega2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(
            group="subject",
            coordinate="visit_time",
            var=sigma2,
            rho=rho,
            nugget=omega2,
        ),
        coordinate="visit_time",
        n=2,
    )
    dense = model.dense(subject=[0, 0], visit_time=[0, 1])
    assert sp.simplify(dense.covariance(0, 0, form="sum")) == sigma2 + omega2
    assert sp.simplify(dense.covariance(0, 1, form="sum")) == sigma2 * rho


def test_nugget_constraint():
    sigma2, rho, omega2 = sp.symbols("sigma2 rho omega2", positive=True)
    cov = AR1(group="subject", coordinate="x", var=sigma2, rho=rho, nugget=omega2)
    assert any(
        constraint.message == "omega2 >= 0" for constraint in cov.constraints()
    )
```

Add to `tests/unit/test_spatial_covariances.py`:

```python
def test_spatial_group_gate_and_nugget():
    sigma2, phi, omega2 = sp.symbols("sigma2 phi omega2", positive=True)
    model = IndexedModel(n=2, groups=(Group("subject"),), coordinate="x")
    i, j = model.indices()
    cov = Exponential(var=sigma2, range=phi, group="subject", nugget=omega2)
    entry = cov.entry(i, j, model)
    assert sp.KroneckerDelta(
        model.group(i, "subject"), model.group(j, "subject")
    ) in entry.atoms(sp.KroneckerDelta)
    assert sp.simplify(entry.subs({i: 0, j: 0})) == sigma2 + omega2
```

- [ ] **Step 2: Run the residual tests to verify they fail**

Run: `uv run pytest tests/unit/test_closed_forms.py tests/unit/test_spatial_covariances.py -q`
Expected: FAIL on `coordinate=`, `nugget=`, and grouped spatial kwargs.

- [ ] **Step 3: Implement within coordinates and nuggets**

In `src/symlmm/covariances.py`, add:

```python
def _nonnegative(value: sp.Expr) -> Constraint:
    """Return a non-negativity constraint naming the actual expression."""
    return Constraint(value, Relation.GE, f"{value} >= 0")


def _nugget_entry(nugget: sp.Expr, i: Any, j: Any) -> sp.Expr:
    """Return the zero-lag measurement-error contribution."""
    return sp.Integer(0) if nugget == 0 else nugget * sp.KroneckerDelta(i, j)
```

For `AR1`:

```python
def __init__(
    self,
    *,
    group: str,
    coordinate: str,
    var: sp.Expr,
    rho: sp.Expr,
    nugget: sp.Expr = sp.Integer(0),
) -> None:
    self.group = group
    self.coordinate = coordinate
    self.var = var
    self.rho = rho
    self.nugget = nugget

def entry(self, i: int, j: int, model: Any) -> sp.Expr:
    """Return the (i, j) observation entry."""
    ti = model.coordinate(i, self.coordinate, within=self.group)
    tj = model.coordinate(j, self.coordinate, within=self.group)
    same = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
    return self.var * self.rho ** sp.Abs(ti - tj) * same + _nugget_entry(self.nugget, i, j)

def constraints(self) -> tuple[Constraint, ...]:
    values = [
        Constraint(self.rho + 1, Relation.GT, "-1 < rho"),
        Constraint(1 - self.rho, Relation.GT, "rho < 1"),
        _positive(self.var),
    ]
    if self.nugget != 0:
        values.append(_nonnegative(self.nugget))
    return tuple(values)
```

Apply the same pattern to `Toeplitz`:
`ti/tj` via within coordinate; `_lag_value(lag) * same + _nugget_entry(...)`.

For `CompoundSymmetry`:
`Piecewise((self.var + self.nugget, sp.Eq(i, j)), (self.var * self.rho * same, True))`;
append `_nonnegative(self.nugget)` when nonzero.

For spatial structures, add fields:

```python
def __init__(
    self,
    *,
    var: sp.Expr,
    range: sp.Expr,
    group: str | None = None,
    coords: Sequence[str] = ("x", "y"),
    nugget: sp.Expr = sp.Integer(0),
) -> None:
    self.var = var
    self.range = range
    self.group = group
    self.coords = tuple(coords)
    self.nugget = nugget
```

Replace `_euclidean_distance`:

```python
def _euclidean_distance(
    i: int,
    j: int,
    model: Any,
    coords: Sequence[str],
    group: str | None,
) -> sp.Expr:
    total = sp.Integer(0)
    for name in coords:
        delta = model.coordinate(i, name, within=group) - model.coordinate(j, name, within=group)
        total += delta**2
    return sp.sqrt(total)
```

Spatial `entry`:

```python
def entry(self, i: int, j: int, model: Any) -> sp.Expr:
    distance = _euclidean_distance(i, j, model, self.coords, self.group)
    value = self.entry_from_distance(distance)
    if self.group is not None:
        value *= sp.KroneckerDelta(
            model.group(i, self.group), model.group(j, self.group)
        )
    return value + _nugget_entry(self.nugget, i, j)
```

Spatial constraints append `_nonnegative(self.nugget)` when nonzero.

- [ ] **Step 4: Update existing covariance tests**

Rename `time=` to `coordinate=` in tests. Update `test_ar1_uses_named_time_coordinate_as_canonical_symbol` to assert `ar1.coordinate == "visit_time"` and the two-level indexed atom.

Add dense same-subject/cross-subject tests:

```python
def test_dense_ar1_same_subject_adjacent_and_cross_subject():
    b0 = sp.Symbol("b0")
    sigma2, rho = sp.symbols("sigma2 rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
        coordinate="visit_time",
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], visit_time=[0, 1, 0, 1])
    assert sp.simplify(dense.covariance(0, 1, form="sum")) == sigma2 * rho
    assert dense.covariance(1, 2, form="sum") == 0
```

- [ ] **Step 5: Remove the transitional `time` alias**

After all covariance structures use `coordinate`, delete `IndexedModel.time` from
`src/symlmm/model.py`. In `LMM.__init__`, remove the `getattr(residual, "time", None)`
fallback and read only `getattr(residual, "coordinate", None)`.

Add to `tests/unit/test_indices.py`:

```python
def test_public_time_alias_is_removed():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), coordinate="visit_time")
    assert not hasattr(model, "time")
```

- [ ] **Step 6: Run the task gate and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add src/symlmm/covariances.py src/symlmm/model.py tests/unit/test_indices.py tests/unit/test_residual_covariances.py tests/unit/test_spatial_covariances.py tests/unit/test_closed_forms.py tests/unit/test_random_slopes.py tests/unit/test_dense.py
git commit -m "feat: add within-group coordinates, spatial gates, and nuggets"
```

---

### Task 3: Within-group delta resolution for queries

**Files:**
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/moments.py`
- Modify: `src/symlmm/variogram.py`
- Test: `tests/unit/test_moments.py`
- Test: `tests/unit/test_variograms.py`
- Test: `tests/unit/test_dense.py`
- Test: `tests/unit/test_closed_forms.py`

**Interfaces:**
- Consumes: `group(i, factor)` level indices, `covariance_entry`.
- Produces: `covariance_entry(i, j, within=None)`, `covariance(..., within=None)`, `variance(..., within=None)`, `correlation(..., within=None)`, `variogram(..., within=None)`, `covariogram(..., within=None)`.

- [ ] **Step 1: Write the failing delta-resolution tests**

Append to `tests/unit/test_moments.py`:

```python
def test_within_resolves_delta_for_same_group():
    b0 = sp.Symbol("b0")
    tau2, sigma2, rho = sp.symbols("tau2 sigma2 rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
        coordinate="visit_time",
    )
    i, j = model.indices()

    general = model.covariance(i, j, form="sum")
    inside = model.covariance(i, j, form="sum", within="subject")

    assert general.has(sp.KroneckerDelta)
    assert not inside.has(sp.KroneckerDelta)
    assert inside != general
```

Add to `tests/unit/test_dense.py`:

```python
def test_within_dense_same_and_cross_group():
    b0 = sp.Symbol("b0")
    tau2, sigma2, rho = sp.symbols("tau2 sigma2 rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
        coordinate="visit_time",
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], visit_time=[0, 1, 0, 1])
    assert sp.simplify(
        dense.covariance(0, 1, form="sum", within="subject") - (tau2 + sigma2 * rho)
    ) == 0
    assert dense.covariance(1, 2, form="sum") == 0
```

- [ ] **Step 2: Implement `within` resolution**

In `LMM`:

```python
def _apply_within(self, expr: Any, i: Any, j: Any, within: str) -> Any:
    """Resolve a factor's delta by placing both rows in one common level."""
    spec = self._group_spec(within)
    common = sp.Symbol(f"g_{spec.label.replace(':', '_')}", integer=True)
    return expr.subs({self.group(i, within): common, self.group(j, within): common})

def covariance_entry(self, i: Any, j: Any, within: str | None = None) -> Any:
    """Return the structured sum-form covariance entry."""
    total = self.residual.entry(i, j, self)
    for term in self.random:
        total += self._random_term_entry(term, i, j)
    if within is not None and not self._dense:
        total = self._apply_within(total, i, j, within)
    return total
```

Dense queries already use concrete level codes, so `KroneckerDelta` resolves to
`1`/`0` without substitution. Do not call `_apply_within` in dense mode; numeric
substitution of a level code would rewrite unrelated literals.

Update `LMM.covariance`, `.variance`, `.correlation`, `.variogram`, `.covariogram` to accept `within: str | None = None` and pass it to the module functions.

In `moments.py`, add `within` to `covariance`, `variance`, `correlation`, and all `EntryMap` closures. In `variogram.py`, add `within` to `_residual_variogram`, `_conditional_variogram` (via `model._apply_within`), `covariogram`, and `variogram`.

- [ ] **Step 3: Run the query tests**

Run: `uv run pytest tests/unit/test_moments.py tests/unit/test_variograms.py tests/unit/test_dense.py tests/unit/test_closed_forms.py -q`
Expected: PASS.

- [ ] **Step 4: Run the task gate and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add src/symlmm/model.py src/symlmm/moments.py src/symlmm/variogram.py tests/unit/test_moments.py tests/unit/test_variograms.py tests/unit/test_dense.py tests/unit/test_closed_forms.py
git commit -m "feat: resolve within-group deltas in moment queries"
```

---

### Task 4: Documentation, M3 plan, and changelog

**Files:**
- Modify: `PLAN.md`
- Modify: `docs/superpowers/plans/2026-11-09-symlmm-marginal-gee.md`
- Modify: `CHANGELOG.md`
- Test: no code test; verify docs contain no public `time` coordinate names.

**Interfaces:**
- Produces: current architecture docs and M3 plan aligned with index model A, `coordinate`, `occasion(i, factor)`, and issue #1.

- [ ] **Step 1: Update `PLAN.md`**

Replace public coordinate examples:

```python
model.group(i, "subject")       # g_subject[i], level index
model.occasion(i, "subject")    # k_subject[i], within-group occasion
model.coordinate(i)             # coordinate[i], global row coordinate
model.coordinate(i, "visit_time", within="subject")
# visit_time[g_subject[i], k_subject[i]]
```

State that `i` is a row cursor, group and occasion are distinct index spaces, and
index model A is temporary with issue #1 tracking the full `(g, k)` redesign.
Update `RandomSlope`, `AR1`, `Toeplitz`, `Exponential`, `Gaussian`, `Matern`,
`icc(at=...)`, and the M3/GEE occasion description.

- [ ] **Step 2: Update the M3 plan**

In `docs/superpowers/plans/2026-11-09-symlmm-marginal-gee.md`:

- Rename `time` -> `coordinate` in `WorkingAR1`, `IndexedModel`, and tests.
- Replace `GEE.occasion(i)` with `GEE.occasion(i, cluster)` using the true
  within-cluster occasion.
- Update `icc(at={"time": ...})` to `at={"coordinate": ...}`.
- Ensure `MarginalLMM` passes `coordinate=` to `IndexedModel`.

- [ ] **Step 3: Update the changelog**

Add a `0.2.0` section:

```markdown
## [0.2.0] - 2026-11-10

### Added

- `model.occasion(i, factor)` and two-level `model.coordinate(i, name, within=factor)`.
- `within=` resolution for covariance, variance, correlation, variogram, and covariogram.
- Grouped spatial covariances with `group=` and generic `coords=`.
- Additive `nugget=` on AR(1), Toeplitz, compound-symmetry, and spatial covariances.

### Changed

- Public `time` names replaced by `coordinate`; `model.time` and `time=` are removed.
- `RandomSlope` uses a within-group coordinate by default.
- Coordinate access without a canonical name raises `ModelError` instead of falling back to `t`.

### Fixed

- Spatial covariances no longer return nonzero cross-group covariance when `group=` is supplied.
- Group levels, occasions, and coordinates no longer share the observation row index.
```

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add PLAN.md docs/superpowers/plans/2026-11-09-symlmm-marginal-gee.md CHANGELOG.md
git commit -m "docs: document index model A and within-group coordinates"
```

---

## Final Gate

- [ ] `uv run pytest -q`
- [ ] `uv run ruff check`
- [ ] `uv run ruff format --check`
- [ ] `uv run mypy src/symlmm`
- [ ] Whole-branch review on the most capable model.
- [ ] Confirm issue #1 is linked from `src/symlmm/model.py` and the design doc.
- [ ] Merge to `main` with `--no-ff`, push, and clean up the worktree.
