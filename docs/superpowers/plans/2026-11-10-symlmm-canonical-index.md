# Canonical Group/Occasion Index — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Y_i` the group response vector and `Y_{it}` its occasion component, resolving issue #1 and replacing the row-index blackboard notation.

**Architecture:** Keep the stacked observation engine for matrix/dense computation, but add a canonical public layer: group index, group size, response vector, group mean/covariance blocks, and per-occasion entries. Rebuild the blackboard facade on that layer.

**Tech Stack:** Python 3.14, SymPy 1.14, pytest 9, ruff 0.17, mypy 2.4, uv.

## Global Constraints

- Every public query method requires `form`; missing values raise `MissingFormError`.
- Exceptions raised by library code come from `symlmm.errors`.
- Internal indices are 0-based; rendered documentation is 1-based.
- Do not call `as_explicit()` outside dense mode.
- `python_version = "3.14"`, ruff `target-version = "py314"`, line length 100.
- `src/symlmm/__init__.py` is controller-owned; implementers must not edit it.
- Public functions/classes keep docstrings.
- Canonical notation: `i` indexes group/cluster/subject; `t` indexes occasion.
- `Y_i` is a vector; `Y_{it}` is a scalar. The stacked row `i` is not public notation.
- Full gate: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`.

## Concurrency Plan

**Verdict:** one code implementer at a time. Optional one docs agent after Task 2.
Peak implementation concurrency: 2. Do not run Tasks 0–3 in parallel.

### Dependency graph

```text
Task 0 (index-bearing definitions)
   └── Task 1 (canonical symbols)
          └── Task 2 (group/time moments)
                 ├── Task 3 (blackboard)
                 └── Task 4 (docs, parallel with Task 3)
```

### File-conflict matrix

| Task | Writes | Conflicts with |
|---|---|---|
| 0 | `specs.py`, `model.py`, `test_specs.py`, `test_indices.py` | 1, 2, 3 |
| 1 | `canonical.py`, `model.py`, `test_canonical.py` | 0, 2, 3 |
| 2 | `model.py`, `test_canonical.py`, dense integration test | 1, 3 |
| 3 | `blackboard.py`, `test_blackboard.py`, `test_canonical.py` | 1, 2 |
| 4 | docs, `pyproject.toml`, `CHANGELOG.md`, `uv.lock` | none (docs worktree) |

### Schedule

1. Task 0 implementer → task review → fix.
2. Task 1 implementer → review → fix.
3. Task 2 implementer → review → fix.
4. Start Task 4 docs in a separate worktree, parallel with Task 3.
5. Task 3 implementer → review → fix.
6. Merge docs into code branch; controller updates `__init__.py`; run final
   whole-branch review; close issue #1.

### Worktrees

- Code: `feat/canonical-index`, one worktree, sequential commits.
- Docs: `docs/canonical-index`, branched from the code branch after Task 2,
  merged or cherry-picked before final review.
- No per-task worktrees: Tasks 0–3 share `model.py` and test files.

### Agent policy

- Implementers: Flash `max`.
- Task reviewers: Flash `max`.
- Final review: Pro `default`.
- Controller owns exports, issue closure, merge, cleanup.

---

### Task 0: Index-bearing group and covariate definitions

**Files:**
- Modify: `src/symlmm/specs.py`
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/covariances.py`
- Test: `tests/unit/test_specs.py`
- Test: `tests/unit/test_indices.py`
- Test: `tests/unit/test_residual_covariances.py`

**Interfaces:**
- Produces: `Group(name, index=...)`, `Nested(child, within, index=...)`,
  `Crossed(left, right, index=...)`, `Covariate(name, index=...)`;
  `Fixed.slopes` keyed by `Covariate`; `RandomEffects.group: Group`;
  `RandomSlope.coordinate: Covariate`; `AR1/Toeplitz(group=Group,
  coordinate=Covariate, ...)`.

- [ ] **Step 1: Add index fields**

```python
@dataclass(frozen=True)
class Group:
    name: str
    index: sp.Symbol | None = None

    @property
    def label(self) -> str:
        return self.name
```

Add the same optional `index` to `Nested` and `Crossed`. Add:

```python
@dataclass(frozen=True)
class Covariate:
    """A named covariate/coordinate definition with its index symbol."""

    name: str
    index: sp.Symbol | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            object.__setattr__(self, "index", sp.Symbol(self.name, integer=True))
```

Default indices for `Group/Nested/Crossed` are derived from their label when
`index` is omitted.

- [ ] **Step 2: Normalize string compatibility**

`Fixed.slopes`, `RandomEffects.group`, `RandomSlope.coordinate`, `AR1.group`,
`AR1.coordinate`, `Toeplitz.group`, and `Toeplitz.coordinate` accept either the
new definitions or the existing string forms. Normalize strings to
`Group(name, index=sp.Symbol(name, integer=True))` and
`Covariate(name, index=sp.Symbol(name, integer=True))` at construction.

- [ ] **Step 3: Derive model indices**

`LMM` stores:

```python
primary = self.random[0].group if self.random else ...
self.group_index = primary.index
self.occasion_index = canonical_covariate.index
```

`group_index` and `occasion_index` must be distinct. `model.symbols()` uses these
symbols, never a hardcoded `i`/`t`.

- [ ] **Step 4: Tests and commit**

Test:
- `Group("subject", index=i).index is i`;
- default group/covariate indices are derived from names;
- `Fixed.slopes={Covariate("time", index=t): b1}`;
- `RandomSlope(Covariate("time", index=t))`;
- `AR1(group=subject, coordinate=visit)`;
- string forms normalize with equal semantics;
- equal group/occasion symbols raise `ModelError`.

Run full gate and commit:
`git commit -m "feat: attach indices to group and covariate definitions"`

---

### Task 1: Canonical group index and response symbols

**Files:**
- Create: `src/symlmm/canonical.py`
- Modify: `src/symlmm/model.py`
- Test: `tests/unit/test_canonical.py` (new)

**Interfaces:**
- Consumes: `LMM`, `RandomEffects`, `Fixed`, `IndexedModel`.
- Produces: `model.group_index`, `model.occasion_index`, `model.primary_group`, `model.group_count()`, `model.occasion_count()`, `model.group_size(g)`, `model.response_matrix()`, `model.response(g, t=None)`, `model.time_response(t)`, `model.fixed_design(g)`, `model.random_design(g)`, `model.residual_covariance(g)`, `model.random_effect(g)`. Index symbols come from the `Group`/`Covariate` definitions from Task 0.

- [ ] **Step 1: Write the failing canonical symbol tests**

```python
def test_group_size_symbolic_and_dense():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    assert model.group_size(g) == sp.Symbol("n_g", positive=True, integer=True)

    dense = model.dense(subject=[0, 0, 1], time=[0, 1, 0])
    assert dense.group_size(0) == 2
    assert dense.group_size(1) == 1


def test_response_is_group_vector_and_entry_is_scalar():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    assert model.response(g).shape == (model.group_size(g), 1)
    assert model.response(g, 0).shape == (1, 1)
    assert model.response(g, 0) == model.response(g)[0, 0]


def test_design_and_covariance_blocks():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    p = len(model.fixed.coefficients())
    q = model.random[0].dim

    assert model.fixed_design(g).shape == (model.group_size(g), p)
    assert model.random_design(g).shape == (model.group_size(g), q)
    assert model.residual_covariance(g).shape == (
        model.group_size(g),
        model.group_size(g),
    )
    assert model.random_effect(g).shape == (q, 1)


def test_response_matrix_and_occasion_column():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    t = sp.Symbol("t", integer=True)

    assert model.response_matrix().shape == (model.group_count(), model.occasion_count())
    assert model.response(g).shape == (model.group_size(g), 1)
    assert model.time_response(t).shape == (model.group_count(), 1)
```

- [ ] **Step 2: Implement the canonical layer**

`canonical.py`:

```python
"""Canonical group/occasion indexing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp


def group_size_symbol(group: Any) -> sp.Symbol:
    """Return the symbolic size of `group`."""
    return sp.Symbol(f"n_{group}", positive=True, integer=True)


def group_response(group: Any, size: Any) -> sp.MatrixSymbol:
    """Return the response vector Y_group."""
    return sp.MatrixSymbol(f"Y_{group}", size, 1)


def group_mean(group: Any, size: Any, p: int) -> sp.MatrixSymbol:
    """Return the fixed design matrix X_group."""
    return sp.MatrixSymbol(f"X_{group}", size, p)


def group_random_design(group: Any, size: Any, q: int) -> sp.MatrixSymbol:
    """Return the random design matrix Z_group."""
    return sp.MatrixSymbol(f"Z_{group}", size, q)


def group_residual(group: Any, size: Any) -> sp.MatrixSymbol:
    """Return the residual covariance matrix R_group."""
    return sp.MatrixSymbol(f"R_{group}", size, size)


def group_coefficients(group: Any, q: int) -> sp.MatrixSymbol:
    """Return the random-effect vector b_group."""
    return sp.MatrixSymbol(f"b_{group}", q, 1)
```

Use the `Group`/`Covariate` definitions from Task 0. `LMM` stores:

```python
self.group_index = self.primary_group_definition.index
self.occasion_index = self.occasion_definition.index
```

where `primary_group_definition` is the primary `Group`/`Nested`/`Crossed` and
`occasion_definition` is the canonical `Covariate`. No standalone `IndexSpec`
exists.

In `LMM`:

```python
@property
def primary_group(self) -> str | None:
    """Return the primary grouping factor for canonical group indexing."""
    if self.random:
        return self.random[0].group
    if hasattr(self.residual, "group"):
        return self.residual.group
    return self.groups[0].name if self.groups else None

def group_size(self, group: Any) -> Any:
    """Return the number of observations in `group`."""
    if self._dense:
        return len(self._group_rows(group))
    from .canonical import group_size_symbol
    return group_size_symbol(group)

def response(self, group: Any, occasion: Any = None) -> Any:
    """Return Y_group (vector) or Y_group,occasion (scalar)."""
    from .canonical import group_response
    vector = group_response(group, self.group_size(group))
    return vector if occasion is None else vector[occasion, 0]

def group_count(self) -> Any:
    """Return the number of groups (symbolic or dense)."""
    if self._dense:
        return len(set(self._dense_group_codes(self._group_spec(self.primary_group))))
    return sp.Symbol("m", positive=True, integer=True)

def occasion_count(self) -> Any:
    """Return the number of occasions for a balanced panel."""
    if self._dense:
        counts = {
            len(self._group_rows(group))
            for group in set(self._dense_group_codes(self._group_spec(self.primary_group)))
        }
        if len(counts) != 1:
            raise NotDefined("occasion vectors require a balanced panel")
        return counts.pop()
    return sp.Symbol("T", positive=True, integer=True)

def response_matrix(self) -> Any:
    """Return the balanced-panel response matrix Y (groups x occasions)."""
    return sp.MatrixSymbol("Y", self.group_count(), self.occasion_count())

def time_response(self, occasion: Any) -> Any:
    """Return the occasion column vector Y[:, occasion]."""
    return self.response_matrix()[:, occasion]

def fixed_design(self, group: Any) -> Any:
    """Return X_group."""
    from .canonical import group_mean
    if self._dense:
        rows = self._group_rows(group)
        return self.matrix_x()[rows, :]
    return group_mean(group, self.group_size(group), len(self.fixed.coefficients()))

def random_design(self, group: Any) -> Any:
    """Return Z_group."""
    from .canonical import group_random_design
    if self._dense:
        rows = self._group_rows(group)
        return self.matrix_z()[rows, :]
    return group_random_design(
        group, self.group_size(group), sum(term.dim for term in self.random)
    )

def random_effect(self, group: Any) -> Any:
    """Return b_group."""
    from .canonical import group_coefficients
    q = sum(term.dim for term in self.random)
    return group_coefficients(group, q)

def residual_covariance(self, group: Any) -> Any:
    """Return R_group."""
    from .canonical import group_residual
    if self._dense:
        rows = self._group_rows(group)
        return self.matrix_r()[rows, :][:, rows]
    return group_residual(group, self.group_size(group))
```

Add `_group_rows(group)`:

```python
def _group_rows(self, group: Any) -> list[int]:
    """Return dense observation rows belonging to `group`."""
    factor = self.primary_group
    if factor is None:
        raise ModelError("canonical indexing requires a grouping factor")
    if not self._dense:
        raise ModelError("group rows are available only in dense mode")
    return [
        k for k in range(self.n)
        if self.group(k, factor) == group
    ]
```

- [ ] **Step 3: Run the canonical symbol tests and commit**

Run: `uv run pytest tests/unit/test_canonical.py -q`
Expected: PASS.

Run full gate and commit:
`git commit -m "feat: add canonical group index and response symbols"`

---

### Task 2: Group moments and entry queries

**Files:**
- Modify: `src/symlmm/model.py`
- Test: `tests/unit/test_canonical.py`
- Test: `tests/integration/test_canonical_dense.py` (new)

**Interfaces:**
- Produces: `group_mean(i)`, `group_covariance(i)`, `group_cross_covariance(i, j)`, `entry_mean(i, t)`, `entry_covariance(i, t, s)`, `entry_cross_covariance(i, t, j, s)`, `time_mean(t)`, `time_covariance(t, s)`. Occasion-column moments require a concrete group count or dense mode in v1.

- [ ] **Step 1: Write the failing block/entry tests**

```python
def test_group_mean_and_covariance_blocks():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    beta = sp.Matrix(model.fixed.coefficients())
    X = model.fixed_design(g)
    Z = model.random_design(g)
    G = model._primary_random_term().cov.dense()

    assert model.group_mean(g) == X * beta
    assert model.group_covariance(g) == Z * G * Z.T + model.residual_covariance(g)


def test_cross_group_covariance_is_zero():
    model = _random_intercept_model()
    g, h = sp.symbols("g h", integer=True)
    assert model.group_cross_covariance(g, h) == sp.zeros(
        model.group_size(g), model.group_size(h)
    )


def test_entry_moments_are_group_components():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    mean = model.group_mean(g)
    cov = model.group_covariance(g)

    assert model.entry_mean(g, 0) == mean[0, 0]
    assert model.entry_covariance(g, 0, 1) == cov[0, 1]
    assert model.entry_cross_covariance(g, 0, g, 1) == cov[0, 1]


def test_occasion_column_moments_dense():
    model = _random_intercept_model().dense(
        subject=[0, 0, 1, 1], time=[0, 1, 0, 1]
    )
    t, s = 0, 1

    assert model.time_mean(t).shape == (2, 1)
    assert model.time_covariance(t, s).shape == (2, 2)
    assert model.time_covariance(t, s)[0, 0] == model.entry_covariance(0, t, s)
    assert model.time_covariance(t, s)[0, 1] == 0
```

- [ ] **Step 2: Implement block and entry moments**

```python
def group_mean(self, group: Any) -> Any:
    """Return E[Y_group]."""
    return self.fixed_design(group) * sp.Matrix(self.fixed.coefficients())

def group_covariance(self, group: Any) -> Any:
    """Return Cov(Y_group) for the primary grouping factor."""
    Z = self.random_design(group)
    G = self._primary_random_term().cov.dense()
    R = self.residual_covariance(group)
    return Z * G * Z.T + R

def group_cross_covariance(self, group: Any, other: Any) -> Any:
    """Return Cov(Y_group, Y_other); zero for a single grouping factor."""
    groups = {term.group for term in self.random}
    if len(groups) > 1:
        raise NotDefined(
            "canonical group blocks require a single grouping factor"
        )
    if group == other:
        return self.group_covariance(group)
    return sp.zeros(self.group_size(group), self.group_size(other))

def entry_mean(self, group: Any, occasion: Any) -> Any:
    """Return E[Y_group,occasion]."""
    return self.group_mean(group)[occasion, 0]

def entry_covariance(self, group: Any, t1: Any, t2: Any) -> Any:
    """Return Cov(Y_group,t1, Y_group,t2)."""
    return self.group_covariance(group)[t1, t2]

def entry_cross_covariance(
    self, group: Any, t1: Any, other: Any, t2: Any
) -> Any:
    """Return Cov(Y_group,t1, Y_other,t2)."""
    return self.group_cross_covariance(group, other)[t1, t2]
```

Add occasion-column moments. For v1 these require a concrete group count or
dense mode; a symbolic group count returns `NotDefined` because the group index
set cannot be enumerated:

```python
def time_mean(self, occasion: Any) -> Any:
    """Return E[Y_{.occasion}] for a concrete balanced panel."""
    count = self.group_count()
    if not isinstance(count, int):
        raise NotDefined("occasion-column moments require a concrete group count")
    return sp.Matrix([self.entry_mean(i, occasion) for i in range(count)])

def time_covariance(self, t1: Any, t2: Any) -> Any:
    """Return Cov(Y_{.t1}, Y_{.t2}) as a diagonal group matrix."""
    count = self.group_count()
    if not isinstance(count, int):
        raise NotDefined("occasion-column moments require a concrete group count")
    return sp.diag(*(self.entry_covariance(i, t1, t2) for i in range(count)))
```

`group_cross_covariance` compares group labels and rejects multiple grouping
factors because crossed effects do not form diagonal group blocks. Import
`NotDefined` from `symlmm.errors` in `model.py`.

- [ ] **Step 3: Dense tests and commit**

Dense test: build a balanced model with 2 groups, assert group means/covariances
equal the corresponding submatrices of `matrix_mean`/`matrix_covariance`, and
cross-group blocks are zero.

Run full gate and commit:
`git commit -m "feat: add group-level canonical moments"`

---

### Task 3: Rebuild blackboard symbols on canonical indexing

**Files:**
- Modify: `src/symlmm/blackboard.py`
- Test: `tests/unit/test_blackboard.py`
- Test: `tests/unit/test_canonical.py`

**Interfaces:**
- Produces: `syms.Y[i]` vector, `syms.Y[i, t]` scalar, `syms.b[i]` vector, `syms.X(i)`, `syms.Z(i)`, `syms.R(i)`, `syms.G`; `E`/`Var`/`Cov` for canonical group responses and entries; total variance for `Y_i`.

- [ ] **Step 1: Write the failing blackboard tests**

```python
def test_blackboard_uses_group_response_vectors():
    model = _random_intercept_model()
    syms = model.symbols()
    g = sp.Symbol("g", integer=True)

    assert syms.Y[g].shape == (model.group_size(g), 1)
    assert syms.Y[g, 0] == syms.Y[g][0, 0]
    assert syms.E(syms.Y[g]) == model.group_mean(g)
    assert syms.Var(syms.Y[g]) == model.group_covariance(g)
    assert syms.Cov(syms.Y[g], syms.Y[g]) == model.group_covariance(g)


def test_blackboard_entry_moments():
    model = _random_intercept_model()
    syms = model.symbols()
    g = sp.Symbol("g", integer=True)

    assert syms.E(syms.Y[g, 0]) == model.entry_mean(g, 0)
    assert syms.Var(syms.Y[g, 0]) == model.entry_covariance(g, 0, 0)
    assert syms.Cov(syms.Y[g, 0], syms.Y[g, 1]) == model.entry_covariance(g, 0, 1)


def test_total_variance_for_group_vector():
    model = _random_intercept_model()
    syms = model.symbols()
    g = sp.Symbol("g", integer=True)

    assert sp.simplify(
        syms.total_variance(syms.Y[g], given=syms.b[g])
        - model.group_covariance(g)
    ) == sp.zeros(model.group_size(g), model.group_size(g))


def test_blackboard_occasion_column():
    model = _random_intercept_model()
    syms = model.symbols()
    t, s = sp.symbols("t s", integer=True)

    assert syms.Y[:, t].shape == (model.group_count(), 1)
    assert syms.Y[syms.t].shape == (model.group_count(), 1)
    assert syms.E(syms.Y[:, t]).shape == (model.group_count(), 1)
    assert syms.Var(syms.Y[:, t]).shape == (model.group_count(), model.group_count())
```

- [ ] **Step 2: Replace the blackboard scalar row layer**

`ModelSymbols` fields:

```python
Y: ResponseSymbols       # Y[i] row, Y[:, t] column, Y[i, t] scalar
b: GroupEffectSymbols    # b[i] vector
i: sp.Symbol
t: sp.Symbol
X: Callable[[Any], Any]
Z: Callable[[Any], Any]
R: Callable[[Any], Any]
G: Any
beta: sp.Matrix
```

`LMM.symbols()` sets `i=model.group_index` and `t=model.occasion_index`; no
index name is hardcoded in the facade.

Implement `ResponseSymbols`:

```python
class ResponseSymbols:
    def __init__(self, model: LMM) -> None:
        self.model = model

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            row, column = key
            if row is slice(None):
                return self.model.time_response(column)
            if column is slice(None):
                return self.model.response(row)
            return self.model.response(row, column)
        if key == self.model.occasion_index:
            return self.model.time_response(key)
        if key == self.model.group_index:
            return self.model.response(key)
        raise ModelError(
            "ambiguous response index; use Y[group_index], Y[occasion_index], "
            "Y[i, :], or Y[:, t]"
        )
```

`GroupEffectSymbols.__getitem__` returns `model.random_effect(group)`.

Moment engine changes:

- `Y_i` row vector: delegate `E`/`Var`/`Cov` to `group_mean`/`group_covariance`/`group_cross_covariance`.
- `Y_{.t}` occasion column: delegate `E`/`Var`/`Cov` to `time_mean`/`time_covariance`.
- `Y_{it}` scalar: delegate to `entry_mean`/`entry_covariance`/`entry_cross_covariance`.
- `b_i` vector: `E=0`, `Var=G`, `Cov(b_i,b_j)=0` for `i != j`.
- `given=syms.b[g]`: conditional group response uses `X_i beta + Z_i b_i`, conditional covariance `R_i`.
- `total_variance(Y_i)` returns `Var(E(Y_i|b_i)) + E(Var(Y_i|b_i))`.
- Nonlinear/vector-combination expressions raise `UnsupportedMoment` in the first version.

- [ ] **Step 3: Run blackboard tests and commit**

Run: `uv run pytest tests/unit/test_blackboard.py tests/unit/test_canonical.py -q`

Run full gate and commit:
`git commit -m "feat: rebuild blackboard on canonical group indexing"`

---

### Task 4: Docs, migration, issue closure

**Files:**
- Modify: `PLAN.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/pedagogy/blackboard.md`
- Modify: `pyproject.toml`
- Controller: `src/symlmm/__init__.py`
- Issue: #1

- [ ] **Step 1: Update docs**

Document:

```text
i = group/cluster/subject
t = occasion
Y_i = response vector
Y_{it} = scalar response
b_i = random-effect vector
V_i = Z_i G Z_i' + R_i
Cov(Y_i, Y_j) = 0 for i != j
```

Mark the old row-index notation as stacked-engine internals, not canonical model
notation.

- [ ] **Step 2: Version and exports**

- Bump to 0.5.0.
- Export canonical helper classes and `ModelSymbols`.
- Remove row-index wording from blackboard examples.
- Run `uv lock` if needed.

- [ ] **Step 3: Verify, close issue #1, commit**

Run full gate. Add a comment to issue #1 with the canonical API and close it as
completed.

```bash
git commit -m "docs: document canonical group indexing"
```

---

## Final Gate

- [ ] `uv run pytest -q`
- [ ] `uv run ruff check`
- [ ] `uv run ruff format --check`
- [ ] `uv run mypy src/symlmm`
- [ ] Whole-branch review.
- [ ] Merge, push, close issue #1.
