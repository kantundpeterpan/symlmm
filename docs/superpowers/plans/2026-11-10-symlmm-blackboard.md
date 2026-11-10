# Blackboard-Style Symbolic Moments — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pedagogical `model.symbols()` facade with `E`, `Var`, `Cov`, conditional moments, and law-of-total-variance rewriting over the existing LMM moment engine.

**Architecture:** `ModelSymbols` exposes indexed response, residual, random-effect, and fixed-effect symbols. A small linear-moment engine classifies model atoms (`Y[i]`, `eps[i]`, `b[group][level, a]`), delegates atom moments to the existing model/covariance API, and uses bilinearity to evaluate linear expressions. `given=syms.b` conditions on all random effects. Nonlinear expressions fail loudly.

**Tech Stack:** Python 3.14, SymPy 1.14, pytest 9, ruff 0.17, mypy 2.4, uv.

## Global Constraints

- Every public query method requires `form`; missing values raise `MissingFormError`.
- Exceptions raised by library code come from `symlmm.errors`.
- Internal indices are 0-based; rendered documentation is 1-based.
- Do not call `as_explicit()` outside dense mode.
- `python_version = "3.14"`, ruff `target-version = "py314"`, line length 100.
- `src/symlmm/__init__.py` is controller-owned; implementers must not edit it.
- Public functions/classes keep docstrings.
- The facade supports **linear expressions in model atoms** only.
- Full gate: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`.

## Concurrency Plan

- **Lane A — code:** Task 1 → Task 2 → Task 3, sequential. All three touch
  `blackboard.py`, `model.py`, `moments.py`, and the same test file.
- **Lane B — docs:** Task 4 starts after Task 2 is reviewed, in a separate
  docs worktree, parallel with Task 3. It edits docs/examples only.
- Peak: one code implementer + one docs agent. Reviews are read-only.
- Controller updates `src/symlmm/__init__.py` exports after code review.

---

### Task 1: ModelSymbols and atom representation

**Files:**
- Create: `src/symlmm/blackboard.py`
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/moments.py`
- Test: `tests/unit/test_blackboard.py` (new)
- Test: `tests/unit/test_moments.py`

**Interfaces:**
- Consumes: `LMM`, `RandomEffects`, `IndexedModel`, `Fixed`, `sp.IndexedBase`.
- Produces: `LMM.symbols() -> ModelSymbols`; `ModelSymbols.Y`, `.eps`, `.b`, `.beta`, `.X(i)`, `.Z(i)`, `.G`, `.R`, `.i`, `.j`, `.n`; `RandomEffectCollection`.

- [ ] **Step 1: Write the failing symbol tests**

Create `tests/unit/test_blackboard.py`:

```python
"""Blackboard-style symbolic moment tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar, Unstructured
from symlmm.errors import UnsupportedMoment
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects, RandomSlope


def _intercept_model():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                cov=Scalar(tau2),
            )
        ],
        residual=IID(var=sigma2),
    )


def test_symbols_are_model_indexed():
    model = _intercept_model()
    syms = model.symbols()
    i, j = model.indices()

    assert syms.Y[i].base == sp.IndexedBase("Y", shape=(model.n,))
    assert syms.eps[i].base == sp.IndexedBase("eps", shape=(model.n,))
    assert syms.i == i
    assert syms.j == j
    assert syms.beta.shape == (1, 1)


def test_random_effect_collection_matches_engine_symbol():
    model = _intercept_model()
    syms = model.symbols()
    b_atom = syms.b["subject"][0, 0]

    assert b_atom.base.name == "b_subject"
    conditional = model.conditional_mean(model.indices()[0], form="sum")
    assert any(atom.base.name == "b_subject" for atom in conditional.atoms(sp.Indexed))


def test_design_accessors():
    model = _intercept_model()
    syms = model.symbols()
    i, _ = model.indices()

    assert syms.X(i) == model.design(i).T
    assert syms.Z(i).shape[0] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_blackboard.py -q`
Expected: FAIL with missing `LMM.symbols`, `ModelSymbols`, and `b_*` symbol.

- [ ] **Step 3: Implement `ModelSymbols` and atom collection**

Create `src/symlmm/blackboard.py` with the symbol layer:

```python
"""Blackboard-style symbolic moments for linear mixed models."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sympy as sp

from .errors import ModelError

if TYPE_CHECKING:
    from .model import LMM


class RandomEffectCollection(Mapping[str, sp.IndexedBase]):
    """Random-effect index bases keyed by grouping factor."""

    def __init__(self, model: LMM) -> None:
        self.model = model
        self._bases = {
            term.group: sp.IndexedBase(
                f"b_{term.group.replace(':', '_')}",
                shape=(model.levels(term.group), term.dim),
            )
            for term in model.random
        }

    def __getitem__(self, group: str) -> sp.IndexedBase:
        return self._bases[group]

    def __iter__(self) -> Iterator[str]:
        return iter(self._bases)

    def __len__(self) -> int:
        return len(self._bases)

    def group_of(self, atom: sp.Indexed) -> str:
        """Return the grouping factor for a random-effect atom."""
        for group, base in self._bases.items():
            if atom.base == base:
                return group
        raise ModelError(f"not a random-effect atom: {atom!r}")


@dataclass(frozen=True)
class ModelSymbols:
    """Model symbols and moment operators for blackboard-style work."""

    model: LMM
    Y: sp.IndexedBase
    eps: sp.IndexedBase
    b: RandomEffectCollection
    beta: sp.Matrix
    i: Any
    j: Any
    n: Any
    G: Any
    R: Any

    def X(self, i: Any) -> sp.Matrix:
        """Return the fixed-effect design row for observation `i`."""
        return self.model.design(i).T

    def Z(self, i: Any) -> sp.Matrix:
        """Return the random-effect design row for observation `i`."""
        columns: list[Any] = []
        for term in self.model.random:
            columns.extend(list(term.z(self.model, i)))
        return sp.Matrix(columns).T
```

Add `LMM.symbols()`:

```python
def symbols(self) -> Any:
    """Return blackboard-style model symbols and moment operators."""
    from .blackboard import ModelSymbols, RandomEffectCollection

    i, j = self.indices()
    return ModelSymbols(
        model=self,
        Y=sp.IndexedBase("Y", shape=(self.n,)),
        eps=sp.IndexedBase("eps", shape=(self.n,)),
        b=RandomEffectCollection(self),
        beta=sp.Matrix(self.fixed.coefficients()),
        i=i,
        j=j,
        n=self.n,
        G=self.matrix_g(),
        R=self.matrix_r(),
    )
```

Change the conditional-mean base in `src/symlmm/moments.py` from
`f"b_{term.group}_{term_index}"` to `f"b_{term.group}"`; duplicate group blocks
are rejected, so the term index is unnecessary.

Update `tests/unit/test_moments.py` expectations from `b_subject_0` to
`b_subject`.

- [ ] **Step 4: Run the symbols tests**

Run: `uv run pytest tests/unit/test_blackboard.py tests/unit/test_moments.py -q`
Expected: PASS after updating conditional-mean symbol assertions.

- [ ] **Step 5: Commit Task 1**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add src/symlmm/blackboard.py src/symlmm/model.py src/symlmm/moments.py tests/unit/test_blackboard.py tests/unit/test_moments.py
git commit -m "feat: add blackboard model symbols"
```

---

### Task 2: Linear E, Var, Cov engine

**Files:**
- Modify: `src/symlmm/blackboard.py`
- Modify: `src/symlmm/errors.py`
- Test: `tests/unit/test_blackboard.py`

**Interfaces:**
- Consumes: `ModelSymbols`, `LMM.covariance_entry`, `LMM.mean`, `LMM.variance`, `RandomEffects.cov`, `RandomEffects.z`.
- Produces: `ModelSymbols.E(expr, given=None)`, `.Var(expr, given=None)`, `.Cov(expr1, expr2, given=None)`; `UnsupportedMoment`.

- [ ] **Step 1: Write the failing moment tests**

Append to `tests/unit/test_blackboard.py`:

```python
def test_expectation_of_response_and_linear_combination():
    model = _intercept_model()
    syms = model.symbols()
    i, j = model.indices()
    expr = 2 * syms.Y[i] + 3 * syms.Y[j]

    expected = 2 * model.mean(i, form="sum") + 3 * model.mean(j, form="sum")
    assert sp.simplify(syms.E(expr) - expected) == 0


def test_variance_and_covariance_of_responses():
    model = _intercept_model()
    syms = model.symbols()
    i, j = model.indices()

    assert sp.simplify(syms.Var(syms.Y[i]) - model.variance(i, form="sum")) == 0
    assert sp.simplify(syms.Cov(syms.Y[i], syms.Y[j]) - model.covariance(i, j, form="sum")) == 0


def test_random_effect_covariance():
    b0 = sp.Symbol("b0")
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Unstructured.from_sd_corr([s0, s1], rho),
            )
        ],
        residual=IID(var=sigma2),
    )
    syms = model.symbols()
    b = syms.b["subject"]

    assert sp.simplify(syms.Var(b[0, 0]) - s0**2) == 0
    assert sp.simplify(syms.Cov(b[0, 0], b[0, 1]) - rho * s0 * s1) == 0
    assert syms.Cov(b[0, 0], b[1, 0]) == 0


def test_nonlinear_expression_rejected():
    model = _intercept_model()
    syms = model.symbols()
    i, _ = model.indices()

    with pytest.raises(UnsupportedMoment):
        syms.Var(syms.Y[i] ** 2)
```


- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_blackboard.py -q`
Expected: FAIL with missing `E`, `Var`, `Cov`, and `UnsupportedMoment`.

- [ ] **Step 3: Add `UnsupportedMoment`**

In `src/symlmm/errors.py`:

```python
class UnsupportedMoment(ModelError, ValueError):
    """Raised when the blackboard moment engine cannot evaluate an expression."""
```

Update `blackboard.py` to import it:

```python
from .errors import ModelError, UnsupportedMoment
```

- [ ] **Step 4: Implement the linear moment engine**

In `blackboard.py`, add helpers:

```python
def _atom_kind(symbols: ModelSymbols, atom: sp.Indexed) -> str | None:
    base = atom.base.name
    if base == symbols.Y.name:
        return "Y"
    if base == symbols.eps.name:
        return "eps"
    if base.startswith("b_"):
        return "b"
    return None


def _linear_terms(symbols: ModelSymbols, expr: Any) -> list[tuple[Any, sp.Indexed | None]]:
    expanded = sp.expand(expr)
    terms = sp.Add.make_args(expanded)
    result: list[tuple[Any, sp.Indexed | None]] = []
    for term in terms:
        atoms = [
            atom
            for atom in term.atoms(sp.Indexed)
            if _atom_kind(symbols, atom) is not None
        ]
        if not atoms:
            result.append((term, None))
            continue
        if len(atoms) != 1:
            raise UnsupportedMoment("moment engine supports linear expressions only")
        atom = atoms[0]
        coefficient = sp.simplify(term / atom)
        if coefficient.has(atom):
            raise UnsupportedMoment("moment engine supports linear expressions only")
        result.append((coefficient, atom))
    return result
```

Add `ModelSymbols` methods:

```python
def E(self, expr: Any, *, given: Any = None) -> Any:
    """Return the expectation of a linear expression."""
    return expectation(self, expr, given=given)

def Var(self, expr: Any, *, given: Any = None) -> Any:
    """Return the variance of a linear expression."""
    return covariance(self, expr, expr, given=given)

def Cov(self, expr1: Any, expr2: Any, *, given: Any = None) -> Any:
    """Return the covariance of two linear expressions."""
    return covariance(self, expr1, expr2, given=given)
```

Implement `expectation` and `covariance` (given handling is Task 3; `given=None`
only for Task 2):

```python
def expectation(symbols: ModelSymbols, expr: Any, *, given: Any = None) -> Any:
    total = sp.Integer(0)
    for coefficient, atom in _linear_terms(symbols, expr):
        if atom is None:
            total += coefficient
            continue
        kind = _atom_kind(symbols, atom)
        if kind == "Y":
            total += coefficient * symbols.model.mean(atom.indices[0], form="sum")
        elif kind == "eps":
            continue
        elif kind == "b":
            total += coefficient * sp.Integer(0) if given is None else coefficient * atom
    return sp.expand(total)


def covariance(symbols: ModelSymbols, expr1: Any, expr2: Any, *, given: Any = None) -> Any:
    total = sp.Integer(0)
    for c1, a1 in _linear_terms(symbols, expr1):
        if a1 is None:
            continue
        for c2, a2 in _linear_terms(symbols, expr2):
            if a2 is None:
                continue
            total += c1 * c2 * _atom_covariance(symbols, a1, a2, given)
    return sp.expand(total)
```

`_atom_covariance` handles Y/Y, eps/eps, b/b, Y/eps, Y/b, b/eps; use
`model.covariance_entry`, `model.residual.entry`, and the `RandomEffects` block:

```python
def _term_for_group(model: Any, group: str) -> Any:
    return next(term for term in model.random if term.group == group)

def _atom_covariance(symbols: ModelSymbols, a1: Any, a2: Any, given: Any) -> Any:
    k1 = _atom_kind(symbols, a1)
    k2 = _atom_kind(symbols, a2)
    if k1 == "Y" and k2 == "Y":
        return symbols.model.covariance_entry(a1.indices[0], a2.indices[0])
    if k1 == "eps" and k2 == "eps":
        return symbols.model.residual.entry(a1.indices[0], a2.indices[0], symbols.model)
    if k1 == "Y" and k2 == "eps":
        return symbols.model.residual.entry(a1.indices[0], a2.indices[0], symbols.model)
    if k1 == "eps" and k2 == "Y":
        return symbols.model.residual.entry(a2.indices[0], a1.indices[0], symbols.model)
    if k1 == "b" and k2 == "b":
        group1 = symbols.b.group_of(a1)
        group2 = symbols.b.group_of(a2)
        if group1 != group2 or a1.indices[0] != a2.indices[0]:
            return sp.Integer(0)
        term = _term_for_group(symbols.model, group1)
        return term.cov.dense()[a1.indices[1], a2.indices[1]]
    if k1 == "Y" and k2 == "b":
        return _cov_y_b(symbols, a1, a2)
    if k1 == "b" and k2 == "Y":
        return _cov_y_b(symbols, a2, a1)
    return sp.Integer(0)
```

`_cov_y_b`:

```python
def _cov_y_b(symbols: ModelSymbols, y_atom: sp.Indexed, b_atom: sp.Indexed) -> Any:
    group = symbols.b.group_of(b_atom)
    term = _term_for_group(symbols.model, group)
    i = y_atom.indices[0]
    level, a = b_atom.indices
    if symbols.model.group(i, group) != level:
        return sp.Integer(0)
    return (term.cov.dense() * term.z(symbols.model, i))[a]
```

- [ ] **Step 5: Run the moment tests and commit Task 2**

Run: `uv run pytest tests/unit/test_blackboard.py -q`
Expected: PASS.

Run the full gate:
`uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add src/symlmm/blackboard.py src/symlmm/errors.py tests/unit/test_blackboard.py
git commit -m "feat: add linear blackboard moment engine"
```

---

### Task 3: Conditional moments and total variance/covariance

**Files:**
- Modify: `src/symlmm/blackboard.py`
- Test: `tests/unit/test_blackboard.py`

**Interfaces:**
- Consumes: `LMM.conditional_mean`, `LMM.conditional_covariance`, `RandomEffectCollection`.
- Produces: `given=syms.b` support; `ModelSymbols.total_variance`, `.total_covariance`.

- [ ] **Step 1: Write the failing decomposition tests**

Append to `tests/unit/test_blackboard.py`:

```python
def test_conditional_mean_uses_all_random_effects():
    model = _intercept_model()
    syms = model.symbols()
    i, _ = model.indices()

    conditional = syms.E(syms.Y[i], given=syms.b)
    assert sp.simplify(conditional - model.conditional_mean(i, form="sum")) == 0


def test_total_variance_matches_model_variance():
    model = _intercept_model()
    syms = model.symbols()
    i, _ = model.indices()

    assert sp.simplify(
        syms.total_variance(syms.Y[i], given=syms.b)
        - model.variance(i, form="sum")
    ) == 0


def test_total_variance_components():
    model = _intercept_model()
    syms = model.symbols()
    i, _ = model.indices()

    between = syms.Var(syms.E(syms.Y[i], given=syms.b))
    within = syms.E(syms.Var(syms.Y[i], given=syms.b))
    assert between == syms.model.random[0].cov.dense()[0, 0]
    assert within == model.residual.entry(i, i, model)


def test_total_covariance_matches_model_covariance():
    model = _intercept_model()
    syms = model.symbols()
    i, j = model.indices()

    assert sp.simplify(
        syms.total_covariance(syms.Y[i], syms.Y[j], given=syms.b)
        - model.covariance(i, j, form="sum")
    ) == 0
```

- [ ] **Step 2: Implement conditional handling**

Extend `expectation`:

```python
if kind == "Y":
    if given is symbols.b:
        value = symbols.model.conditional_mean(atom.indices[0], form="sum")
    else:
        value = symbols.model.mean(atom.indices[0], form="sum")
    total += coefficient * value
elif kind == "eps":
    continue
elif kind == "b":
    total += coefficient * atom if given is symbols.b else sp.Integer(0)
```

Extend `_atom_covariance`:

```python
if given is symbols.b:
    if k1 == "Y" and k2 == "Y":
        return symbols.model.conditional_covariance(
            a1.indices[0], a2.indices[0], form="sum"
        )
    if k1 == "b" or k2 == "b":
        return sp.Integer(0)
```

Residual/residual and residual/response rules remain unchanged.

Add:

```python
def total_variance(self, expr: Any, *, given: Any = None) -> Any:
    """Return Var(E(expr|given)) + E(Var(expr|given))."""
    given = self.b if given is None else given
    between = self.Var(self.E(expr, given=given))
    within = self.E(self.Var(expr, given=given))
    return sp.expand(between + within)

def total_covariance(self, expr1: Any, expr2: Any, *, given: Any = None) -> Any:
    """Return Cov(E1|given, E2|given) + E(Cov(expr1, expr2|given))."""
    given = self.b if given is None else given
    between = self.Cov(self.E(expr1, given=given), self.E(expr2, given=given))
    within = self.E(self.Cov(expr1, expr2, given=given))
    return sp.expand(between + within)
```

- [ ] **Step 3: Run the decomposition tests and commit Task 3**

Run: `uv run pytest tests/unit/test_blackboard.py -q`
Expected: PASS.

Run the full gate:
`uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add src/symlmm/blackboard.py tests/unit/test_blackboard.py
git commit -m "feat: add conditional blackboard moments"
```

---

### Task 4: Docs, examples, exports

**Files:**
- Create: `docs/pedagogy/blackboard.md`
- Modify: `PLAN.md`
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml`
- Controller: `src/symlmm/__init__.py`

**Interfaces:**
- Produces: public `ModelSymbols` export; `UnsupportedMoment` export; a worked
  blackboard example; version 0.4.0.

- [ ] **Step 1: Write the pedagogy example**

Create `docs/pedagogy/blackboard.md` with a complete random-intercept example:

```python
model = LMM(...)
syms = model.symbols()

E_Y = syms.E(syms.Y[i])
V_Y = syms.Var(syms.Y[i])
V_Y_conditional = syms.Var(syms.Y[i], given=syms.b)
V_E_Y_given_b = syms.Var(syms.E(syms.Y[i], given=syms.b))
E_V_Y_given_b = syms.E(syms.Var(syms.Y[i], given=syms.b))
assert sp.simplify(V_Y - (V_E_Y_given_b + E_V_Y_given_b)) == 0
```

Explain the linear-only restriction and the `sympy.stats` distinction.

- [ ] **Step 2: Update PLAN and changelog**

- Add a blackboard section to `PLAN.md` under the public API/teaching surface.
- Changelog 0.4.0: `model.symbols()`, `E`/`Var`/`Cov`, `given`, total variance/covariance, `UnsupportedMoment`.
- `pyproject.toml`: version 0.4.0; run `uv lock` and include `uv.lock` if changed.

- [ ] **Step 3: Controller export update**

After implementation, update `src/symlmm/__init__.py`:

```python
from .blackboard import ModelSymbols
from .errors import ..., UnsupportedMoment
```

Add `"ModelSymbols"` and `"UnsupportedMoment"` to `__all__`; run `ruff --fix`.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`

```bash
git add docs/pedagogy/blackboard.md PLAN.md CHANGELOG.md pyproject.toml uv.lock src/symlmm/__init__.py
git commit -m "docs: add blackboard moment examples and exports"
```

---

## Final Gate

- [ ] `uv run pytest -q`
- [ ] `uv run ruff check`
- [ ] `uv run ruff format --check`
- [ ] `uv run mypy src/symlmm`
- [ ] Whole-branch review on the most capable model.
- [ ] Merge to `main` with `--no-ff`, push, clean up.
