# symlmm Core Implementation Plan (M0–M2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the symbolic IR, dual-form engine, conditional LMM moments, covariance structures, variograms, BLUP, and dense mode for symlmm v0.1.

**Architecture:** A single model IR holds fixed, random, and residual specifications. Matrix form returns dimensionally abstract `MatrixExpr` objects. Sum form returns structured entries built from `IndexedBase`, `Sum`, and `KroneckerDelta`. Both derive from the same IR; dense mode is the only concrete path.

**Tech Stack:** Python 3.14, SymPy, `uv`, `pytest`, `ruff`, `mypy`, NumPy (development and validation only).

## Global Constraints

- `requires-python = ">=3.14"`; `sympy>=1.14`; `numpy>=2` is a `dev` extra only.
- Every public method requires `form`; `None` raises `MissingFormError`.
- Internal indices are 0-based; rendered docs convert to 1-based.
- No `as_explicit()` outside dense mode.
- Exceptions come only from `symlmm.errors`.
- Covariance structures default to PD-safe constructors; raw matrices are dense-only.
- Source files live under `src/symlmm/`; tests mirror `src/` under `tests/`.
- Branch names: `feat/<ticket>` for features, `fix/<ticket>` for bugs, `chore/<ticket>` otherwise.
- Commit after each task's tests pass. Do not commit failing tests.
- Every public function has a docstring. Every new module is exported through `symlmm/__init__.py` when public.

---

### Task T0.1: Scaffold and tooling

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/symlmm/__init__.py`
- Create: `src/symlmm/py.typed`
- Create: `tests/unit/test_package.py`
- Create: `.gitignore`
- Create: `LICENSE` (BSD-3-Clause)
- Create: `CHANGELOG.md`
- Create: `CITATION.cff`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `symlmm.__version__ == "0.1.0.dev0"`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_package.py
import symlmm


def test_version_is_exposed():
    assert symlmm.__version__ == "0.1.0.dev0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'symlmm'`.

- [ ] **Step 3: Write the package and tooling configuration**

```toml
# pyproject.toml
[project]
name = "symlmm"
version = "0.1.0.dev0"
description = "Symbolic moments for linear mixed, marginal, and GEE models"
readme = "README.md"
requires-python = ">=3.14"
license = { text = "BSD-3-Clause" }
dependencies = ["sympy>=1.14"]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6", "mypy>=1.11", "numpy>=2"]
docs = ["quartodoc>=0.9", "jupyter>=1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py314"

[tool.mypy]
python_version = "3.14"
strict = true
```

```python
# src/symlmm/__init__.py
"""Symbolic moments for linear mixed, marginal, and GEE models."""

__version__ = "0.1.0.dev0"
```

```text
# .python-version
3.14
```

- [ ] **Step 4: Lock and sync dependencies**

Run: `uv lock && uv sync --all-extras`
Expected: `uv.lock` created; environment resolves.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_package.py -v`
Expected: PASS.

- [ ] **Step 6: Run static checks**

Run: `uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version src/symlmm tests/unit/test_package.py .gitignore LICENSE CHANGELOG.md CITATION.cff .github/workflows/ci.yml
git commit -m "chore: scaffold symlmm package"
```

---

### Task T0.2: Error hierarchy and forms

**Files:**
- Create: `src/symlmm/errors.py`
- Create: `src/symlmm/forms.py`
- Test: `tests/unit/test_forms.py`
- Test: `tests/unit/test_errors.py`

**Interfaces:**
- Produces: `SymlmmError`, `FormError`, `MissingFormError`, `InvalidFormError`, `FormArityError`, `UnsupportedForm`, `ModelError`, `NotDecomposable`, `NoConditionalModel`, `NotDefined`, `UnresolvedSymbols`, `ParameterConstraintError`
- Produces: `Form`, `require_form`, `DualForm`, `EntryMap`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_forms.py
import pytest
import sympy as sp

from symlmm.errors import FormArityError, InvalidFormError, MissingFormError
from symlmm.forms import DualForm, EntryMap, Form, require_form


def test_missing_form_raises():
    with pytest.raises(MissingFormError):
        require_form(None)


def test_invalid_form_raises():
    with pytest.raises(InvalidFormError):
        require_form("latex")


def test_valid_forms_return_enum():
    assert require_form("sum") is Form.SUM
    assert require_form("matrix") is Form.MATRIX
    assert require_form("both") is Form.BOTH


def test_entry_map_lookup():
    i, j = sp.symbols("i j", integer=True)
    entry_map = EntryMap(lambda a, b: a + b)
    assert entry_map[i, j] == i + j


def test_dual_form_holds_both_notations():
    dual = DualForm(sum=sp.Symbol("s"), matrix=sp.Symbol("m"))
    assert dual.sum == sp.Symbol("s")
    assert dual.matrix == sp.Symbol("m")
```

```python
# tests/unit/test_errors.py
from symlmm.errors import NotDecomposable, ParameterConstraintError, SymlmmError


def test_not_decomposable_carries_reason_and_suggestion():
    error = NotDecomposable("AR(1) has no exact split", "use InducedCovariance")
    assert error.reason == "AR(1) has no exact split"
    assert error.suggestion == "use InducedCovariance"
    assert isinstance(error, SymlmmError)


def test_parameter_constraint_error_carries_violations():
    error = ParameterConstraintError(["rho < 1"])
    assert error.violations == ("rho < 1",)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_forms.py tests/unit/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the error hierarchy**

```python
# src/symlmm/errors.py
"""Public exception hierarchy for symlmm."""


class SymlmmError(Exception):
    """Base class for all symlmm errors."""


class FormError(SymlmmError):
    """Base class for notation-selection errors."""


class MissingFormError(FormError, TypeError):
    """Raised when a public call omits the required `form` argument."""


class InvalidFormError(FormError, ValueError):
    """Raised when `form` is not one of 'sum', 'matrix', or 'both'."""


class FormArityError(FormError, ValueError):
    """Raised when an index arity is not valid for the requested form."""


class UnsupportedForm(FormError):
    """Raised when a quantity has no representation in the requested form."""


class ModelError(SymlmmError):
    """Base class for model-level errors."""


class NotDecomposable(ModelError):
    """Raised when a marginal model has no exact conditional representation."""

    def __init__(self, reason: str, suggestion: str) -> None:
        super().__init__(f"{reason}; suggestion: {suggestion}")
        self.reason = reason
        self.suggestion = suggestion


class NoConditionalModel(ModelError):
    """Raised when a model has no conditional counterpart."""


class NotDefined(ModelError):
    """Raised when a quantity is undefined for a model class."""


class UnresolvedSymbols(ModelError):
    """Raised when numeric evaluation still has free symbols."""


class ParameterConstraintError(ModelError):
    """Raised when covariance parameter constraints are violated."""

    def __init__(self, violations: list[str]) -> None:
        super().__init__("; ".join(violations))
        self.violations = tuple(violations)
```

- [ ] **Step 4: Implement forms**

```python
# src/symlmm/forms.py
"""Notation selection and dual-form containers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import sympy as sp

from .errors import InvalidFormError, MissingFormError


class Form(str, Enum):
    """Supported notation modes."""

    SUM = "sum"
    MATRIX = "matrix"
    BOTH = "both"


def require_form(form: str | Form | None) -> Form:
    """Validate and normalize a required form argument."""
    if form is None:
        raise MissingFormError("form is required: use 'sum', 'matrix', or 'both'")
    try:
        return Form(form)
    except ValueError as exc:
        raise InvalidFormError(f"invalid form: {form!r}") from exc


@dataclass(frozen=True)
class EntryMap:
    """Lazy map from an index tuple to a symbolic entry."""

    func: Callable[..., sp.Expr]

    def __getitem__(self, key: tuple[sp.Expr, ...]) -> sp.Expr:
        if not isinstance(key, tuple):
            key = (key,)
        return self.func(*key)


@dataclass(frozen=True)
class DualForm:
    """Both notations for one quantity.

    `sum` is an expression for entry queries or an EntryMap for whole-matrix
    queries. `matrix` is a MatrixExpr or a MatrixElement.
    """

    sum: sp.Expr | EntryMap | None
    matrix: sp.MatrixExpr | sp.Expr | None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_forms.py tests/unit/test_errors.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/errors.py src/symlmm/forms.py tests/unit/test_forms.py tests/unit/test_errors.py
git commit -m "feat: add error hierarchy and form contracts"
```

---

### Task T0.3: Index and data model

**Files:**
- Create: `src/symlmm/symbols.py`
- Create: `src/symlmm/specs.py`
- Create: `src/symlmm/model.py`
- Test: `tests/unit/test_indices.py`

**Interfaces:**
- Consumes: `Group`, `Nested`, `Crossed`
- Produces: `IndexedModel(n, groups, time, covariates)`, `.indices()`, `.group(i, factor)`, `.time(i)`, `.covariate(i, name)`, `.design(i)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_indices.py
import sympy as sp

from symlmm.model import IndexedModel
from symlmm.specs import Crossed, Group, Nested


def test_observation_accessors():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), time="time")
    i, _ = model.indices()

    assert model.group(i, "subject") == sp.IndexedBase("subject", shape=(n,))[i]
    assert model.time(i) == sp.IndexedBase("t", shape=(n,))[i]
    assert model.covariate(i, "time") == sp.IndexedBase("time", shape=(n,))[i]
    assert model.design(i) == sp.Matrix([1, sp.IndexedBase("time", shape=(n,))[i]])


def test_nested_label_is_composite():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Nested("visit", within="subject"),))
    i, _ = model.indices()
    label = model.group(i, "visit")
    assert label.func == sp.IndexedBase("subject:visit", shape=(n,))


def test_crossed_label_is_composite():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Crossed("subject", "item"),))
    i, _ = model.indices()
    label = model.group(i, "item")
    assert label.func == sp.IndexedBase("subject:item", shape=(n,))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_indices.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement grouping specs**

```python
# src/symlmm/specs.py
"""Model specifications: grouping factors and designs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    """A single grouping factor."""

    name: str

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class Nested:
    """A factor nested within another factor."""

    child: str
    within: str

    @property
    def name(self) -> str:
        return self.child

    @property
    def label(self) -> str:
        return f"{self.within}:{self.child}"


@dataclass(frozen=True)
class Crossed:
    """A crossed combination of factors."""

    left: str
    right: str

    @property
    def name(self) -> str:
        return self.right

    @property
    def label(self) -> str:
        return f"{self.left}:{self.right}"
```

- [ ] **Step 4: Implement the index model**

```python
# src/symlmm/symbols.py
"""Observation and parameter symbol helpers."""

from __future__ import annotations

import sympy as sp


def observation_index(name: str = "i") -> sp.Symbol:
    """Return a nonnegative integer observation index."""
    return sp.Symbol(name, integer=True, nonnegative=True)


def parameter(name: str, *, positive: bool = False, real: bool = False) -> sp.Symbol:
    """Return a model parameter symbol with explicit assumptions."""
    return sp.Symbol(name, positive=positive, real=real)
```

```python
# src/symlmm/model.py
"""Model base classes and index accessors."""

from __future__ import annotations

import sympy as sp

from .symbols import observation_index


class IndexedModel:
    """Base class for models over a stacked observation vector."""

    def __init__(self, *, n, groups=(), time=None, covariates=()) -> None:
        self.n = n
        self.groups = tuple(groups)
        self.time_name = time
        self.covariates = tuple(covariates)
        self._i = observation_index("i")
        self._j = observation_index("j")

    def indices(self):
        """Return the two observation indices used by entry queries."""
        return self._i, self._j

    def _group_spec(self, factor: str):
        for spec in self.groups:
            if spec.name == factor:
                return spec
        raise KeyError(f"unknown grouping factor: {factor!r}")

    def group(self, i, factor: str):
        """Return the group label for observation `i` and factor `factor`."""
        spec = self._group_spec(factor)
        return sp.IndexedBase(spec.label, shape=(self.n,))[i]

    def time(self, i, coordinate: str = "time"):
        """Return the time or coordinate value for observation `i`."""
        name = self.time_name if coordinate == "time" else coordinate
        if name is None:
            raise KeyError("model has no time coordinate")
        return sp.IndexedBase(name, shape=(self.n,))[i]

    def covariate(self, i, name: str):
        """Return a covariate value for observation `i`."""
        return sp.IndexedBase(name, shape=(self.n,))[i]

    def design(self, i):
        """Return the fixed-effect design row `[1, covariates...]`."""
        return sp.Matrix([1, *[self.covariate(i, name) for name in self.covariates]])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_indices.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/symbols.py src/symlmm/specs.py src/symlmm/model.py tests/unit/test_indices.py
git commit -m "feat: add observation index model and grouping factors"
```

---

### Task T0.4: Constraints and positive-definiteness

**Files:**
- Create: `src/symlmm/constraints.py`
- Test: `tests/unit/test_constraints.py`

**Interfaces:**
- Produces: `Relation`, `Constraint`, `Constraint.holds(subs)`, `is_positive_definite(matrix, positive=())`, `check_constraints(constraints, subs)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_constraints.py
import sympy as sp

from symlmm.constraints import Constraint, Relation, is_positive_definite


def test_pd_tri_state():
    a, b, c = sp.symbols("a b c", positive=True)
    matrix = sp.Matrix([[a, b], [b, c]])

    assert is_positive_definite(matrix) is None
    assert is_positive_definite(matrix, positive=[a * c - b**2]) is True


def test_constraint_evaluation():
    x = sp.Symbol("x", positive=True)
    constraint = Constraint(x - 1, Relation.GT, "x > 1")

    assert constraint.holds({x: 2}) is True
    assert constraint.holds({x: 1}) is False
    assert constraint.holds({}) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_constraints.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement constraints and PD checks**

```python
# src/symlmm/constraints.py
"""Parameter constraints and positive-definiteness checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import sympy as sp


class Relation(str, Enum):
    """Constraint relation."""

    EQ = "eq"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


@dataclass(frozen=True)
class Constraint:
    """A named parameter constraint."""

    expr: sp.Expr
    relation: Relation
    message: str

    def holds(self, subs: Mapping[sp.Symbol, object]) -> bool | None:
        """Evaluate the constraint under substitutions; None means unknown."""
        value = sp.simplify(self.expr.subs(subs))
        if value.is_positive:
            return self.relation in (Relation.GT, Relation.GE)
        if value.is_zero:
            return self.relation in (Relation.EQ, Relation.LE, Relation.GE)
        if value.is_negative:
            return self.relation in (Relation.LT, Relation.LE)
        return None


def is_positive_definite(matrix: sp.MatrixBase, positive: Iterable[sp.Expr] = ()) -> bool | None:
    """Tri-state PD check using Sylvester minors.

    `positive` lists expressions known to be positive, such as
    `a*c - b**2` for a 2x2 covariance.
    """
    if not isinstance(matrix, sp.MatrixBase):
        return None
    size = matrix.rows
    if size == 0:
        return True
    known_positive = tuple(positive)
    for k in range(1, size + 1):
        minor = sp.simplify(matrix[:k, :k].det())
        if minor.is_positive:
            continue
        if any(sp.simplify(minor - expr) == 0 for expr in known_positive):
            continue
        if minor.is_nonpositive:
            return False
        return None
    return True


def check_constraints(
    constraints: Iterable[Constraint], subs: Mapping[sp.Symbol, object]
) -> tuple[str, ...]:
    """Return messages for constraints that evaluate to False."""
    violations = []
    for constraint in constraints:
        if constraint.holds(subs) is False:
            violations.append(constraint.message)
    return tuple(violations)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_constraints.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/constraints.py tests/unit/test_constraints.py
git commit -m "feat: add constraint objects and PD tri-state checks"
```

---

### Task T0.5: SymPy capability spike

**Files:**
- Create: `scripts/spike_sympy.py`
- Create: `docs/design/2026-11-09-sympy-capabilities.md`
- Test: `tests/unit/test_sympy_capabilities.py`

**Interfaces:**
- Produces: a decision record stating which SymPy features are usable and which fallbacks are required.

- [ ] **Step 1: Write the capability tests**

```python
# tests/unit/test_sympy_capabilities.py
import pytest
import sympy as sp


def test_matmul_entry_expands_to_sum():
    n, q = sp.symbols("n q", positive=True, integer=True)
    Z = sp.MatrixSymbol("Z", n, q)
    G = sp.MatrixSymbol("G", q, q)
    V = Z * G * Z.T
    assert V[0, 1].has(sp.Sum)


def test_blockdiag_requires_explicit_blocks():
    A = sp.MatrixSymbol("A", 2, 2)
    B = sp.MatrixSymbol("B", 3, 3)
    block = sp.BlockDiagMatrix([A, B])
    assert block.shape == (5, 5)


def test_elementwise_apply_function():
    n = sp.Symbol("n", positive=True, integer=True)
    x = sp.MatrixSymbol("x", n, 1)
    f = sp.matrices.expressions.applyfunc.ElementwiseApplyFunction(sp.exp, x)
    assert f.shape == (n, 1)


def test_diagonal_of_shape():
    n = sp.Symbol("n", positive=True, integer=True)
    V = sp.MatrixSymbol("V", n, n)
    d = sp.matrices.expressions.diagonal.DiagonalOf(V)
    assert d.shape == (n, 1)


def test_kronecker_delta_sum_collapses():
    i, j = sp.symbols("i j", integer=True)
    tau2 = sp.Symbol("tau2", positive=True)
    expr = sp.Sum(tau2 * sp.KroneckerDelta(i, j), (i, 0, 3)).doit()
    assert sp.simplify(expr.subs(j, 1)) == tau2


def test_symbolic_matrix_dimensions():
    n = sp.Symbol("n", positive=True, integer=True)
    A = sp.MatrixSymbol("A", n, n)
    assert A.shape == (n, n)
```

- [ ] **Step 2: Run the tests to establish the baseline**

Run: `uv run pytest tests/unit/test_sympy_capabilities.py -v`
Expected: all pass, or failures documented in Step 4.

- [ ] **Step 3: Write the spike script**

```python
# scripts/spike_sympy.py
"""Manual exploration of the SymPy capabilities required by symlmm."""

import sympy as sp


def main() -> None:
    n, q = sp.symbols("n q", positive=True, integer=True)
    Z = sp.MatrixSymbol("Z", n, q)
    G = sp.MatrixSymbol("G", q, q)
    print("MatMul entry:", (Z * G * Z.T)[0, 1])

    A = sp.MatrixSymbol("A", 2, 2)
    print("BlockDiag:", sp.BlockDiagMatrix([A, A]).shape)

    x = sp.MatrixSymbol("x", n, 1)
    print("Elementwise:", sp.matrices.expressions.applyfunc.ElementwiseApplyFunction(sp.exp, x))

    V = sp.MatrixSymbol("V", n, n)
    print("DiagonalOf:", sp.matrices.expressions.diagonal.DiagonalOf(V).shape)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Record the outcome and fallbacks**

Create `docs/design/2026-11-09-sympy-capabilities.md` with a table:

| Capability | Result | Fallback |
|---|---|---|
| `MatMul` entry expands to `Sum` | pass/fail | manual `Sum` construction in `model.py` |
| `BlockDiagMatrix` with symbolic block count | fail expected | abstract `G` symbol in matrix form |
| `ElementwiseApplyFunction` | pass/fail | custom `MatrixExpr` subclass in `forms.py` |
| `DiagonalOf` shape | pass/fail | `EntryMap` diagonal helper |
| `KroneckerDelta` sum collapse | pass/fail | explicit rewrite rule in `simplify.py` |
| symbolic `MatrixSymbol` dimensions | pass/fail | integer-only dense fallback |

If a capability fails, update `PLAN.md` §11 and the affected task before proceeding.

- [ ] **Step 5: Run the tests and static checks**

Run: `uv run pytest tests/unit/test_sympy_capabilities.py -v && uv run ruff check`
Expected: PASS, or recorded failures with fallbacks.

- [ ] **Step 6: Commit**

```bash
git add scripts/spike_sympy.py tests/unit/test_sympy_capabilities.py docs/design/2026-11-09-sympy-capabilities.md
git commit -m "chore: validate SymPy capabilities for symbolic matrices"
```

---

### Task T1.1: Fixed effects and grouping specs

**Files:**
- Modify: `src/symlmm/specs.py`
- Test: `tests/unit/test_specs.py`

**Interfaces:**
- Consumes: `Group`, `Nested`, `Crossed`
- Produces: `Fixed(intercept, slopes)`, `.names()`, `.coefficients()`, `.mean_at(model, i)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_specs.py
import sympy as sp

from symlmm.model import IndexedModel
from symlmm.specs import Fixed, RandomIntercept, Group


def test_fixed_names_and_coefficients():
    b0, b1 = sp.symbols("b0 b1")
    fixed = Fixed(intercept=b0, slopes={"time": b1})

    assert fixed.names() == ("intercept", "time")
    assert fixed.coefficients() == (b0, b1)


def test_fixed_mean_at_observation():
    b0, b1 = sp.symbols("b0 b1")
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), time="time", covariates=("time",))
    i, _ = model.indices()
    fixed = Fixed(intercept=b0, slopes={"time": b1})

    expected = b0 + b1 * model.covariate(i, "time")
    assert sp.simplify(fixed.mean_at(model, i) - expected) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_specs.py -v`
Expected: FAIL with `ImportError: cannot import name 'Fixed'`.

- [ ] **Step 3: Implement `Fixed`**

```python
# append to src/symlmm/specs.py
from typing import Mapping


@dataclass(frozen=True)
class Fixed:
    """Fixed-effect specification."""

    intercept: sp.Expr | None = None
    slopes: Mapping[str, sp.Expr] = None

    def __post_init__(self) -> None:
        if self.slopes is None:
            object.__setattr__(self, "slopes", {})

    def names(self) -> tuple[str, ...]:
        names = [] if self.intercept is None else ["intercept"]
        return tuple(names) + tuple(self.slopes)

    def coefficients(self) -> tuple[sp.Expr, ...]:
        values = [] if self.intercept is None else [self.intercept]
        return tuple(values) + tuple(self.slopes.values())

    def mean_at(self, model, i) -> sp.Expr:
        value = sp.Integer(0) if self.intercept is None else self.intercept
        for name, coefficient in self.slopes.items():
            value += coefficient * model.covariate(i, name)
        return value
```

Add `import sympy as sp` at the top of `specs.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_specs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/specs.py tests/unit/test_specs.py
git commit -m "feat: add fixed-effect specification"
```

---

### Task T1.2: Random-effect covariance structures

**Files:**
- Create: `src/symlmm/covariances.py`
- Test: `tests/unit/test_random_covariances.py`

**Interfaces:**
- Consumes: `Constraint`, `Relation`, `is_positive_definite`
- Produces: `RandomCovariance.entry(a, b)`, `.dim`, `.constraints()`, `.dense()`
- Produces: `Scalar`, `Diagonal`, `IID`, `Unstructured.from_sd_corr`, `Unstructured.from_cholesky`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_random_covariances.py
import sympy as sp

from symlmm.covariances import Diagonal, Scalar, Unstructured


def test_scalar_entry():
    tau2 = sp.Symbol("tau2", positive=True)
    covariance = Scalar(tau2)

    assert covariance.dim == 1
    assert covariance.entry(0, 0) == tau2
    assert covariance.entry(0, 1) == 0


def test_diagonal_entry():
    d0, d1 = sp.symbols("d0 d1", positive=True)
    covariance = Diagonal([d0, d1])

    assert covariance.dim == 2
    assert covariance.entry(0, 0) == d0
    assert covariance.entry(0, 1) == 0


def test_unstructured_from_sd_corr():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    covariance = Unstructured.from_sd_corr([s0, s1], rho)

    assert covariance.entry(0, 0) == s0**2
    assert covariance.entry(0, 1) == s0 * s1 * rho
    assert covariance.entry(1, 1) == s1**2
    assert any("rho" in constraint.message for constraint in covariance.constraints())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_random_covariances.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement random covariance structures**

```python
# src/symlmm/covariances.py
"""Covariance structures for random effects and residuals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import sympy as sp

from .constraints import Constraint, Relation


class CovarianceSpec(ABC):
    """Base class for covariance structures."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the covariance matrix."""

    @abstractmethod
    def entry(self, a, b) -> sp.Expr:
        """Return the (a, b) entry."""

    def dense(self) -> sp.Matrix:
        """Return the explicit matrix for this structure."""
        return sp.Matrix(
            self.dim, self.dim, lambda a, b: self.entry(a, b)
        )

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return ()


class Scalar(CovarianceSpec):
    """Scalar (1x1) covariance for a random intercept."""

    def __init__(self, var: sp.Expr) -> None:
        self.var = var

    @property
    def dim(self) -> int:
        return 1

    def entry(self, a, b) -> sp.Expr:
        return self.var if a == b == 0 else sp.Integer(0)

    def constraints(self) -> tuple[Constraint, ...]:
        return (Constraint(self.var, Relation.GT, "tau2 > 0"),)


class ResidualCovariance(ABC):
    """Base class for residual covariance structures."""

    @abstractmethod
    def entry(self, i, j, model) -> sp.Expr:
        """Return the (i, j) observation entry."""

    def constraints(self) -> tuple[Constraint, ...]:
        """Return parameter constraints."""
        return ()


class IID(ResidualCovariance):
    """IID residual covariance."""

    def __init__(self, *, var: sp.Expr) -> None:
        self.var = var

    def entry(self, i, j, model) -> sp.Expr:
        return self.var * sp.KroneckerDelta(i, j)

    def constraints(self) -> tuple[Constraint, ...]:
        return (Constraint(self.var, Relation.GT, "sigma2 > 0"),)


class Diagonal(CovarianceSpec):
    """Diagonal random-effect covariance."""

    def __init__(self, variances: Sequence[sp.Expr]) -> None:
        self.variances = tuple(variances)

    @property
    def dim(self) -> int:
        return len(self.variances)

    def entry(self, a, b) -> sp.Expr:
        if a != b:
            return sp.Integer(0)
        return self.variances[a]

    def constraints(self) -> tuple[Constraint, ...]:
        return tuple(
            Constraint(value, Relation.GT, f"d{a} > 0")
            for a, value in enumerate(self.variances)
        )


class Unstructured(CovarianceSpec):
    """Unstructured covariance, parameterized by SDs and correlations."""

    def __init__(self, matrix: sp.MatrixBase) -> None:
        self.matrix = matrix

    @classmethod
    def from_sd_corr(cls, sd: Sequence[sp.Expr], corr: sp.Expr) -> "Unstructured":
        if len(sd) != 2:
            raise ValueError("v0.1 supports 2x2 SD+corr construction")
        s0, s1 = sd
        matrix = sp.Matrix([[s0**2, s0 * s1 * corr], [s0 * s1 * corr, s1**2]])
        instance = cls(matrix)
        instance._corr = corr
        return instance

    @classmethod
    def from_cholesky(cls, lower: sp.MatrixBase) -> "Unstructured":
        return cls(lower * lower.T)

    @property
    def dim(self) -> int:
        return self.matrix.rows

    def entry(self, a, b) -> sp.Expr:
        return self.matrix[a, b]

    def constraints(self) -> tuple[Constraint, ...]:
        corr = getattr(self, "_corr", None)
        if corr is None:
            return ()
        return (
            Constraint(corr + 1, Relation.GT, "-1 < rho"),
            Constraint(1 - corr, Relation.GT, "rho < 1"),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_random_covariances.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/covariances.py tests/unit/test_random_covariances.py
git commit -m "feat: add random-effect covariance structures"
```

---

### Task T1.3: LMM assembly and matrix form

**Files:**
- Modify: `src/symlmm/model.py`
- Test: `tests/unit/test_lmm_matrix.py`

**Interfaces:**
- Consumes: `Fixed`, `RandomIntercept` term, `Scalar`, `IID`
- Produces: `LMM(fixed, random, residual, n)`, `.dimensions() -> (n, q)`, `.matrix_mean()`, `.matrix_covariance()`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_lmm_matrix.py
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept, RandomIntercept


def test_matrix_shapes():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    n, q = model.dimensions()
    m_subject = sp.Symbol("m_subject", positive=True, integer=True)

    assert model.matrix_mean().shape == (n, 1)
    assert model.matrix_covariance().shape == (n, n)
    assert q == m_subject
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_lmm_matrix.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add the `RandomIntercept` term, `LMM`, and group registration**

```python
# append to src/symlmm/specs.py
@dataclass(frozen=True)
class RandomIntercept:
    """Random-intercept term."""

    group: str
    cov: object
```

```python
# append to src/symlmm/model.py
import sympy as sp

from .covariances import IID  # noqa: F401  (re-exported for tests)
from .specs import Group


class LMM(IndexedModel):
    """Linear mixed model."""

    def __init__(self, *, fixed, random, residual, n=None) -> None:
        self.fixed = fixed
        self.random = tuple(random)
        self.residual = residual
        groups = tuple(Group(term.group) for term in self.random)
        super().__init__(
            n=n if n is not None else sp.Symbol("n", positive=True, integer=True),
            groups=groups,
            time=None,
            covariates=tuple(fixed.slopes),
        )

    def levels(self, factor: str):
        """Return the symbolic number of levels for a grouping factor."""
        return sp.Symbol(f"m_{factor}", positive=True, integer=True)

    def dimensions(self):
        """Return observations and total random-coefficient dimension."""
        q = sum(self.levels(term.group) * term.cov.dim for term in self.random)
        return self.n, q

    def matrix_mean(self):
        """Return the abstract matrix form of E[y]."""
        n = self.n
        p = len(self.fixed.coefficients())
        X = sp.MatrixSymbol("X", n, p)
        beta = sp.MatrixSymbol("beta", p, 1)
        return X * beta

    def matrix_covariance(self):
        """Return the abstract matrix form of V = Z G Z' + R."""
        n, q = self.dimensions()
        Z = sp.MatrixSymbol("Z", n, q)
        G = sp.MatrixSymbol("G", q, q)
        R = sp.MatrixSymbol("R", n, n)
        return Z * G * Z.T + R
```

The grouping factors must be registered from the random terms. Extend `LMM.__init__` to pass `groups=tuple(Group(term.group) for term in self.random)` and `time` when a random slope uses it. Import `Group` from `.specs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_lmm_matrix.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/model.py tests/unit/test_lmm_matrix.py
git commit -m "feat: assemble LMM matrix form"
```

---

### Task T1.4: Sum form and matrix-to-sum bridge

**Files:**
- Modify: `src/symlmm/covariances.py`
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/forms.py`
- Test: `tests/unit/test_forms_bridge.py`

**Interfaces:**
- Consumes: `LMM`, `EntryMap`
- Produces: `LMM.covariance_entry(i, j)`, `matrix_to_sum(expr, model)`, `sum_to_matrix(expr, model)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_forms_bridge.py
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def test_random_intercept_sum_entry():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()

    expected = (
        tau2 * sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject"))
        + sigma2 * sp.KroneckerDelta(i, j)
    )
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_forms_bridge.py -v`
Expected: FAIL with `AttributeError: 'LMM' object has no attribute 'covariance_entry'`.

- [ ] **Step 3: Implement entry methods and bridge**

```python
# IID.entry is already implemented in T1.2; do not redefine IID here.
```

```python
# append to src/symlmm/model.py
    def _random_term_entry(self, term, i, j):
        """Return the structured contribution of one random term."""
        zi = term.z(self, i) if hasattr(term, "z") else sp.Matrix([1])
        zj = term.z(self, j) if hasattr(term, "z") else sp.Matrix([1])
        value = (zi.T * term.cov.dense() * zj)[0, 0]
        return value * sp.KroneckerDelta(
            self.group(i, term.group), self.group(j, term.group)
        )

    def covariance_entry(self, i, j):
        """Return the structured sum-form covariance entry."""
        total = self.residual.entry(i, j, self)
        for term in self.random:
            total += self._random_term_entry(term, i, j)
        return total
```

`Scalar.dense()` returns `Matrix([[var]])`, so the random-intercept case reduces to
`tau2 * delta`.

```python
# append to src/symlmm/forms.py
def matrix_to_sum(expr, model) -> EntryMap:
    """Return a lazy entry map for an abstract matrix expression."""
    return EntryMap(model.covariance_entry)


def sum_to_matrix(expr, model):
    """Return the abstract matrix for a structured expression."""
    return model.matrix_covariance()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_forms_bridge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/covariances.py src/symlmm/model.py src/symlmm/forms.py tests/unit/test_forms_bridge.py
git commit -m "feat: add sum-form covariance bridge"
```

---

### Task T1.5: Moments and form dispatch

**Files:**
- Create: `src/symlmm/moments.py`
- Modify: `src/symlmm/model.py`
- Test: `tests/unit/test_moments.py`

**Interfaces:**
- Consumes: `require_form`, `DualForm`, `EntryMap`
- Produces: `LMM.mean`, `.covariance`, `.variance`, `.correlation`, `.conditional_mean`, `.conditional_variance`, `.conditional_covariance`, `.icc`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_moments.py
import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import FormArityError, MissingFormError
from symlmm.forms import DualForm
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def _model():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )


def test_form_is_required():
    model = _model()
    with pytest.raises(MissingFormError):
        model.mean()


def test_invalid_arity_raises():
    model = _model()
    i, j = model.indices()
    with pytest.raises(FormArityError):
        model.mean(i, j, form="sum")


def test_both_returns_dual_form():
    model = _model()
    i, _ = model.indices()
    result = model.mean(i, form="both")
    assert isinstance(result, DualForm)
    assert result.sum is not None
    assert result.matrix is not None


def test_correlation_entry():
    model = _model()
    i, j = model.indices()
    assert model.correlation(i, j, form="sum") is not None


def test_conditional_mean_entry():
    model = _model()
    i, _ = model.indices()
    assert model.conditional_mean(i, form="sum") is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_moments.py -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement moments and dispatch**

```python
# src/symlmm/moments.py
"""Moment queries for symbolic models."""

from __future__ import annotations

import sympy as sp

from .errors import FormArityError
from .forms import DualForm, EntryMap, Form, require_form


def mean(model, i=None, form=None):
    """Return the marginal mean."""
    form = require_form(form)
    if i is not None:
        if form is Form.BOTH:
            return DualForm(
                sum=model.fixed.mean_at(model, i),
                matrix=model.matrix_mean()[i, 0],
            )
        if form is Form.SUM:
            return model.fixed.mean_at(model, i)
        return model.matrix_mean()[i, 0]
    if form is Form.SUM:
        raise FormArityError("mean requires an index for form='sum'")
    matrix = model.matrix_mean()
    if form is Form.MATRIX:
        return matrix
    return DualForm(sum=EntryMap(lambda k, _=0: model.fixed.mean_at(model, k)), matrix=matrix)


def covariance(model, i=None, j=None, form=None):
    """Return the marginal covariance."""
    form = require_form(form)
    if (i is None) != (j is None):
        raise FormArityError("covariance requires both indices or neither")
    if i is None:
        if form is Form.SUM:
            raise FormArityError("covariance requires indices for form='sum'")
        matrix = model.matrix_covariance()
        if form is Form.MATRIX:
            return matrix
        return DualForm(sum=EntryMap(model.covariance_entry), matrix=matrix)
    if form is Form.SUM:
        return model.covariance_entry(i, j)
    if form is Form.MATRIX:
        return model.matrix_covariance()[i, j]
    return DualForm(
        sum=model.covariance_entry(i, j),
        matrix=model.matrix_covariance()[i, j],
    )


def variance(model, i=None, form=None):
    """Return the marginal variance."""
    form = require_form(form)
    if i is None:
        if form is Form.SUM:
            raise FormArityError("variance requires an index for form='sum'")
        matrix = sp.matrices.expressions.diagonal.DiagonalOf(model.matrix_covariance())
        if form is Form.MATRIX:
            return matrix
        return DualForm(sum=EntryMap(lambda k: model.covariance_entry(k, k)), matrix=matrix)
    if form is Form.SUM:
        return model.covariance_entry(i, i)
    if form is Form.MATRIX:
        return model.matrix_covariance()[i, i]
    return DualForm(
        sum=model.covariance_entry(i, i),
        matrix=model.matrix_covariance()[i, i],
    )


def correlation(model, i=None, j=None, form=None):
    """Return the marginal correlation."""
    form = require_form(form)
    if (i is None) != (j is None):
        raise FormArityError("correlation requires both indices or neither")
    if i is None:
        V = model.matrix_covariance()
        D = sp.matrices.expressions.diagonal.DiagonalOf(V)
        corr = V * D**sp.Rational(-1, 2)
        if form is Form.MATRIX:
            return corr
        return DualForm(
            sum=EntryMap(lambda a, b: _correlation_entry(model, a, b)),
            matrix=corr,
        )
    entry = _correlation_entry(model, i, j)
    if form is Form.SUM:
        return entry
    matrix = model.matrix_covariance()[i, j]
    if form is Form.MATRIX:
        return matrix
    return DualForm(sum=entry, matrix=matrix)


def _correlation_entry(model, i, j):
    vi = model.covariance_entry(i, i)
    vj = model.covariance_entry(j, j)
    return model.covariance_entry(i, j) / sp.sqrt(vi * vj)


def conditional_mean(model, i=None, form=None):
    """Return E[y | b]."""
    form = require_form(form)
    if i is None:
        n, q = model.dimensions()
        Z = sp.MatrixSymbol("Z", n, q)
        b = sp.MatrixSymbol("b", q, 1)
        matrix = model.matrix_mean() + Z * b
        if form is Form.SUM:
            raise FormArityError("conditional_mean requires an index for form='sum'")
        if form is Form.MATRIX:
            return matrix
        return DualForm(
            sum=EntryMap(lambda k: _conditional_mean_entry(model, k)),
            matrix=matrix,
        )
    entry = _conditional_mean_entry(model, i)
    if form is Form.SUM:
        return entry
    matrix = sp.MatrixSymbol("conditional_mu", model.n, 1)[i, 0]
    if form is Form.MATRIX:
        return matrix
    return DualForm(sum=entry, matrix=matrix)


def _conditional_mean_entry(model, i):
    value = model.fixed.mean_at(model, i)
    for term in model.random:
        z = term.z(model, i) if hasattr(term, "z") else sp.Matrix([1])
        for a in range(term.cov.dim):
            value += z[a] * sp.IndexedBase(f"b_{term.group}")[a]
    return value
```

Add methods to `LMM` that delegate to these functions. ICC arrives in T1.6:

```python
    def mean(self, i=None, *, form=None):
        from .moments import mean

        return mean(self, i, form)

    def covariance(self, i=None, j=None, *, form=None):
        from .moments import covariance

        return covariance(self, i, j, form)

    def variance(self, i=None, *, form=None):
        from .moments import variance

        return variance(self, i, form)

    def correlation(self, i=None, j=None, *, form=None):
        from .moments import correlation

        return correlation(self, i, j, form)

    def conditional_mean(self, i=None, *, form=None):
        from .moments import conditional_mean

        return conditional_mean(self, i, form)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_moments.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/moments.py src/symlmm/model.py tests/unit/test_moments.py
git commit -m "feat: add moment dispatch and form validation"
```

---

### Task T1.6: Closed-form RI, RS, and AR(1) expressions

**Files:**
- Modify: `src/symlmm/covariances.py`
- Modify: `src/symlmm/model.py`
- Create: `src/symlmm/simplify.py`
- Test: `tests/unit/test_closed_forms.py`

**Interfaces:**
- Consumes: `LMM.covariance_entry`
- Produces: `canonicalize(expr)`, closed-form `RI`, `RS`, `AR1` entries

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_closed_forms.py
import sympy as sp

from symlmm.covariances import AR1, IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept, RandomSlope


def test_random_intercept_closed_form():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    expected = tau2 * sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject")) + sigma2 * sp.KroneckerDelta(i, j)
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def test_ar1_residual_closed_form():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(group="subject", time="time", var=sigma2, rho=rho),
    )
    i, j = model.indices()
    expected = (
        sigma2
        * rho ** sp.Abs(model.time(i) - model.time(j))
        * sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject"))
    )
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def test_icc_for_random_intercept():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    assert sp.simplify(model.icc(form="sum") - tau2 / (tau2 + sigma2)) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_closed_forms.py -v`
Expected: FAIL with `ImportError` for `AR1` or `RandomSlope`.

- [ ] **Step 3: Implement `AR1` and canonicalization**

```python
# append to src/symlmm/covariances.py
class AR1:
    """AR(1) residual covariance within groups."""

    def __init__(self, *, group: str, time: str, var: sp.Expr, rho: sp.Expr) -> None:
        self.group = group
        self.time = time
        self.var = var
        self.rho = rho

    def entry(self, i, j, model) -> sp.Expr:
        same_group = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
        lag = sp.Abs(model.time(i, self.time) - model.time(j, self.time))
        return self.var * self.rho**lag * same_group

    def constraints(self) -> tuple[Constraint, ...]:
        return (
            Constraint(self.rho + 1, Relation.GT, "-1 < rho"),
            Constraint(1 - self.rho, Relation.GT, "rho < 1"),
            Constraint(self.var, Relation.GT, "sigma2 > 0"),
        )
```

```python
# src/symlmm/simplify.py
"""Canonicalization and rewrite rules."""

from __future__ import annotations

import sympy as sp


def canonicalize(expr: sp.Expr) -> sp.Expr:
    """Return a canonical form for comparison and printing."""
    return sp.simplify(sp.expand(expr))
```

`RandomSlope` is implemented in T2.6; keep the RS closed-form test in that task. The RI and AR(1) tests above are the acceptance gate for this task.

Update `LMM.__init__` to register the residual grouping factor and time:

```python
        groups = tuple(Group(term.group) for term in self.random)
        if hasattr(residual, "group"):
            groups = groups + (Group(residual.group),)
        time = getattr(residual, "time", None) or next(
            (getattr(term, "time", None) for term in self.random if getattr(term, "time", None)),
            None,
        )
        super().__init__(n=..., groups=groups, time=time, covariates=tuple(fixed.slopes))
```

Implement `LMM.icc`:

```python
    def icc(self, *, at=None, form=None):
        """Return the conditional variance partition coefficient."""
        from .forms import DualForm, require_form

        form = require_form(form)
        random_part = sp.Integer(0)
        for term in self.random:
            z = term.z(self, 0) if hasattr(term, "z") else sp.Matrix([1])
            if at and hasattr(term, "time"):
                z = z.subs({self.time(0, term.time): at.get("time", self.time(0, term.time))})
            random_part += (z.T * term.cov.dense() * z)[0, 0]
        residual_part = self.residual.entry(0, 0, self)
        value = random_part / (random_part + residual_part)
        if form.value == "sum":
            return value
        return DualForm(sum=value, matrix=value)
```

For random slopes, `at={"time": ...}` substitutes the evaluation point; for random intercepts, `at` is ignored.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_closed_forms.py -v -k "not random_slope"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/covariances.py src/symlmm/model.py src/symlmm/simplify.py tests/unit/test_closed_forms.py
git commit -m "feat: add RI and AR(1) closed forms"
```

---

### Task T2.1: Residual covariance structures

**Files:**
- Modify: `src/symlmm/covariances.py`
- Test: `tests/unit/test_residual_covariances.py`

**Interfaces:**
- Produces: `IID`, `CompoundSymmetry`, `Toeplitz`, `Heteroscedastic`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_residual_covariances.py
import sympy as sp

from symlmm.covariances import CompoundSymmetry, Heteroscedastic, IID, Toeplitz
from symlmm.model import IndexedModel
from symlmm.specs import Group


def _model():
    n = sp.Symbol("n", positive=True, integer=True)
    return IndexedModel(n=n, groups=(Group("subject"),), time="time")


def test_compound_symmetry_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)
    model = _model()
    i, j = model.indices()
    cov = CompoundSymmetry(group="subject", var=sigma2, rho=rho)
    assert cov.entry(i, i, model) == sigma2
    assert cov.entry(i, j, model) == sigma2 * rho


def test_iid_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = _model()
    i, j = model.indices()
    cov = IID(var=sigma2)
    assert cov.entry(i, i, model) == sigma2
    assert cov.entry(i, j, model) == 0


def test_heteroscedastic_entry():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    model = _model()
    cov = Heteroscedastic(variances=[s0, s1])
    assert cov.entry(0, 0, model) == s0
    assert cov.entry(1, 1, model) == s1
    assert cov.entry(0, 1, model) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_residual_covariances.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the structures**

```python
# append to src/symlmm/covariances.py
class CompoundSymmetry(ResidualCovariance):
    """Compound-symmetry residual covariance within groups."""

    def __init__(self, *, group: str, var: sp.Expr, rho: sp.Expr) -> None:
        self.group = group
        self.var = var
        self.rho = rho

    def entry(self, i, j, model) -> sp.Expr:
        same = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
        return sp.Piecewise(
            (self.var, sp.Eq(i, j)),
            (self.var * self.rho * same, True),
        )

    def constraints(self) -> tuple[Constraint, ...]:
        return (
            Constraint(self.rho + 1, Relation.GT, "rho > -1/(m_min - 1)"),
            Constraint(self.var, Relation.GT, "sigma2 > 0"),
        )


class Toeplitz(ResidualCovariance):
    """Toeplitz residual covariance with lag parameters."""

    def __init__(
        self,
        *,
        group: str,
        time: str,
        variances: Sequence[sp.Expr],
        correlations: Sequence[sp.Expr],
    ) -> None:
        self.group = group
        self.time = time
        self.variances = tuple(variances)
        self.correlations = tuple(correlations)

    def entry(self, i, j, model) -> sp.Expr:
        lag = sp.Abs(model.time(i, self.time) - model.time(j, self.time))
        value = self.variances[0] * self.correlations[lag - 1] if lag > 0 else self.variances[0]
        same = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
        return value * same

    def constraints(self) -> tuple[Constraint, ...]:
        return (Constraint(self.variances[0], Relation.GT, "sigma2 > 0"),)


class Heteroscedastic(ResidualCovariance):
    """Observation-specific residual variances."""

    def __init__(self, *, variances: Sequence[sp.Expr]) -> None:
        self.variances = tuple(variances)

    def entry(self, i, j, model) -> sp.Expr:
        if i == j:
            return self.variances[int(i)]
        return sp.Integer(0)

    def constraints(self) -> tuple[Constraint, ...]:
        return tuple(
            Constraint(value, Relation.GT, f"sigma2[{k}] > 0")
            for k, value in enumerate(self.variances)
        )
```

Use `model.group(i, "subject")` only when the structure is constructed with a group. Add a `group` argument to `CompoundSymmetry` and `Toeplitz`; update the tests to pass `group="subject"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_residual_covariances.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/covariances.py tests/unit/test_residual_covariances.py
git commit -m "feat: add residual covariance structures"
```

---

### Task T2.2: Spatial covariance structures

**Files:**
- Modify: `src/symlmm/covariances.py`
- Test: `tests/unit/test_spatial_covariances.py`

**Interfaces:**
- Consumes: `model.covariate(i, "x")`, `model.covariate(i, "y")`
- Produces: `Exponential`, `Gaussian`, `Matern`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_spatial_covariances.py
import sympy as sp

from symlmm.covariances import Exponential, Gaussian, Matern


def test_exponential_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)
    cov = Exponential(var=sigma2, range=phi)
    x0, x1 = sp.symbols("x0 x1", real=True)
    expected = sigma2 * sp.exp(-sp.Abs(x0 - x1) / phi)
    assert sp.simplify(cov.entry_from_distance(sp.Abs(x0 - x1)) - expected) == 0


def test_matern_half_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)
    cov = Matern(var=sigma2, range=phi, nu=sp.Rational(1, 2))
    h = sp.Symbol("h", positive=True)
    assert sp.simplify(cov.entry_from_distance(h) - sigma2 * sp.exp(-h / phi)) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_spatial_covariances.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement spatial kernels**

```python
# append to src/symlmm/covariances.py
class Exponential:
    """Exponential spatial covariance."""

    def __init__(self, *, var: sp.Expr, range: sp.Expr) -> None:
        self.var = var
        self.range = range

    def entry_from_distance(self, distance: sp.Expr) -> sp.Expr:
        return self.var * sp.exp(-distance / self.range)


class Gaussian:
    """Gaussian spatial covariance."""

    def __init__(self, *, var: sp.Expr, range: sp.Expr) -> None:
        self.var = var
        self.range = range

    def entry_from_distance(self, distance: sp.Expr) -> sp.Expr:
        return self.var * sp.exp(-(distance**2) / (2 * self.range**2))


class Matern:
    """Matern spatial covariance for nu in {1/2, 3/2, 5/2}."""

    def __init__(self, *, var: sp.Expr, range: sp.Expr, nu: sp.Expr) -> None:
        self.var = var
        self.range = range
        self.nu = nu

    def entry_from_distance(self, distance: sp.Expr) -> sp.Expr:
        scaled = distance / self.range
        if self.nu == sp.Rational(1, 2):
            return self.var * sp.exp(-scaled)
        if self.nu == sp.Rational(3, 2):
            return self.var * (1 + sp.sqrt(3) * scaled) * sp.exp(-sp.sqrt(3) * scaled)
        if self.nu == sp.Rational(5, 2):
            return self.var * (1 + sp.sqrt(5) * scaled + 5 * scaled**2 / 3) * sp.exp(-sp.sqrt(5) * scaled)
        raise ValueError(f"unsupported Matern nu: {self.nu}")
```

Add `entry(i, j, model)` that computes Euclidean distance from `model.covariate(i, "x")` and `model.covariate(i, "y")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_spatial_covariances.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/covariances.py tests/unit/test_spatial_covariances.py
git commit -m "feat: add spatial covariance kernels"
```

---

### Task T2.3: Variograms and covariogram

**Files:**
- Create: `src/symlmm/variogram.py`
- Modify: `src/symlmm/model.py`
- Test: `tests/unit/test_variograms.py`

**Interfaces:**
- Consumes: `LMM.covariance_entry`, `LMM.mean`
- Produces: `variogram(model, i, j, kind, form)`, `covariogram(model, i, j, form)`, `LMM.variogram`, `LMM.covariogram`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_variograms.py
import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import MissingFormError
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def _model():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )


def test_variogram_is_zero_on_diagonal():
    model = _model()
    i, _ = model.indices()
    assert sp.simplify(model.variogram(i, i, kind="residual", form="sum")) == 0


def test_kind_is_required():
    model = _model()
    i, j = model.indices()
    with pytest.raises(TypeError):
        model.variogram(i, j, form="sum")


def test_covariogram_equals_covariance():
    model = _model()
    i, j = model.indices()
    assert sp.simplify(
        model.covariogram(i, j, form="sum") - model.covariance(i, j, form="sum")
    ) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_variograms.py -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement variograms**

```python
# src/symlmm/variogram.py
"""Variogram and covariogram queries."""

from __future__ import annotations

import sympy as sp

from .errors import InvalidFormError
from .forms import DualForm, Form, require_form


def covariogram(model, i, j, form=None):
    """Return Cov(y_i, y_j)."""
    form = require_form(form)
    entry = model.covariance_entry(i, j)
    if form is Form.SUM:
        return entry
    matrix = model.matrix_covariance()[i, j]
    if form is Form.MATRIX:
        return matrix
    return DualForm(sum=entry, matrix=matrix)


def variogram(model, i, j, kind, form=None):
    """Return a raw, residual, or conditional variogram entry."""
    form = require_form(form)
    if kind not in {"raw", "residual", "conditional"}:
        raise ValueError(f"invalid variogram kind: {kind!r}")
    if kind == "residual":
        value = (model.covariance_entry(i, i) + model.covariance_entry(j, j) - 2 * model.covariance_entry(i, j)) / 2
    elif kind == "conditional":
        value = (model.residual.entry(i, i, model) + model.residual.entry(j, j, model) - 2 * model.residual.entry(i, j, model)) / 2
    else:
        mean_difference = model.fixed.mean_at(model, i) - model.fixed.mean_at(model, j)
        value = mean_difference**2 / 2 + (
            model.covariance_entry(i, i) + model.covariance_entry(j, j) - 2 * model.covariance_entry(i, j)
        ) / 2
    if form is Form.SUM:
        return value
    matrix = sp.MatrixSymbol("Gamma", model.n, model.n)[i, j]
    if form is Form.MATRIX:
        return matrix
    return DualForm(sum=value, matrix=matrix)
```

Add delegating methods to `LMM`:

```python
    def variogram(self, i, j, *, kind, form=None):
        from .variogram import variogram

        return variogram(self, i, j, kind, form)

    def covariogram(self, i, j, *, form=None):
        from .variogram import covariogram

        return covariogram(self, i, j, form)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_variograms.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/variogram.py src/symlmm/model.py tests/unit/test_variograms.py
git commit -m "feat: add variogram and covariogram queries"
```

---

### Task T2.4: BLUP moments

**Files:**
- Create: `src/symlmm/blup.py`
- Modify: `src/symlmm/model.py`
- Test: `tests/unit/test_blup.py`

**Interfaces:**
- Consumes: `LMM.matrix_covariance`, `LMM.matrix_mean`
- Produces: `blup_mean`, `blup_covariance`, `blup_cross_covariance`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_blup.py
import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import UnsupportedForm
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def _model():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )


def test_blup_matrix_form():
    model = _model()
    result = model.blup_mean(form="matrix")
    assert result.matrix is not None


def test_blup_sum_form_is_gated():
    model = _model()
    with pytest.raises(UnsupportedForm):
        model.blup_mean(form="sum")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_blup.py -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement BLUP**

```python
# src/symlmm/blup.py
"""BLUP moment expressions."""

from __future__ import annotations

import sympy as sp

from .errors import UnsupportedForm
from .forms import DualForm, require_form


def blup_mean(model, form=None):
    """Return E[b | y] = G Z' V^-1 (y - X beta)."""
    form = require_form(form)
    if form.value == "sum":
        raise UnsupportedForm("BLUP mean has no general sum form")
    n, q = model.dimensions()
    G = sp.MatrixSymbol("G", q, q)
    Z = sp.MatrixSymbol("Z", n, q)
    V = model.matrix_covariance()
    X = sp.MatrixSymbol("X", n, len(model.fixed.coefficients()))
    beta = sp.MatrixSymbol("beta", len(model.fixed.coefficients()), 1)
    y = sp.MatrixSymbol("y", n, 1)
    expression = G * Z.T * V.inverse() * (y - X * beta)
    matrix = expression
    if form.value == "matrix":
        return matrix
    return DualForm(sum=None, matrix=matrix)


def blup_covariance(model, form=None):
    """Return Cov(b | y) = (G^-1 + Z' R^-1 Z)^-1."""
    form = require_form(form)
    if form.value == "sum":
        raise UnsupportedForm("BLUP covariance has no general sum form")
    n, q = model.dimensions()
    G = sp.MatrixSymbol("G", q, q)
    Z = sp.MatrixSymbol("Z", n, q)
    R = sp.MatrixSymbol("R", n, n)
    expression = (G.inverse() + Z.T * R.inverse() * Z).inverse()
    if form.value == "matrix":
        return expression
    return DualForm(sum=None, matrix=expression)
```

Add `blup_cross_covariance(i, form)` returning `Z[i, :] * G`; add delegating methods on `LMM`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_blup.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/blup.py src/symlmm/model.py tests/unit/test_blup.py
git commit -m "feat: add BLUP moments"
```

---

### Task T2.5: Dense mode

**Files:**
- Modify: `src/symlmm/model.py`
- Create: `src/symlmm/evaluate.py`
- Test: `tests/unit/test_dense.py`
- Test: `tests/unit/test_evaluate.py`

**Interfaces:**
- Produces: `LMM.dense(observations)`, `.matrix_covariance() -> Matrix`, `.matrix_z()`, `.matrix_x()`
- Produces: `evaluate.subs`, `evaluate.evaluate`, `evaluate.lambdify`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_dense.py
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def test_dense_covariance_is_concrete():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    matrix = dense.matrix_covariance()
    assert matrix.shape == (4, 4)
    assert matrix[0, 1] == tau2
    assert matrix[0, 2] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_dense.py -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement dense mode**

```python
# append to src/symlmm/model.py
    def dense(self, **observations):
        """Return a concrete copy or dense view with observation data."""
        dense = object.__new__(type(self))
        dense.__dict__ = self.__dict__.copy()
        dense._observations = observations
        dense._dense = True
        dense.n = len(next(iter(observations.values())))
        return dense

    def _column(self, name: str):
        data = self._observations[name]
        return sp.Matrix(data)

    def matrix_x(self):
        if not getattr(self, "_dense", False):
            return self.matrix_mean()
        columns = [sp.ones(self.n, 1)]
        for name in self.fixed.slopes:
            columns.append(self._column(name))
        return sp.Matrix.hstack(*columns)

    def matrix_z(self):
        if not getattr(self, "_dense", False):
            n, q = self.dimensions()
            return sp.MatrixSymbol("Z", n, q)
        columns = []
        for term in self.random:
            labels = list(self._observations[term.group])
            times = list(self._observations.get(term.time, [0] * len(labels)))
            for level in sorted(set(labels)):
                for a in range(term.cov.dim):
                    column = []
                    for label, t in zip(labels, times):
                        if label != level:
                            column.append(0)
                        elif term.cov.dim == 1:
                            column.append(1)
                        else:
                            column.append(1 if a == 0 else t)
                    columns.append(sp.Matrix(column))
        return sp.Matrix.hstack(*columns)

    def matrix_g(self):
        if not getattr(self, "_dense", False):
            _, q = self.dimensions()
            return sp.MatrixSymbol("G", q, q)
        blocks = []
        for term in self.random:
            labels = sorted(set(self._observations[term.group]))
            blocks.extend([term.cov.dense()] * len(labels))
        return sp.BlockDiagMatrix(blocks).as_explicit()

    def matrix_r(self):
        if not getattr(self, "_dense", False):
            return sp.MatrixSymbol("R", self.n, self.n)
        return sp.Matrix(
            self.n,
            self.n,
            lambda a, b: self.residual.entry(sp.Integer(a), sp.Integer(b), self),
        )

    def matrix_mean(self):
        if not getattr(self, "_dense", False):
            n = self.n
            p = len(self.fixed.coefficients())
            X = sp.MatrixSymbol("X", n, p)
            beta = sp.MatrixSymbol("beta", p, 1)
            return X * beta
        return self.matrix_x() * sp.Matrix(self.fixed.coefficients())

    def matrix_covariance(self):
        if not getattr(self, "_dense", False):
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
```

Problem: `covariance_entry` calls `model.group(i, ...)`, which returns `IndexedBase(...)[i]`. In dense mode, replace `group`/`time`/`covariate` to return actual values. Override in `dense()` or make accessors check `_dense`. Implement:

```python
    def group(self, i, factor):
        if getattr(self, "_dense", False):
            return sp.Integer(self._observations[factor][int(i)])
        return super().group(i, factor)

    def time(self, i, coordinate="time"):
        if getattr(self, "_dense", False):
            name = self.time_name if coordinate == "time" else coordinate
            return sp.Integer(self._observations[name][int(i)])
        return super().time(i, coordinate)

    def covariate(self, i, name):
        if getattr(self, "_dense", False):
            return sp.Integer(self._observations[name][int(i)])
        return super().covariate(i, name)
```

Also `LMM.__init__` must register groups. Update:

```python
        groups = tuple(Group(term.group) for term in self.random)
        super().__init__(n=..., groups=groups, time=None, covariates=tuple(fixed.slopes))
```

- [ ] **Step 4: Implement evaluation helpers**

```python
# src/symlmm/evaluate.py
"""Substitution, numeric evaluation, and lambdification."""

from __future__ import annotations

import sympy as sp

from .errors import UnresolvedSymbols


def subs(model, mapping):
    """Return a model with substituted parameters."""
    clone = object.__new__(type(model))
    clone.__dict__ = model.__dict__.copy()
    clone._substitutions = {**getattr(model, "_substitutions", {}), **mapping}
    return clone


def evaluate(model, mapping):
    """Return a numeric SymPy object after substitution."""
    result = model.subs(mapping)
    expr = result.matrix_covariance()
    free = expr.free_symbols - set(mapping)
    if free:
        raise UnresolvedSymbols(f"unresolved symbols: {sorted(map(str, free))}")
    return expr


def lambdify(model, args, form="matrix"):
    """Return a callable over the declared arguments."""
    expr = model.matrix_covariance() if form == "matrix" else model.mean()
    return sp.lambdify(args, expr, modules="numpy")
```

Add delegating `LMM.subs`, `.evaluate`, `.lambdify` methods.

```python
# tests/unit/test_evaluate.py
import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import UnresolvedSymbols
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def _model():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=4,
    )


def test_subs_is_non_mutating():
    model = _model()
    original = model.__dict__.copy()
    substituted = model.subs({sp.Symbol("b0"): 1})
    assert model.__dict__ == original
    assert substituted is not model


def test_evaluate_reports_unresolved_symbols():
    model = _model()
    with pytest.raises(UnresolvedSymbols):
        model.evaluate({sp.Symbol("b0"): 1})
```

- [ ] **Step 4b: Wire dense BLUP and add the equality test**

In `blup.py`, select concrete matrices when the model is dense:

```python
    dense = getattr(model, "_dense", False)
    G = model.matrix_g() if dense else sp.MatrixSymbol("G", q, q)
    Z = model.matrix_z() if dense else sp.MatrixSymbol("Z", n, q)
    R = model.matrix_r() if dense else sp.MatrixSymbol("R", n, n)
```

```python
# tests/integration/test_blup_dense.py
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def test_blup_dense_matches_explicit_formula():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    V = dense.matrix_covariance()
    G = dense.matrix_g()
    Z = dense.matrix_z()
    expected = (G.inverse() + Z.T * V.inverse() * Z).inverse()
    assert sp.simplify(dense.blup_covariance(form="matrix") - expected) == sp.zeros(2, 2)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_dense.py tests/unit/test_evaluate.py tests/integration/test_blup_dense.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/model.py src/symlmm/evaluate.py src/symlmm/blup.py tests/unit/test_dense.py tests/unit/test_evaluate.py tests/integration/test_blup_dense.py
git commit -m "feat: add dense mode and evaluation helpers"
```

---

### Task T2.6: Random slopes, multiple terms, nested and crossed factors

**Files:**
- Modify: `src/symlmm/specs.py`
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/covariances.py`
- Test: `tests/unit/test_random_slopes.py`
- Test: `tests/integration/test_multiple_terms.py`
- Test: `tests/integration/test_nested_crossed.py`

**Interfaces:**
- Produces: `RandomSlope(group, time, cov)`, multi-term assembly, composite group labels
- Produces: `BlockDiag(blocks)` for explicit block structures (no symbolic block count)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_random_slopes.py
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept, RandomSlope


def test_random_slope_closed_form():
    b0 = sp.Symbol("b0")
    s0 = sp.Symbol("s0", positive=True)
    s1 = sp.Symbol("s1", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomSlope(
                group="subject",
                time="time",
                cov=_sd_corr(s0, s1, rho_b),
            )
        ],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    ti, tj = model.time(i), model.time(j)
    expected = (
        s0**2
        + s0 * s1 * rho_b * (ti + tj)
        + s1**2 * ti * tj
    ) * sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject")) + sigma2 * sp.KroneckerDelta(i, j)
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def _sd_corr(s0, s1, rho):
    from symlmm.covariances import Unstructured

    return Unstructured.from_sd_corr([s0, s1], rho)
```

```python
# tests/integration/test_multiple_terms.py
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def test_two_random_terms_sum():
    b0 = sp.Symbol("b0")
    tau_a = sp.Symbol("tau_a", positive=True)
    tau_b = sp.Symbol("tau_b", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomIntercept(group="subject", cov=Scalar(tau_a)),
            RandomIntercept(group="item", cov=Scalar(tau_b)),
        ],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    expected = (
        tau_a * sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject"))
        + tau_b * sp.KroneckerDelta(model.group(i, "item"), model.group(j, "item"))
        + sigma2 * sp.KroneckerDelta(i, j)
    )
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0
```

```python
# tests/integration/test_nested_crossed.py
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Crossed, Fixed, Nested, RandomIntercept


def test_nested_and_crossed_labels_feed_sum_form():
    b0 = sp.Symbol("b0")
    tau_v = sp.Symbol("tau_v", positive=True)
    tau_x = sp.Symbol("tau_x", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomIntercept(group="visit", cov=Scalar(tau_v)),
            RandomIntercept(group="item", cov=Scalar(tau_x)),
        ],
        residual=IID(var=sigma2),
        groups=(Nested("visit", within="subject"), Crossed("subject", "item")),
    )
    i, j = model.indices()
    assert model.group(i, "visit").func == sp.IndexedBase("subject:visit", shape=(model.n,))
    assert model.group(i, "item").func == sp.IndexedBase("subject:item", shape=(model.n,))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_random_slopes.py tests/integration/test_multiple_terms.py tests/integration/test_nested_crossed.py -v`
Expected: FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 3: Implement `RandomSlope` and group registration**

```python
# append to src/symlmm/specs.py
@dataclass(frozen=True)
class RandomSlope:
    """Random slope with a 2x2 covariance."""

    group: str
    time: str
    cov: object

    def z(self, model, i):
        return sp.Matrix([1, model.time(i, self.time)])

    def entry(self, model, i, j):
        zi = self.z(model, i)
        zj = self.z(model, j)
        value = (zi.T * self.cov.matrix * zj)[0, 0]
        return value * sp.KroneckerDelta(
            model.group(i, self.group), model.group(j, self.group)
        )
```

Update `LMM.covariance_entry` to use `term.entry(self, i, j)` when available, otherwise the RI path. Extend `LMM.__init__` with `groups=None`; use the supplied `Group`, `Nested`, and `Crossed` specs when present, otherwise infer `Group(term.group)` from random terms. Register time from `RandomSlope` and from residual structures that declare a time variable.

`BlockDiag` accepts an explicit list of blocks plus a grouping factor and is dense-construction only in v0.1; symbolic block counts are not supported (see decision 9).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_random_slopes.py tests/integration/test_multiple_terms.py tests/integration/test_nested_crossed.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full core suite**

Run: `uv run pytest tests/unit tests/integration -v`
Expected: PASS for all core tasks.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/specs.py src/symlmm/model.py src/symlmm/covariances.py tests/unit/test_random_slopes.py tests/integration/test_multiple_terms.py tests/integration/test_nested_crossed.py
git commit -m "feat: add random slopes, multiple terms, and nested/crossed factors"
```
