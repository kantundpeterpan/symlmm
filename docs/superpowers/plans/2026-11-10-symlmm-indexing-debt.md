# M2.5 Indexing-Debt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the load-bearing M2 indexing, design, time, level, and ICC inconsistencies before M3.

**Architecture:** Two sequential tasks. Task 1 changes `model.py` only: fixed-effect design, canonical time resolution, dense composite level codes, and ICC time evaluation. Task 2 changes covariance/query internals only: constraint messages, covariogram delegation, and heteroscedastic range checks.

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

### Task 1: Model design, time, levels, and ICC

**Files:**
- Modify: `src/symlmm/model.py`
- Test: `tests/unit/test_indices.py`
- Test: `tests/unit/test_lmm_matrix.py`
- Test: `tests/unit/test_dense.py`
- Test: `tests/unit/test_closed_forms.py`

**Interfaces:**
- Consumes: `Fixed.names()`, `Fixed.coefficients()`, `Group`, `Nested`, `Crossed`, `RandomSlope`, `ModelError`.
- Produces: `LMM(..., time: str | None = None)`, `LMM.design(i)` aligned with `Fixed`, `LMM.levels(factor) -> int | sp.Symbol`, dense composite group codes, `LMM.icc(at=...)` requiring `time` for random slopes.

- [ ] **Step 1: Write the failing design/time tests**

Append to `tests/unit/test_indices.py`:

```python
def test_lmm_design_uses_fixed_columns_without_intercept():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomIntercept

    model = LMM(
        fixed=Fixed(intercept=None, slopes={"time": b1}),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    i, _ = model.indices()
    assert model.design(i) == sp.Matrix([model.covariate(i, "time")])
    assert model.design(i).shape == (len(model.fixed.coefficients()), 1)


def test_lmm_design_uses_declared_slope_name():
    b0 = sp.Symbol("b0")
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomIntercept

    model = LMM(
        fixed=Fixed(intercept=b0, slopes={"visit_time": b1}),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        time="visit_time",
    )
    i, _ = model.indices()
    assert model.design(i) == sp.Matrix([1, model.covariate(i, "visit_time")])


def test_conflicting_term_time_coordinates_raise():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.errors import ModelError
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomSlope

    with pytest.raises(ModelError, match="conflicting time coordinates"):
        LMM(
            fixed=Fixed(intercept=b1),
            random=[RandomSlope(group="subject", time="time",
                                cov=Diagonal([tau2, tau2]))],
            residual=AR1(group="subject", time="visit_time", var=sigma2, rho=rho),
        )


def test_single_declared_time_coordinate_is_inferred():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomSlope

    model = LMM(
        fixed=Fixed(intercept=b1),
        random=[RandomSlope(group="subject", time="visit_time",
                            cov=Diagonal([tau2, tau2]))],
        residual=AR1(group="subject", time="visit_time", var=sigma2, rho=rho),
    )
    assert model.time_name == "visit_time"


def test_explicit_time_coordinate_validates_terms():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.errors import ModelError
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomSlope

    with pytest.raises(ModelError, match="conflicts with term coordinates"):
        LMM(
            fixed=Fixed(intercept=b1),
            random=[RandomSlope(group="subject", time="time",
                                cov=Diagonal([tau2, tau2]))],
            residual=AR1(group="subject", time="visit_time", var=sigma2, rho=rho),
            time="visit_time",
        )
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/test_indices.py -q`
Expected: FAIL on the new design/time tests (missing `time` keyword, old design rows, no conflict error).

- [ ] **Step 3: Implement design and time resolution**

In `src/symlmm/model.py`:

```python
class LMM(IndexedModel):
    def __init__(
        self,
        *,
        fixed: Fixed,
        random: Iterable[RandomIntercept | RandomSlope],
        residual: ResidualCovariance,
        groups: Sequence[GroupSpec] | None = None,
        n: Any | None = None,
        time: str | None = None,
    ) -> None:
        ...
        declared: list[str] = []
        residual_time = getattr(residual, "time", None)
        if residual_time is not None:
            declared.append(residual_time)
        for term in self.random:
            term_time = getattr(term, "time", None)
            if term_time is not None:
                declared.append(term_time)
        unique = tuple(dict.fromkeys(declared))
        if time is not None:
            conflicts = tuple(name for name in unique if name != time)
            if conflicts:
                raise ModelError(
                    f"time coordinate {time!r} conflicts with term coordinates "
                    f"{conflicts!r}"
                )
            time_name = time
        elif len(unique) > 1:
            raise ModelError(
                f"conflicting time coordinates {unique!r}; "
                "declare one canonical time with LMM(time=...)"
            )
        else:
            time_name = unique[0] if unique else None
        super().__init__(
            n=n if n is not None else sp.Symbol("n", positive=True, integer=True),
            groups=tuple(registered.values()),
            time=time_name,
            covariates=tuple(fixed.slopes),
        )

    def design(self, i: Any) -> Any:
        """Return the fixed-effect design row in ``Fixed.names()`` order."""
        terms = [] if self.fixed.intercept is None else [sp.Integer(1)]
        terms.extend(self.covariate(i, name) for name in self.fixed.slopes)
        return sp.Matrix(terms) if terms else sp.zeros(0, 1)
```

- [ ] **Step 4: Write the failing levels/dense tests**

Append to `tests/unit/test_lmm_matrix.py`:

```python
def test_levels_match_composite_dimensions():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    nested = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="visit", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=[Nested("visit", within="subject")],
    )
    assert nested.levels("visit") == sp.Symbol(
        "m_subject_visit", positive=True, integer=True
    )
    assert nested.dimensions()[1] == nested.levels("visit")
```

Append to `tests/unit/test_dense.py`:

```python
def test_dense_nested_uses_composite_level_codes():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.specs import Nested

    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="visit", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=[Nested("visit", within="subject")],
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], visit=[0, 1, 0, 1])
    assert dense.dimensions()[1] == 4
    assert dense.levels("visit") == 4
    assert dense.group(0, "visit") == dense.group(1, "visit") - 1
    assert dense.matrix_z().shape == (4, 4)
    assert dense.matrix_g().shape == (4, 4)


def test_dense_crossed_uses_composite_level_codes():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.specs import Crossed

    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="item", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=[Crossed("subject", "item")],
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], item=[0, 1, 0, 1])
    assert dense.dimensions()[1] == 4
    assert dense.matrix_z().shape == (4, 4)


def test_dense_design_matches_matrix_x_rows():
    b0 = sp.Symbol("b0")
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0, slopes={"time": b1}),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=3,
    )
    dense = model.dense(subject=[0, 0, 1], time=[0, 2, 1])
    for k in range(3):
        assert dense.design(k) == dense.matrix_x()[k, :].T
```

- [ ] **Step 5: Implement dense level codes and `levels`**

In `src/symlmm/model.py`, replace the `group` dense branch and `levels`/`dimensions`/`matrix_z`/`matrix_g` label sources:

```python
class IndexedModel:
    def group(self, i: Any, factor: str) -> Any:
        """Return the group label for observation `i` and factor `factor`."""
        spec = self._group_spec(factor)
        if self._dense:
            return self._dense_group_codes(spec)[int(i)]
        return sp.IndexedBase(spec.label, shape=(self.n,))[i]

    def _dense_group_codes(self, spec: GroupSpec) -> list[int]:
        """Return integer codes for the composite label of `spec`."""
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
        return [codes[label] for label in raw]


class LMM:
    def levels(self, factor: str) -> Any:
        """Return the number of distinct labels for a grouping factor."""
        spec = self._group_spec(factor)
        if self._dense:
            return len(set(self._dense_group_codes(spec)))
        label = spec.label.replace(":", "_")
        return sp.Symbol(f"m_{label}", positive=True, integer=True)

    def dimensions(self) -> tuple[Any, Any]:
        """Return observations and total random-coefficient dimension."""
        q = sum(self.levels(term.group) * term.cov.dim for term in self.random)
        return self.n, q

    def matrix_z(self) -> Any:
        if not self._dense:
            n, q = self.dimensions()
            return sp.MatrixSymbol("Z", n, q)
        columns = []
        for term in self.random:
            labels = self._dense_group_codes(self._group_spec(term.group))
            time_name = getattr(term, "time", None)
            times = (
                [self.time(k, time_name) for k in range(self.n)]
                if time_name is not None
                else [0] * self.n
            )
            for level in sorted(set(labels)):
                for a in range(term.cov.dim):
                    column = []
                    for label, t in zip(labels, times, strict=True):
                        if label != level:
                            column.append(0)
                        elif term.cov.dim == 1:
                            column.append(1)
                        else:
                            column.append(1 if a == 0 else t)
                    columns.append(sp.Matrix(column))
        if not columns:
            return sp.zeros(self.n, 0)
        return sp.Matrix.hstack(*columns)

    def matrix_g(self) -> Any:
        if not self._dense:
            _, q = self.dimensions()
            return sp.MatrixSymbol("G", q, q)
        from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix

        blocks = []
        for term in self.random:
            labels = sorted(set(self._dense_group_codes(self._group_spec(term.group))))
            blocks.extend([term.cov.dense()] * len(labels))
        if not blocks:
            return sp.zeros(0, 0)
        return BlockDiagMatrix(*blocks).as_explicit()
```

- [ ] **Step 6: Run the levels/dense tests**

Run: `uv run pytest tests/unit/test_lmm_matrix.py tests/unit/test_dense.py tests/unit/test_indices.py -q`
Expected: PASS.

- [ ] **Step 7: Write the failing ICC tests**

Append to `tests/unit/test_closed_forms.py`:

```python
def test_icc_random_slope_requires_time():
    b0 = sp.Symbol("b0")
    tau2, sigma2 = sp.symbols("tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomSlope(group="subject", time="time", cov=Diagonal([tau2, tau2]))],
        residual=IID(var=sigma2),
    )
    with pytest.raises(ModelError, match="icc requires at"):
        model.icc(form="sum")


def test_icc_random_slope_evaluates_at_given_time():
    b0 = sp.Symbol("b0")
    tau2, sigma2 = sp.symbols("tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomSlope(group="subject", time="time", cov=Diagonal([tau2, tau2]))],
        residual=IID(var=sigma2),
    )
    value = model.icc(at={"time": 2}, form="sum")
    assert sp.simplify(value - (tau2 + 4 * tau2) / (tau2 + 4 * tau2 + sigma2)) == 0
    assert not value.has(sp.Indexed)
```

Add imports to `tests/unit/test_closed_forms.py`:

```python
from symlmm.covariances import AR1, Diagonal, IID, Scalar
from symlmm.errors import MissingFormError, ModelError
from symlmm.specs import Fixed, Group, RandomIntercept, RandomSlope
```

- [ ] **Step 8: Implement ICC time requirement**

In `src/symlmm/model.py`, replace `icc`:

```python
def icc(
    self,
    *,
    at: Mapping[str, Any] | None = None,
    form: str | Form | None = None,
) -> Any:
    """Return the conditional variance partition coefficient."""
    from .forms import DualForm, require_form

    form = require_form(form)
    time_terms = [term for term in self.random if getattr(term, "time", None)]
    if time_terms and at is None:
        raise ModelError(
            "icc requires at={'time': value} for time-varying random slopes"
        )
    canonical = self.time_name
    time_value = None
    if at is not None:
        time_value = at.get("time")
        if time_value is None and canonical is not None:
            time_value = at.get(canonical)
        if time_terms and time_value is None:
            raise ModelError(
                "icc requires at={'time': value} for time-varying random slopes"
            )
    random_part = sp.Integer(0)
    for term in self.random:
        z_accessor = getattr(term, "z", None)
        z = z_accessor(self, 0) if z_accessor is not None else sp.Matrix([1])
        term_time = getattr(term, "time", None)
        if time_value is not None and term_time is not None:
            z = z.subs({self.time(0, term_time): time_value})
        random_part += (z.T * term.cov.dense() * z)[0, 0]
    residual_part = self.residual.entry(0, 0, self)
    if time_value is not None and canonical is not None:
        residual_part = residual_part.subs({self.time(0, canonical): time_value})
    value = random_part / (random_part + residual_part)
    if form.value == "sum":
        return value
    return DualForm(sum=value, matrix=value)
```

- [ ] **Step 9: Run the ICC tests**

Run: `uv run pytest tests/unit/test_closed_forms.py -q`
Expected: PASS.

- [ ] **Step 10: Run the full task gate and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`
Expected: all pass.

```bash
git add src/symlmm/model.py tests/unit/test_indices.py tests/unit/test_lmm_matrix.py tests/unit/test_dense.py tests/unit/test_closed_forms.py
git commit -m "fix: align design, time, level, and ICC semantics"
```

---

### Task 2: Covariance and query minors

**Files:**
- Modify: `src/symlmm/covariances.py`
- Modify: `src/symlmm/variogram.py`
- Test: `tests/unit/test_random_covariances.py`
- Test: `tests/unit/test_residual_covariances.py`
- Test: `tests/unit/test_spatial_covariances.py`
- Test: `tests/unit/test_variograms.py`

**Interfaces:**
- Consumes: `Constraint`, `Relation`, `ModelError`, `moments.covariance`.
- Produces: symbol-derived constraint messages, `covariogram` delegation, `Heteroscedastic` concrete range errors.

- [ ] **Step 1: Write failing constraint-message and range tests**

Append to `tests/unit/test_random_covariances.py`:

```python
def test_scalar_constraint_message_uses_actual_symbol():
    s0 = sp.Symbol("s0", positive=True)
    messages = [constraint.message for constraint in Scalar(s0).constraints()]
    assert messages == ["s0 > 0"]
```

Update `tests/unit/test_residual_covariances.py`:

```python
def test_heteroscedastic_constraints():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    cov = Heteroscedastic(variances=[s0, s1])
    messages = [constraint.message for constraint in cov.constraints()]
    assert messages == ["s0 > 0", "s1 > 0"]


def test_heteroscedastic_rejects_concrete_out_of_range():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    model = _model()
    cov = Heteroscedastic(variances=[s0, s1])
    with pytest.raises(ModelError, match="out of range"):
        cov.entry(2, 2, model)
    with pytest.raises(ModelError, match="out of range"):
        cov.entry(2, 0, model)
```

Add `ModelError` to the errors import in `tests/unit/test_residual_covariances.py`.

Update the expected spatial messages in `tests/unit/test_spatial_covariances.py`:

```python
assert [constraint.message for constraint in constraints] == ["sigma2 > 0", "phi > 0"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/test_random_covariances.py tests/unit/test_residual_covariances.py -q`
Expected: FAIL on `s0 > 0` vs hardcoded messages and missing range error.

- [ ] **Step 3: Implement symbol-derived positive constraints**

In `src/symlmm/covariances.py`, add:

```python
def _positive(value: sp.Expr) -> Constraint:
    """Return a positivity constraint naming the actual expression."""
    return Constraint(value, Relation.GT, f"{value} > 0")
```

Use `_positive(...)` for the variance in `Scalar`, `IID`, `CompoundSymmetry`, `Toeplitz`, `AR1`, `Heteroscedastic`, `Diagonal`, `Exponential`, `Gaussian`, and `Matern`. Use it for the spatial `range` too. Keep the explicit rho messages in `CompoundSymmetry`, `AR1`, and `Unstructured`.

- [ ] **Step 4: Implement heteroscedastic concrete range checks**

In `Heteroscedastic.entry`:

```python
def _concrete_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if getattr(value, "is_Integer", False):
        return int(value)
    return None


class Heteroscedastic(ResidualCovariance):
    def entry(self, i: int, j: int, model: Any) -> sp.Expr:
        """Return the (i, j) observation entry."""
        for value in (i, j):
            index = _concrete_index(value)
            if index is not None and not 0 <= index < len(self.variances):
                raise ModelError(
                    f"heteroscedastic variance index out of range: {index}"
                )
        diagonal = sp.Piecewise(
            *((value, sp.Eq(i, k)) for k, value in enumerate(self.variances)),
            (sp.Integer(0), True),
        )
        return diagonal * sp.KroneckerDelta(i, j)
```

Import `ModelError` from `.errors`.

- [ ] **Step 5: Write the covariogram-delegation test**

Append to `tests/unit/test_variograms.py`:

```python
def test_covariogram_delegates_to_covariance():
    from symlmm.moments import covariance
    from symlmm.variogram import covariogram

    model = _model()
    i, j = model.indices()
    assert covariogram(model, i, j, form="sum") == covariance(model, i, j, form="sum")
```

- [ ] **Step 6: Implement covariogram delegation**

In `src/symlmm/variogram.py`:

```python
from .moments import covariance as _covariance


def covariogram(model: Any, i: Any, j: Any, form: str | Form | None = None) -> Any:
    """Return Cov(y_i, y_j); alias of ``moments.covariance``."""
    return _covariance(model, i, j, form)
```

- [ ] **Step 7: Document AR(1) time semantics**

In `src/symlmm/covariances.py`, expand the `AR1.__init__` docstring:

```python
"""Build an AR(1) residual covariance.

The covariance is ``var * rho**Abs(t_i - t_j)``. This is the discrete-time
AR(1) form when ``time`` is numeric and unit-spaced; for irregular or
non-unit spacing it is the continuous-time exponential covariance with
``rho`` per unit time. ``LMM`` rejects residual/random time declarations
that conflict with the canonical time coordinate.
"""
```

- [ ] **Step 8: Run the full task gate and commit**

Run: `uv run pytest -q && uv run ruff check && uv run ruff format --check && uv run mypy src/symlmm`
Expected: all pass.

```bash
git add src/symlmm/covariances.py src/symlmm/variogram.py tests/unit/test_random_covariances.py tests/unit/test_residual_covariances.py tests/unit/test_spatial_covariances.py tests/unit/test_variograms.py
git commit -m "fix: tighten covariance messages and query aliases"
```

---

## Final Gate

- [ ] Run `uv run pytest -q` (all tests).
- [ ] Run `uv run ruff check`.
- [ ] Run `uv run ruff format --check`.
- [ ] Run `uv run mypy src/symlmm`.
- [ ] Whole-branch review on the most capable model.
- [ ] Merge to `main` with `--no-ff`, push, update `CHANGELOG.md` if behavior changed.
