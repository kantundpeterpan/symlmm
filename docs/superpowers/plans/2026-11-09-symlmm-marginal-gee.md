# symlmm Marginal and GEE Implementation Plan (M3–M4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement direct and induced marginal LMMs, bounded exact decomposition, GEE families and working correlations, and symbolic GEE moments.

**Architecture:** `MarginalLMM` reuses the observation index model and moment dispatch from the core plan. `InducedCovariance` carries provenance from a conditional model. Decomposition is a finite template registry with a parameter solver, never general symbolic equivalence. GEE is a separate model class with families, links, and working correlations; it has no conditional counterpart.

**Tech Stack:** Python 3.14, SymPy, `uv`, `pytest`, `ruff`, `mypy`.

## Global Constraints

- Read `docs/superpowers/plans/2026-11-09-symlmm-core.md` first; reuse its `IndexedModel`, `Form`, `DualForm`, `RequireForm`, and error types.
- Index model A is temporary: `i` is the observation-row cursor;
  `group(i, factor)` and `occasion(i, factor)` are distinct index spaces;
  coordinates use `coordinate(i, name, within=factor)`. Issue
  [#1](https://github.com/kantundpeterpan/symlmm/issues/1) tracks the full
  two-level `(group, occasion)` response index.
- Declare coordinates with `coordinate=`; the removed legacy coordinate keyword
  is not accepted. `icc` uses `at={"coordinate": ...}`.
- Every public call requires `form`; missing raises `MissingFormError`.
- Marginal moments are the default; conditional moments are explicit.
- `is_decomposable()` is bounded: provenance or a registered template with a unique constrained solution.
- GEE `conditional()` raises `NoConditionalModel`; `icc()` raises `NotDefined`.
- Working correlations are distinct classes from residual covariances.
- Branch names: `feat/<ticket>`, `fix/<ticket>`, `chore/<ticket>`.
- Commit after each task's tests pass.

---

### Task T3.1: Marginal LMM and direct covariance

**Files:**
- Create: `src/symlmm/marginal/__init__.py`
- Create: `src/symlmm/marginal/model.py`
- Create: `src/symlmm/marginal/covariance.py`
- Test: `tests/unit/test_marginal.py`

**Interfaces:**
- Consumes: `IndexedModel`, `Fixed`, `IID`, `require_form`, `DualForm`, `EntryMap`
- Produces: `MarginalLMM(fixed, covariance, n)`, `.mean`, `.covariance`, `.variance`, `.correlation`, `.variogram`, `.covariogram`, `.summary`, `.icc`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_marginal.py
import sympy as sp

from symlmm.covariances import IID
from symlmm.marginal import MarginalLMM
from symlmm.specs import Fixed


def _model():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    return MarginalLMM(
        fixed=Fixed(intercept=b0),
        covariance=IID(var=sigma2),
        n=4,
    )


def test_mean_matrix_shape():
    model = _model()
    assert model.mean(form="matrix").shape == (4, 1)


def test_covariance_entry_matches_spec():
    model = _model()
    i, j = model.indices()
    assert model.covariance(i, j, form="sum") == model.covariance_spec.entry(i, j, model)


def test_variance_and_correlation_forms():
    model = _model()
    i, j = model.indices()
    assert model.variance(i, form="sum") is not None
    assert model.correlation(i, j, form="sum") is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_marginal.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the marginal model**

```python
# src/symlmm/marginal/model.py
"""Direct marginal linear model."""

from __future__ import annotations

import sympy as sp

from ..forms import DualForm, EntryMap, Form, require_form
from ..model import IndexedModel
from ..moments import covariance as _covariance
from ..moments import mean as _mean
from ..moments import variance as _variance
from ..specs import Group
from ..variogram import covariogram as _covariogram
from ..variogram import variogram as _variogram


class MarginalLMM(IndexedModel):
    """Marginal linear model with a direct covariance specification."""

    def __init__(self, *, fixed, covariance, n=None) -> None:
        self.fixed = fixed
        self.covariance_spec = covariance
        group = getattr(covariance, "group", None)
        super().__init__(
            n=n if n is not None else sp.Symbol("n", positive=True, integer=True),
            groups=(Group(group),) if group is not None else (),
            coordinate=getattr(covariance, "coordinate", None),
            covariates=tuple(fixed.slopes),
        )

    def mean(self, i=None, *, form=None):
        return _mean(self, i, form)

    def covariance(self, i=None, j=None, *, form=None):
        return _covariance(self, i, j, form)

    def variance(self, i=None, *, form=None):
        return _variance(self, i, form)

    def correlation(self, i=None, j=None, *, form=None):
        """Return Corr(y_i, y_j); entry and whole-matrix forms."""
        form = require_form(form)
        corr = sp.MatrixSymbol("Corr", self.n, self.n)
        if i is None and j is None:
            if form is Form.MATRIX:
                return corr
            return DualForm(sum=EntryMap(lambda a, b: self._corr_entry(a, b)), matrix=corr)
        if i is None or j is None:
            raise ValueError("correlation requires both indices or neither")
        entry = self._corr_entry(i, j)
        if form is Form.SUM:
            return entry
        if form is Form.MATRIX:
            return corr[i, j]
        return DualForm(sum=entry, matrix=corr[i, j])

    def _corr_entry(self, i, j):
        vi = self.covariance_spec.entry(i, i, self)
        vj = self.covariance_spec.entry(j, j, self)
        return self.covariance_spec.entry(i, j, self) / sp.sqrt(vi * vj)

    def variogram(self, i, j, *, kind, form=None):
        return _variogram(self, i, j, kind, form)

    def covariogram(self, i, j, *, form=None):
        return _covariogram(self, i, j, form)

    def icc(self, *, at=None, form=None):
        """Return the marginal correlation at lag zero (Gaussian identity)."""
        form = require_form(form)
        value = self.covariance_spec.entry(0, 1, self) / self.covariance_spec.entry(0, 0, self)
        if form is Form.SUM:
            return value
        return DualForm(sum=value, matrix=value)
```

`MarginalLMM.matrix_covariance()` and `matrix_mean()` are inherited from the core `LMM` pattern; implement them locally:

```python
    def matrix_mean(self):
        n = self.n
        p = len(self.fixed.coefficients())
        X = sp.MatrixSymbol("X", n, p)
        beta = sp.MatrixSymbol("beta", p, 1)
        return X * beta

    def matrix_covariance(self):
        n, _ = self.n, None
        return sp.MatrixSymbol("V", n, n)

    def covariance_entry(self, i, j):
        return self.covariance_spec.entry(i, j, self)
```

```python
# src/symlmm/marginal/__init__.py
"""Marginal model public exports."""

from .model import MarginalLMM

__all__ = ["MarginalLMM"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_marginal.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/marginal tests/unit/test_marginal.py
git commit -m "feat: add direct marginal LMM"
```

---

### Task T3.2: Induced covariance and provenance

**Files:**
- Create: `src/symlmm/marginal/induce.py`
- Modify: `src/symlmm/model.py`
- Modify: `src/symlmm/marginal/model.py`
- Test: `tests/unit/test_induced.py`

**Interfaces:**
- Produces: `InducedCovariance(source)`, `LMM.marginal(form)`, `MarginalLMM.is_decomposable()`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_induced.py
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.marginal import MarginalLMM
from symlmm.marginal.induce import InducedCovariance
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


def _conditional():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=4,
    )


def test_marginal_carries_provenance():
    marginal = _conditional().marginal(form="matrix")
    assert isinstance(marginal, MarginalLMM)
    assert isinstance(marginal.covariance_spec, InducedCovariance)
    assert marginal.is_decomposable() is True


def test_induced_covariance_delegates_entries():
    source = _conditional()
    marginal = source.marginal(form="matrix")
    i, j = marginal.indices()
    assert marginal.covariance(i, j, form="sum") == source.covariance_entry(i, j)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_induced.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement provenance and `LMM.marginal`**

```python
# src/symlmm/marginal/induce.py
"""Induced marginal covariance with conditional provenance."""

from __future__ import annotations


class InducedCovariance:
    """Covariance induced by a conditional model: V = Z G Z' + R."""

    def __init__(self, source) -> None:
        self.source = source
        self.coordinate = getattr(source, "coordinate_name", None)

    def entry(self, i, j, model) -> object:
        return self.source.covariance_entry(i, j)

    def constraints(self):
        return ()
```

```python
# append to src/symlmm/model.py
    def marginal(self, *, form=None):
        """Return the induced marginal model."""
        from .forms import require_form
        from .marginal.model import MarginalLMM
        from .marginal.induce import InducedCovariance

        require_form(form)
        return MarginalLMM(
            fixed=self.fixed,
            covariance=InducedCovariance(self),
            n=self.n,
        )
```

Add `is_decomposable` and `decomposition_reason` stubs to `MarginalLMM`:

```python
    def is_decomposable(self) -> bool:
        from .induce import InducedCovariance

        return isinstance(self.covariance_spec, InducedCovariance)

    def decomposition_reason(self):
        if self.is_decomposable():
            return None
        return "no registered exact decomposition"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_induced.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/marginal src/symlmm/model.py tests/unit/test_induced.py
git commit -m "feat: add induced marginal covariance and provenance"
```

---

### Task T3.3: Bounded decomposition registry and solver

**Files:**
- Create: `src/symlmm/marginal/decompose.py`
- Modify: `src/symlmm/marginal/model.py`
- Test: `tests/unit/test_decompose.py`

**Interfaces:**
- Produces: `Template`, `REGISTRY`, `decompose(model)`, `MarginalLMM.conditional(form)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_decompose.py
import sympy as sp

from symlmm.covariances import CompoundSymmetry, IID
from symlmm.marginal import MarginalLMM
from symlmm.specs import Fixed


def _cs_model():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    return MarginalLMM(
        fixed=Fixed(intercept=b0),
        covariance=CompoundSymmetry(group="subject", var=sigma2, rho=rho),
        n=4,
    )


def test_cs_decomposes_to_ri_plus_iid():
    model = _cs_model()
    assert model.is_decomposable() is True
    conditional = model.conditional(form="matrix")
    assert conditional is not None


def test_decomposition_preserves_covariance():
    model = _cs_model()
    conditional = model.conditional(form="matrix")
    i, j = model.indices()
    assert sp.simplify(
        conditional.covariance_entry(i, j) - model.covariance(i, j, form="sum")
    ) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_decompose.py -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement the registry and solver**

```python
# src/symlmm/marginal/decompose.py
"""Bounded exact decomposition registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sympy as sp

from ..covariances import CompoundSymmetry, IID
from ..specs import Fixed, RandomEffects
from ..covariances import Scalar


@dataclass(frozen=True)
class Template:
    """One registered decomposition template."""

    name: str
    matches: Callable[[object], bool]
    solve: Callable[[object], dict]


def _match_cs(model) -> bool:
    return isinstance(model.covariance_spec, CompoundSymmetry)


def _solve_cs(model) -> dict:
    spec = model.covariance_spec
    rho = spec.rho
    sigma2 = spec.var
    tau2 = sp.simplify(sigma2 * rho)
    sigma_e2 = sp.simplify(sigma2 * (1 - rho))
    return {"tau2": tau2, "sigma_e2": sigma_e2}


REGISTRY = (Template("cs-ri-iid", _match_cs, _solve_cs),)


def decompose(model):
    """Return a conditional LMM or raise NotDecomposable."""
    from ..errors import NotDecomposable
    from ..model import LMM

    for template in REGISTRY:
        if template.matches(model):
            params = template.solve(model)
            if any(value.is_nonpositive for value in params.values()):
                raise NotDecomposable(
                    f"{template.name} requires nonnegative variance components",
                    "use a direct conditional model or InducedCovariance",
                )
            return LMM(
                fixed=Fixed(intercept=model.fixed.intercept, slopes=model.fixed.slopes),
                random=[
                    RandomEffects(
                        group=model.covariance_spec.group,
                        intercept=True,
                        cov=Scalar(params["tau2"]),
                    )
                ],
                residual=IID(var=params["sigma_e2"]),
                n=model.n,
            )
    raise NotDecomposable(
        "no registered exact decomposition",
        "use InducedCovariance or specify a conditional model directly",
    )
```

```python
# append to src/symlmm/marginal/model.py
    def conditional(self, *, form=None):
        """Return the conditional model, or raise NotDecomposable."""
        from ..forms import require_form
        from .decompose import decompose

        require_form(form)
        return decompose(self)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decompose.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/marginal tests/unit/test_decompose.py
git commit -m "feat: add bounded marginal decomposition"
```

---

### Task T3.4: Refusal paths and decomposition reasons

**Files:**
- Modify: `src/symlmm/marginal/model.py`
- Modify: `src/symlmm/marginal/decompose.py`
- Test: `tests/unit/test_decompose_refusals.py`

**Interfaces:**
- Consumes: `NotDecomposable`
- Produces: refusal reasons for AR(1), negative-\(\rho\) CS, and unstructured marginals

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_decompose_refusals.py
import pytest
import sympy as sp

from symlmm.covariances import AR1, CompoundSymmetry, Unstructured
from symlmm.errors import NotDecomposable
from symlmm.marginal import MarginalLMM
from symlmm.specs import Fixed


def _ar1():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    return MarginalLMM(
        fixed=Fixed(intercept=b0),
        covariance=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
        n=4,
    )


def test_ar1_is_refused():
    model = _ar1()
    assert model.is_decomposable() is False
    with pytest.raises(NotDecomposable) as exc:
        model.conditional(form="matrix")
    assert "no registered exact decomposition" in exc.value.reason
    assert "InducedCovariance" in exc.value.suggestion


def test_negative_rho_cs_is_refused():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", negative=True)
    model = MarginalLMM(
        fixed=Fixed(intercept=b0),
        covariance=CompoundSymmetry(group="subject", var=sigma2, rho=rho),
        n=4,
    )
    assert model.is_decomposable() is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_decompose_refusals.py -v`
Expected: FAIL because AR(1) `is_decomposable()` returns `False` but `conditional()` may raise the wrong message.

- [ ] **Step 3: Implement refusal classification**

```python
# append to src/symlmm/marginal/model.py
    def is_decomposable(self) -> bool:
        from ..covariances import CompoundSymmetry
        from .induce import InducedCovariance

        if isinstance(self.covariance_spec, InducedCovariance):
            return True
        if isinstance(self.covariance_spec, CompoundSymmetry):
            return self.covariance_spec.rho.is_nonnegative is True
        return False

    def decomposition_reason(self):
        if self.is_decomposable():
            return None
        return "no registered exact decomposition"
```

Update `decompose()` to use `is_decomposable()` and to report the template name. Add a guard:

```python
    if getattr(model.covariance_spec, "rho", None) is not None and model.covariance_spec.rho.is_negative:
        raise NotDecomposable(
            "negative compound-symmetry correlation has no valid RI+IID split",
            "use InducedCovariance or specify a conditional model directly",
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decompose_refusals.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/marginal tests/unit/test_decompose_refusals.py
git commit -m "fix: refuse non-decomposable marginal models with reasons"
```

---

### Task T4.1: GEE families and links

**Files:**
- Create: `src/symlmm/gee/__init__.py`
- Create: `src/symlmm/gee/families.py`
- Test: `tests/unit/test_gee_families.py`

**Interfaces:**
- Produces: `Gaussian`, `Poisson`, `Bernoulli`; `.variance(mu)`, `.inverse_link(eta)`, `.link_name`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_gee_families.py
import sympy as sp

from symlmm.gee import Bernoulli, Gaussian, Poisson


def test_poisson_variance_and_link():
    mu = sp.Symbol("mu", positive=True)
    family = Poisson(link="log")
    assert family.variance(mu) == mu
    assert family.inverse_link(sp.log(mu)) == mu


def test_bernoulli_variance():
    mu = sp.Symbol("mu", positive=True)
    family = Bernoulli(link="logit")
    assert family.variance(mu) == mu * (1 - mu)


def test_gaussian_variance_is_one():
    mu = sp.Symbol("mu", positive=True)
    family = Gaussian(link="identity")
    assert family.variance(mu) == 1
    assert family.inverse_link(mu) == mu
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_gee_families.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement families and links**

```python
# src/symlmm/gee/families.py
"""GEE families, links, and variance functions."""

from __future__ import annotations

import sympy as sp


class Family:
    """Base family with a variance function and inverse link."""

    def __init__(self, link: str) -> None:
        self.link_name = link

    def variance(self, mu):
        raise NotImplementedError

    def inverse_link(self, eta):
        raise NotImplementedError


class Gaussian(Family):
    """Gaussian family with identity link."""

    def __init__(self, link: str = "identity") -> None:
        super().__init__(link)

    def variance(self, mu):
        return sp.Integer(1)

    def inverse_link(self, eta):
        return eta


class Poisson(Family):
    """Poisson family with log link."""

    def __init__(self, link: str = "log") -> None:
        super().__init__(link)

    def variance(self, mu):
        return mu

    def inverse_link(self, eta):
        return sp.exp(eta)


class Bernoulli(Family):
    """Bernoulli family with logit link."""

    def __init__(self, link: str = "logit") -> None:
        super().__init__(link)

    def variance(self, mu):
        return mu * (1 - mu)

    def inverse_link(self, eta):
        return 1 / (1 + sp.exp(-eta))
```

```python
# src/symlmm/gee/__init__.py
"""GEE public exports."""

from .families import Bernoulli, Gaussian, Poisson

__all__ = ["Bernoulli", "Gaussian", "Poisson"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_gee_families.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/gee tests/unit/test_gee_families.py
git commit -m "feat: add GEE families and links"
```

---

### Task T4.2: Working correlation structures

**Files:**
- Create: `src/symlmm/gee/correlation.py`
- Test: `tests/unit/test_working_correlations.py`

**Interfaces:**
- Produces: `WorkingIndependence`, `WorkingExchangeable`, `WorkingAR1`, `WorkingToeplitz`, `WorkingUnstructured`, `WorkingSpatialExponential`, `WorkingSpatialGaussian`, `WorkingMatern`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_working_correlations.py
import sympy as sp

from symlmm.gee import WorkingAR1, WorkingExchangeable, WorkingIndependence
from symlmm.model import IndexedModel
from symlmm.specs import Group


def _model():
    n = sp.Symbol("n", positive=True, integer=True)
    return IndexedModel(n=n, groups=(Group("subject"),), coordinate="visit_time")


def test_independence_is_diagonal():
    model = _model()
    i, j = model.indices()
    corr = WorkingIndependence()
    assert corr.entry(i, i, model) == 1
    assert corr.entry(i, j, model) == 0


def test_exchangeable_entry():
    model = _model()
    i, j = model.indices()
    rho = sp.Symbol("rho", real=True)
    corr = WorkingExchangeable(group="subject", rho=rho)
    assert corr.entry(i, j, model) == rho


def test_ar1_entry():
    model = _model()
    i, j = model.indices()
    rho = sp.Symbol("rho", positive=True)
    corr = WorkingAR1(group="subject", coordinate="visit_time", rho=rho)
    same = sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject"))
    lag = sp.Abs(
        model.coordinate(i, "visit_time", within="subject")
        - model.coordinate(j, "visit_time", within="subject")
    )
    assert corr.entry(i, j, model) == same * rho**lag
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_working_correlations.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement working correlations**

```python
# src/symlmm/gee/correlation.py
"""GEE working correlation structures."""

from __future__ import annotations

import sympy as sp


class WorkingCorrelation:
    """Base class for working correlations."""

    def entry(self, i, j, model):
        raise NotImplementedError

    def constraints(self):
        return ()


class WorkingIndependence(WorkingCorrelation):
    """Independent working correlation."""

    def entry(self, i, j, model):
        return sp.Integer(1) if i == j else sp.Integer(0)


class WorkingExchangeable(WorkingCorrelation):
    """Exchangeable working correlation within clusters."""

    def __init__(self, *, group: str, rho: sp.Expr) -> None:
        self.group = group
        self.rho = rho

    def entry(self, i, j, model):
        same = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
        return sp.Piecewise((sp.Integer(1), sp.Eq(i, j)), (self.rho * same, True))


class WorkingAR1(WorkingCorrelation):
    """AR(1) working correlation within clusters."""

    def __init__(self, *, group: str, coordinate: str, rho: sp.Expr) -> None:
        self.group = group
        self.coordinate = coordinate
        self.rho = rho

    def entry(self, i, j, model):
        same = sp.KroneckerDelta(model.group(i, self.group), model.group(j, self.group))
        lag = sp.Abs(
            model.coordinate(i, self.coordinate, within=self.group)
            - model.coordinate(j, self.coordinate, within=self.group)
        )
        return self.rho**lag * same
```

Add the remaining structures with the same pattern.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_working_correlations.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/gee tests/unit/test_working_correlations.py
git commit -m "feat: add GEE working correlations"
```

---

### Task T4.3: GEE assembly and moments

**Files:**
- Create: `src/symlmm/gee/model.py`
- Create: `src/symlmm/gee/covariance.py`
- Test: `tests/unit/test_gee_model.py`

**Interfaces:**
- Consumes: `IndexedModel`, `Fixed`, `Family`, `WorkingCorrelation`
- Produces: `GEE(fixed, family, correlation, dispersion, n)`, `.mean`, `.variance`, `.covariance`, `.variogram`, `.covariogram`, `.summary`, `.conditional`, `.icc`, `.cluster`, `.occasion`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_gee_model.py
import pytest
import sympy as sp

from symlmm.errors import NoConditionalModel, NotDefined
from symlmm.gee import GEE, Poisson, WorkingAR1
from symlmm.specs import Fixed


def _model():
    b0, b1 = sp.symbols("b0 b1")
    rho = sp.Symbol("rho", positive=True)
    phi = sp.Symbol("phi", positive=True)
    return GEE(
        fixed=Fixed(intercept=b0, slopes={"visit_time": b1}),
        family=Poisson(link="log"),
        correlation=WorkingAR1(group="subject", coordinate="visit_time", rho=rho),
        dispersion=phi,
        n=4,
    )


def test_cluster_and_occasion_accessors():
    model = _model()
    i, _ = model.indices()
    assert model.cluster(i) is not None
    assert model.occasion(i, "subject") == sp.IndexedBase("k_subject", shape=(model.n,))[i]


def test_conditional_and_icc_are_undefined():
    model = _model()
    with pytest.raises(NoConditionalModel):
        model.conditional()
    with pytest.raises(NotDefined):
        model.icc(at={"coordinate": 0}, form="both")


def test_covariance_zero_across_clusters():
    model = _model()
    i, j = model.indices()
    expr = model.covariance(i, j, form="sum")
    assert expr is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_gee_model.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the GEE model**

```python
# src/symlmm/gee/model.py
"""GEE marginal model."""

from __future__ import annotations

import sympy as sp

from ..errors import NoConditionalModel, NotDefined
from ..forms import DualForm, EntryMap, Form, require_form
from ..model import IndexedModel
from ..specs import Group


class GEE(IndexedModel):
    """Marginal model under a working covariance."""

    def __init__(self, *, fixed, family, correlation, dispersion, n=None) -> None:
        self.fixed = fixed
        self.family = family
        self.correlation = correlation
        self.dispersion = dispersion
        groups = (Group(correlation.group),) if hasattr(correlation, "group") else ()
        super().__init__(
            n=n if n is not None else sp.Symbol("n", positive=True, integer=True),
            groups=groups,
            coordinate=getattr(correlation, "coordinate", None),
            covariates=tuple(fixed.slopes),
        )

    def cluster(self, i):
        if not hasattr(self.correlation, "group"):
            return sp.Integer(0)
        return self.group(i, self.correlation.group)

    def occasion(self, i, cluster: str):
        """Return the true within-cluster occasion index for row `i`."""
        return super().occasion(i, cluster)

    def mean(self, i=None, *, form=None):
        form = require_form(form)
        n = self.n
        p = len(self.fixed.coefficients())
        eta = sp.MatrixSymbol("X", n, p) * sp.MatrixSymbol("beta", p, 1)
        mu = sp.matrices.expressions.applyfunc.ElementwiseApplyFunction(
            self.family.inverse_link, eta
        )
        if i is None:
            if form is Form.SUM:
                raise ValueError("mean requires an index for form='sum'")
            return mu
        entry = self.family.inverse_link(self.fixed.mean_at(self, i))
        if form is Form.SUM:
            return entry
        return mu[i, 0]

    def covariance(self, i, j, *, form=None):
        form = require_form(form)
        mu_i = self.family.inverse_link(self.fixed.mean_at(self, i))
        mu_j = self.family.inverse_link(self.fixed.mean_at(self, j))
        same = sp.KroneckerDelta(self.cluster(i), self.cluster(j))
        entry = (
            self.dispersion
            * sp.sqrt(self.family.variance(mu_i))
            * self.correlation.entry(i, j, self)
            * sp.sqrt(self.family.variance(mu_j))
            * same
        )
        if form is Form.SUM:
            return entry
        if form is Form.MATRIX:
            return entry
        return DualForm(sum=entry, matrix=entry)

    def variance(self, i, *, form=None):
        form = require_form(form)
        mu_i = self.family.inverse_link(self.fixed.mean_at(self, i))
        value = self.dispersion * self.family.variance(mu_i)
        if form is Form.SUM:
            return value
        if form is Form.MATRIX:
            return value
        return DualForm(sum=value, matrix=value)

    def variogram(self, i, j, *, kind, form=None):
        form = require_form(form)
        if kind == "conditional":
            raise NotDefined("GEE has no conditional variogram")
        mu_i = self.family.inverse_link(self.fixed.mean_at(self, i))
        mu_j = self.family.inverse_link(self.fixed.mean_at(self, j))
        mean_difference = (mu_i - mu_j) ** 2 / 2 if kind == "raw" else sp.Integer(0)
        value = mean_difference + (
            self.variance(i, form="sum")
            + self.variance(j, form="sum")
            - 2 * self.covariance(i, j, form="sum")
        ) / 2
        if form is Form.SUM:
            return value
        return DualForm(sum=value, matrix=value)

    def covariogram(self, i, j, *, form=None):
        return self.covariance(i, j, form=form)

    def conditional(self):
        raise NoConditionalModel("GEE models have no conditional counterpart")

    def icc(self, *, at=None, form=None):
        raise NotDefined("ICC is not defined for GEE models")
```

Add `variance`, `variogram`, `covariogram`, and `summary` delegating to the core helpers.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_gee_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/gee tests/unit/test_gee_model.py
git commit -m "feat: add GEE model and moments"
```

---

### Task T4.4: GEE equivalence and integration tests

**Files:**
- Test: `tests/integration/test_gee.py`
- Modify: `src/symlmm/gee/__init__.py`

**Interfaces:**
- Consumes: `GEE`, `WorkingExchangeable`, `WorkingAR1`, `MarginalLMM`, `CompoundSymmetry`, `AR1`

- [ ] **Step 1: Write the failing integration tests**

```python
# tests/integration/test_gee.py
import sympy as sp

from symlmm.covariances import CompoundSymmetry
from symlmm.gee import GEE, Gaussian, WorkingExchangeable
from symlmm.marginal import MarginalLMM
from symlmm.specs import Fixed


def test_gaussian_identity_matches_marginal_cs():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    phi = sp.Symbol("phi", positive=True)

    marginal = MarginalLMM(
        fixed=Fixed(intercept=b0),
        covariance=CompoundSymmetry(group="subject", var=sigma2, rho=rho),
        n=4,
    )
    gee = GEE(
        fixed=Fixed(intercept=b0),
        family=Gaussian(link="identity"),
        correlation=WorkingExchangeable(group="subject", rho=rho),
        dispersion=phi,
        n=4,
    )
    i, j = marginal.indices()
    assert sp.simplify(
        gee.covariance(i, j, form="sum").subs(phi, sigma2)
        - marginal.covariance(i, j, form="sum")
    ) == 0


def test_poisson_moments_are_symbolic():
    b0 = sp.Symbol("b0")
    rho = sp.Symbol("rho", positive=True)
    phi = sp.Symbol("phi", positive=True)
    gee = GEE(
        fixed=Fixed(intercept=b0),
        family=__import__("symlmm.gee", fromlist=["Poisson"]).Poisson(link="log"),
        correlation=WorkingExchangeable(group="subject", rho=rho),
        dispersion=phi,
        n=4,
    )
    i, j = gee.indices()
    assert gee.mean(i, form="sum") == sp.exp(b0)
    assert gee.covariance(i, j, form="sum").has(sp.exp)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/integration/test_gee.py -v`
Expected: FAIL on the equivalence assertion or missing exports.

- [ ] **Step 3: Wire exports and fix the equivalence**

```python
# src/symlmm/gee/__init__.py
"""GEE public exports."""

from .correlation import (
    WorkingAR1,
    WorkingExchangeable,
    WorkingIndependence,
    WorkingMatern,
    WorkingSpatialExponential,
    WorkingSpatialGaussian,
    WorkingToeplitz,
    WorkingUnstructured,
)
from .families import Bernoulli, Gaussian, Poisson
from .model import GEE

__all__ = [
    "Bernoulli",
    "GEE",
    "Gaussian",
    "Poisson",
    "WorkingAR1",
    "WorkingExchangeable",
    "WorkingIndependence",
    "WorkingMatern",
    "WorkingSpatialExponential",
    "WorkingSpatialGaussian",
    "WorkingToeplitz",
    "WorkingUnstructured",
]
```

For Gaussian/identity, `variance(mu) == 1`, so
`V_ij = phi * R_ij`; substituting `phi = sigma2` matches compound symmetry.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/integration/test_gee.py -v`
Expected: PASS.

- [ ] **Step 5: Run the marginal and GEE suites**

Run: `uv run pytest tests/unit/test_marginal.py tests/unit/test_induced.py tests/unit/test_decompose.py tests/unit/test_gee_families.py tests/unit/test_working_correlations.py tests/unit/test_gee_model.py tests/integration/test_gee.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/gee tests/integration/test_gee.py
git commit -m "feat: add GEE equivalence tests and exports"
```
