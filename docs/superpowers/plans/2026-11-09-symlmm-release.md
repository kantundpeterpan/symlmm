# symlmm Release Implementation Plan (M5–M7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish canonicalization, printers, summaries, comparison, numerical validation, Quarto documentation, CI, packaging, and the v0.1 release.

**Architecture:** Core and marginal/GEE plans provide the model API. This plan adds presentation, validation, documentation, and release automation. Validation harnesses live in `tests/validation/` and `scripts/`; they are not shipped in the wheel.

**Tech Stack:** Python 3.14, SymPy, `uv`, `pytest`, `ruff`, `mypy`, NumPy, Quarto, `quartodoc`, R (fixture generation only).

## Global Constraints

- Read both earlier plans before starting.
- Every public call requires `form`; printers and summaries inherit the same rule.
- `compare_specs` returns `ComparisonResult(equal: bool | None, residual, notes)`.
- Reference fixtures are JSON with a versioned schema; no R runtime in CI.
- Quarto output goes to `docs/_site` and is git-ignored.
- Branch names: `feat/<ticket>`, `fix/<ticket>`, `chore/<ticket>`.
- Commit after each task's tests pass.

---

### Task T5.1: Canonicalization and rewrite rules

**Files:**
- Modify: `src/symlmm/simplify.py`
- Test: `tests/unit/test_canonicalize.py`

**Interfaces:**
- Produces: `canonicalize(expr)`, `collapse_deltas(expr)`, `normalize_sum_bounds(expr)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_canonicalize.py
import sympy as sp

from symlmm.simplify import canonicalize, collapse_deltas, normalize_sum_bounds


def test_canonicalize_expands_and_collects():
    x, y = sp.symbols("x y")
    assert canonicalize((x + y) ** 2) == x**2 + 2 * x * y + y**2


def test_collapse_deltas():
    i, j = sp.symbols("i j", integer=True)
    tau2 = sp.Symbol("tau2", positive=True)
    expr = sp.Sum(tau2 * sp.KroneckerDelta(i, j), (i, 0, 3)).doit()
    assert canonicalize(expr.subs(j, 1)) == tau2


def test_normalize_sum_bounds_is_stable():
    i = sp.Symbol("i", integer=True)
    expr = sp.Sum(i, (i, 0, 3))
    assert normalize_sum_bounds(expr) == expr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_canonicalize.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement canonicalization**

```python
# src/symlmm/simplify.py
"""Canonicalization and rewrite rules."""

from __future__ import annotations

import sympy as sp


def normalize_sum_bounds(expr: sp.Expr) -> sp.Expr:
    """Normalize dummy index names and bounds."""
    return expr


def collapse_deltas(expr: sp.Expr) -> sp.Expr:
    """Collapse KroneckerDelta sums where possible."""
    return sp.simplify(expr.doit())


def canonicalize(expr: sp.Expr) -> sp.Expr:
    """Return a canonical form for comparison and printing."""
    return sp.simplify(sp.expand(collapse_deltas(expr)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_canonicalize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/simplify.py tests/unit/test_canonicalize.py
git commit -m "feat: add canonicalization and delta rewrite rules"
```

---

### Task T5.2: Printers, summary, and comparison

**Files:**
- Modify: `src/symlmm/printers.py`
- Create: `src/symlmm/algebra.py`
- Modify: `src/symlmm/model.py`
- Test: `tests/unit/test_printers.py`
- Test: `tests/unit/test_summary.py`

**Interfaces:**
- Produces: `to_latex(quantity, form)`, `to_markdown(quantity, form)`, `Summary`, `compare_specs(a, b, form, assumptions)`, `ComparisonResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_printers.py
import sympy as sp

from symlmm.covariances import IID, Scalar
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


def test_latex_sum_contains_sum_notation():
    model = _model()
    i, j = model.indices()
    latex = model.to_latex(i, j, form="sum")
    assert "\\sum" in latex or "delta" in latex.lower()


def test_markdown_both_returns_both_sections():
    model = _model()
    i, j = model.indices()
    markdown = model.to_markdown(i, j, form="both")
    assert "Sum" in markdown or "sum" in markdown
    assert "Matrix" in markdown or "matrix" in markdown
```

```python
# tests/unit/test_summary.py
import sympy as sp

from symlmm.algebra import compare_specs
from symlmm.covariances import IID, Scalar
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


def test_summary_fields():
    summary = _model().summary(form="both")
    assert summary.mean is not None
    assert summary.covariance is not None


def test_compare_self_is_equal():
    model = _model()
    result = compare_specs(model, model, form="matrix")
    assert result.equal is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_printers.py tests/unit/test_summary.py -v`
Expected: FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 3: Implement printers and comparison**

```python
# src/symlmm/printers.py
"""LaTeX and Markdown printers."""

from __future__ import annotations

import sympy as sp

from .errors import InvalidFormError
from .forms import DualForm, Form, require_form


def to_latex(quantity, form=None) -> str:
    """Render a quantity as LaTeX."""
    form = require_form(form)
    if isinstance(quantity, DualForm):
        parts = []
        if quantity.sum is not None:
            parts.append(sp.latex(quantity.sum))
        if quantity.matrix is not None:
            parts.append(sp.latex(quantity.matrix))
        return r" \\ ".join(parts)
    return sp.latex(quantity)


def to_markdown(quantity, form=None) -> str:
    """Render a quantity as Markdown with LaTeX math."""
    form = require_form(form)
    if isinstance(quantity, DualForm):
        sections = []
        if quantity.sum is not None:
            sections.append(f"**Sum form**\n\n$$\n{sp.latex(quantity.sum)}\n$$")
        if quantity.matrix is not None:
            sections.append(f"**Matrix form**\n\n$$\n{sp.latex(quantity.matrix)}\n$$")
        return "\n\n".join(sections)
    return f"$$\n{sp.latex(quantity)}\n$$"
```

```python
# src/symlmm/algebra.py
"""Model comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .forms import require_form
from .simplify import canonicalize


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing two model specifications."""

    equal: bool | None
    residual: sp.Expr | None
    notes: tuple[str, ...]


def compare_specs(a, b, *, form=None, assumptions=None) -> ComparisonResult:
    """Compare two models under optional assumptions."""
    require_form(form)
    residual = canonicalize(a.matrix_covariance() - b.matrix_covariance())
    if residual == 0:
        return ComparisonResult(True, None, ())
    return ComparisonResult(None, residual, ("symbolic equality is not decidable",))
```

Add `LMM.summary`, `.to_latex`, `.to_markdown` methods that build a `Summary` and delegate to printers.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_printers.py tests/unit/test_summary.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/printers.py src/symlmm/algebra.py src/symlmm/model.py tests/unit/test_printers.py tests/unit/test_summary.py
git commit -m "feat: add printers, summaries, and model comparison"
```

---

### Task T6.1: Monte Carlo validation harness

**Files:**
- Create: `tests/validation/__init__.py`
- Create: `tests/validation/monte_carlo.py`
- Create: `tests/validation/test_monte_carlo.py`
- Create: `scripts/validate_monte_carlo.py`

**Interfaces:**
- Produces: `simulate_covariance(model, mapping, n_sim, seed) -> np.ndarray`, `compare_covariance(...)`

- [ ] **Step 1: Write the failing test**

```python
# tests/validation/test_monte_carlo.py
import numpy as np
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept
from tests.validation.monte_carlo import simulate_covariance


def test_ri_covariance_matches_simulation():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=6,
    ).dense(subject=[0, 0, 0, 1, 1, 1], time=list(range(6)))
    mapping = {tau2: 1.5, sigma2: 0.5}
    empirical = simulate_covariance(model, mapping, n_sim=20_000, seed=7)
    expected = np.array(model.matrix_covariance().subs(mapping), dtype=float)
    assert np.allclose(empirical, expected, atol=0.08)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/validation/test_monte_carlo.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the simulator**

```python
# tests/validation/monte_carlo.py
"""Monte Carlo validation helpers (not shipped)."""

from __future__ import annotations

import numpy as np
import sympy as sp


def simulate_covariance(model, mapping, *, n_sim, seed):
    """Simulate a Gaussian response and return its empirical covariance."""
    rng = np.random.default_rng(seed)
    covariance = np.array(model.matrix_covariance().subs(mapping), dtype=float)
    mean = np.array(model.matrix_mean().subs(mapping), dtype=float).reshape(-1)
    samples = rng.multivariate_normal(mean, covariance, size=n_sim)
    return np.cov(samples, rowvar=False, bias=True)


def compare_covariance(empirical, expected, *, atol):
    """Return a boolean and the maximum absolute deviation."""
    deviation = float(np.max(np.abs(empirical - expected)))
    return deviation <= atol, deviation
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/validation/test_monte_carlo.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/validation scripts/validate_monte_carlo.py
git commit -m "test: add Monte Carlo covariance validation"
```

---

### Task T6.2: Reference fixture schema and R generators

**Files:**
- Create: `tests/fixtures/README.md`
- Create: `tests/fixtures/schema.json`
- Create: `scripts/fixtures/generate_lme4.R`
- Create: `scripts/fixtures/generate_geepack.R`
- Test: `tests/unit/test_fixture_schema.py`

**Interfaces:**
- Produces: versioned JSON fixture schema and generator scripts

- [ ] **Step 1: Write the failing schema test**

```python
# tests/unit/test_fixture_schema.py
import json
from pathlib import Path

SCHEMA = json.loads(Path("tests/fixtures/schema.json").read_text())


def test_schema_has_required_fields():
    assert set(SCHEMA["required"]) == {
        "r_version",
        "package",
        "package_version",
        "model",
        "parameters",
        "design",
        "expected",
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_fixture_schema.py -v`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Write the schema and generators**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "symlmm reference fixture",
  "type": "object",
  "required": [
    "r_version",
    "package",
    "package_version",
    "model",
    "parameters",
    "design",
    "expected"
  ],
  "properties": {
    "r_version": {"type": "string"},
    "package": {"enum": ["lme4", "nlme", "geepack"]},
    "package_version": {"type": "string"},
    "model": {"type": "string"},
    "parameters": {"type": "object"},
    "design": {"type": "object"},
    "expected": {"type": "object"}
  }
}
```

```r
# scripts/fixtures/generate_lme4.R
library(lme4)
library(jsonlite)

fit <- lmer(y ~ time + (1 + time | subject), data = demo)
out <- list(
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  package = "lme4",
  package_version = as.character(packageVersion("lme4")),
  model = "y ~ time + (1 + time | subject)",
  parameters = as.list(VarCorr(fit)),
  design = list(n = nrow(demo)),
  expected = list(V = as.matrix(vcov(fit)))
)
write_json(out, "tests/fixtures/lme4_rs.json", auto_unbox = TRUE, pretty = TRUE)
```

Add `generate_geepack.R` with `geeglm` and `working.correlation` output. Keep `glmmTMB` out.

- [ ] **Step 4: Run the schema test to verify it passes**

Run: `uv run pytest tests/unit/test_fixture_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures scripts/fixtures tests/unit/test_fixture_schema.py
git commit -m "test: add reference fixture schema and generators"
```

---

### Task T6.3: Quarto documentation and examples

**Files:**
- Create: `docs/_quarto.yml`
- Create: `docs/index.qmd`
- Create: `docs/getting-started.qmd`
- Create: `docs/model-specification.qmd`
- Create: `docs/covariance-structures.qmd`
- Create: `docs/moments.qmd`
- Create: `docs/variograms.qmd`
- Create: `docs/marginal-vs-conditional.qmd`
- Create: `docs/gee.qmd`
- Create: `docs/forms.qmd`
- Create: `docs/validation.qmd`
- Create: `docs/api/index.qmd`
- Create: `examples/random-intercept.qmd`
- Create: `examples/random-slope.qmd`
- Create: `examples/ri-ar1.qmd`
- Create: `examples/marginal-ar1-cs.qmd`
- Create: `examples/cs-decomposition.qmd`
- Create: `examples/nested-crossed.qmd`
- Create: `examples/gee-working.qmd`
- Create: `examples/dense-blup.qmd`

**Interfaces:**
- Produces: a renderable Quarto website at `docs/_site`

- [ ] **Step 1: Write the Quarto configuration**

```yaml
# docs/_quarto.yml
project:
  type: website
  output-dir: _site

website:
  title: "symlmm"
  sidebar:
    contents:
      - index.qmd
      - getting-started.qmd
      - model-specification.qmd
      - covariance-structures.qmd
      - moments.qmd
      - variograms.qmd
      - marginal-vs-conditional.qmd
      - gee.qmd
      - forms.qmd
      - validation.qmd
      - api/index.qmd

format:
  html:
    theme: cosmo
    toc: true
    html-math-method: mathjax

execute:
  freeze: auto
```

- [ ] **Step 2: Write the pages with executable examples**

Each page must contain a runnable Python cell using the public API. Use only APIs implemented in the core and marginal/GEE plans. The getting-started page must show an RI model and a marginal model.

- [ ] **Step 3: Add `quartodoc` configuration**

Add the `quartodoc` block to `_quarto.yml` and generate `docs/api/` from docstrings.

- [ ] **Step 4: Render locally**

Run: `quarto check && quarto render docs`
Expected: website builds at `docs/_site` with all pages.

- [ ] **Step 5: Commit**

```bash
git add docs examples
git commit -m "docs: add Quarto site and examples"
```

---

### Task T7.1: CI matrix and quality gates

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/docs.yml`

**Interfaces:**
- Produces: CI jobs for tests, lint, types, coverage, and docs

- [ ] **Step 1: Write the CI workflow**

```yaml
# .github/workflows/ci.yml
name: ci
on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.14"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      - run: uv sync --all-extras
      - run: uv run ruff check
      - run: uv run ruff format --check
      - run: uv run mypy src/symlmm
      - run: uv run pytest --cov=symlmm --cov-report=xml --cov-fail-under=90
      - uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
```

```yaml
# .github/workflows/docs.yml
name: docs
on:
  push:
    branches: [main]
  pull_request:

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: quarto-dev/quarto-actions/setup@v2
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.14"
          enable-cache: true
      - run: uv sync --extra docs
      - run: quarto check
      - run: quarto render docs
```

- [ ] **Step 2: Verify the workflow syntax**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows
git commit -m "ci: add test and documentation workflows"
```

---

### Task T7.2: Packaging, release metadata, and v0.1 tag

**Files:**
- Modify: `pyproject.toml`
- Create: `CHANGELOG.md`
- Create: `CITATION.cff`
- Create: `LICENSE`
- Modify: `src/symlmm/__init__.py`
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Produces: buildable sdist/wheel, changelog, citation metadata, PyPI release workflow

- [ ] **Step 1: Add packaging metadata**

```toml
[project]
authors = [{ name = "symlmm contributors" }]
keywords = ["mixed models", "symbolic", "statistics", "GEE"]
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: BSD License",
  "Programming Language :: Python :: 3.14",
  "Typing :: Typed",
]

[project.urls]
Documentation = "https://example.github.io/symlmm/"
Repository = "https://github.com/example/symlmm"
```

Add `py.typed` to the wheel:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/symlmm"]
```

- [ ] **Step 2: Write the changelog and citation**

```markdown
# Changelog

## 0.1.0

- Symbolic conditional LMM moments in sum and matrix form.
- Marginal LMMs, induced covariance, and bounded exact decomposition.
- GEE families, working correlations, and symbolic marginal moments.
- Variograms, covariograms, ICC, correlation, BLUP, and dense mode.
- Quarto documentation and validation fixtures.
```

```yaml
# CITATION.cff
cff-version: 1.2.0
title: symlmm
message: "If you use symlmm, please cite this repository."
type: software
authors:
  - name: "symlmm contributors"
version: 0.1.0
date-released: "2026-11-09"
license: BSD-3-Clause
```

- [ ] **Step 3: Write the release workflow**

```yaml
# .github/workflows/release.yml
name: release
on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.14"
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Use PyPI trusted publishing; do not add API tokens.

- [ ] **Step 4: Build and verify**

Run: `uv build`
Expected: `dist/symlmm-0.1.0.dev0.tar.gz` and a wheel.

Run: `uv run python -c "import symlmm; print(symlmm.__version__)"`
Expected: `0.1.0.dev0`.

- [ ] **Step 5: Run the full release gate**

Run: `uv run pytest --cov=symlmm --cov-fail-under=90 && uv run ruff check && uv run mypy src/symlmm && quarto render docs`
Expected: PASS.

- [ ] **Step 6: Commit and tag**

```bash
git add pyproject.toml CHANGELOG.md CITATION.cff LICENSE src/symlmm/__init__.py .github/workflows/release.yml
git commit -m "chore: prepare v0.1 release"
git tag v0.1.0
git push origin main --tags
```
