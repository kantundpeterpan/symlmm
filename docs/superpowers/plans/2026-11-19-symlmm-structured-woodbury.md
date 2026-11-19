# Structured Covariance and Woodbury Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace opaque block covariance inversions with a factorized covariance layer, `V = R + Z G Z'`, and a general Woodbury reduction. The model derives design columns from `RandomEffects` and residual precision from `ResidualCovariance`; RI, RS, IID, AR1, and other kernels simplify without per-model hardcoding.

**Architecture:** A new `structured.py` owns design-column construction, residual precision lookup, and Woodbury formulas. `Block` and `BlockSet` use those helpers for precision, shrinkage, BLUP, and conditional covariance. Dense remains a numeric evaluator. The only covariance-family-specific code is a `precision()` override, not a model or query hardcode.

**Tech Stack:** Python 3.14, SymPy, pytest, mypy, ruff.

**Base:** `45ba986` on `main`

**Spec:** No separate spec. This plan is the design.

## Global Constraints

- Python 3.14; no new runtime dependencies.
- Public functions/classes have docstrings.
- Dense behavior must not change.
- Symbolic structured paths must not call `dense()`, `Block.rows()`, or `PanelGrid.row()`.
- No hardcoded RI, RS, or model-specific design in the new helpers.
- Design columns come from `RandomEffects.intercept` and `RandomEffects.slopes`.
- Residual precision comes from `ResidualCovariance.precision(block)` with an `Inverse(R_block)` fallback.
- CI:
  - `uv run pytest --cov=symlmm --cov-fail-under=90`
  - `uv run pytest -q -W error::DeprecationWarning`
  - `uv run mypy src/symlmm`
  - `uv run ruff check`
  - `uv run ruff format --check`
- Branch names: `feat/structured-core`, `feat/structured-blocksets`, `feat/structured-docs`.
- Wave branches merge into `feat/structured-woodbury`; one PR to `main`.

## Concurrency Map

```text
main
└── feat/structured-woodbury
    ├── feat/structured-core          (wave 0, blocks.py + covariances.py)
    ├── feat/structured-blocksets     (wave 1, blocksets.py)
    └── feat/structured-docs          (wave 1, docs + integration guard)
```

| Wave | Workstream | Branch | Files owned | Depends on | Runs with |
|---|---|---|---|---|---|
| 0 | Structured core | `feat/structured-core` | `src/symlmm/structured.py` (new), `src/symlmm/covariances.py`, `src/symlmm/blocks.py`, `tests/unit/test_structured.py` (new), `tests/unit/test_blocks.py` | none | nothing |
| 1 | BlockSet consumers | `feat/structured-blocksets` | `src/symlmm/blocksets.py`, `tests/unit/test_blocksets.py` | core | docs workstream |
| 1 | Docs and guards | `feat/structured-docs` | `docs/pedagogy/blackboard.md`, `CHANGELOG.md`, `tests/integration/test_structured_woodbury.py` | core | BlockSet workstream |
| 2 | Integration | `feat/structured-woodbury` | merge/full gate only | both | nothing |

Rules:

1. `structured-core` is the only writer of `blocks.py` and `covariances.py`.
2. Wave-1 branches are created from the post-core integration branch.
3. Every implementation workspace is created first, then its `.opencode/opencode.json` must exist, then the agent is launched with `modeId: "implementer"`.
4. Reviews use `modeId: "code-reviewer"`.
5. The BlockSet workstream must not edit `blocks.py`; request a core follow-up instead.
6. No parallel edits to the same file.

## File Structure

| File | Responsibility |
|---|---|
| `src/symlmm/structured.py` | Design columns, residual precision, Woodbury formulas, 1×1 scalar conversion |
| `src/symlmm/covariances.py` | `ResidualCovariance.precision(block)` protocol and IID override |
| `src/symlmm/blocks.py` | Structured design, precision, shrinkage, BLUP using `structured.py` |
| `src/symlmm/blocksets.py` | Structured block-set design/precision/shrinkage |
| `tests/unit/test_structured.py` | Woodbury/design/precision unit tests |
| `tests/unit/test_blocks.py` | Block-level structured integration |
| `tests/unit/test_blocksets.py` | BlockSet structured integration |
| `tests/integration/test_structured_woodbury.py` | No-dense and closed-form guards |
| `docs/pedagogy/blackboard.md` | Structured covariance examples |
| `CHANGELOG.md` | Public additions |

## Design Decisions

- Factor the covariance, never invert the full block matrix:

  \[
  V=R+ZGZ'
  \]

  \[
  V^{-1}
  =
  R^{-1}
  -
  R^{-1}Z
  \left(G^{-1}+Z'R^{-1}Z\right)^{-1}
  Z'R^{-1}
  \]

  \[
  GZ'V^{-1}
  =
  \left(G^{-1}+Z'R^{-1}Z\right)^{-1}
  Z'R^{-1}
  \]

- The reduced inverse is \(q\times q\), where \(q\) is the random-effect dimension. It is never \(n_i\times n_i\).
- `Z` is built generically:

  - intercept → `OneMatrix(n_i, 1)`
  - slope → one structured coordinate column
  - multiple slopes → concatenate columns with `BlockMatrix`

- `R^{-1}` is obtained from `ResidualCovariance.precision(block)`. The default returns `block.residual().inverse()`. `IID` returns `Identity(n_i) / var`. Additional covariance families can add a precision override without touching design/BLUP code.
- When the reduced inverse is \(1\times1\), convert it to a scalar with `expr[0, 0]` before multiplying by the row vector.
- This plan covers block-local and independent block-set covariance. Overlapping nested/crossed joint conditioning remains a follow-up; it should not be faked with dense rows.

## Task 1: Structured Core and Residual Precision

**Files:**
- Create: `src/symlmm/structured.py`
- Modify: `src/symlmm/covariances.py`
- Create: `tests/unit/test_structured.py`

**Interfaces:**
- Consumes: `sp.OneMatrix`, `sp.Identity`, `sp.BlockMatrix`, `Block.size()`
- Produces:
  - `woodbury_shrinkage(G, Z, R_inv) -> MatrixExpr`
  - `woodbury_covariance(G, Z, R_inv) -> MatrixExpr`
  - `woodbury_precision(G, Z, R_inv) -> MatrixExpr`
  - `ResidualCovariance.precision(block) -> MatrixExpr`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_structured.py`:

```python
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, RandomEffects
from symlmm.structured import woodbury_shrinkage


def _ri_model():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    subject = Group("subject")
    return LMM(
        fixed=Fixed(),
        random=[
            RandomEffects(
                group=subject,
                intercept=True,
                cov=Scalar(tau2),
            )
        ],
        residual=IID(var=sigma2),
    ), tau2, sigma2


def test_iid_precision_is_scaled_identity():
    model, _, sigma2 = _ri_model()
    i = sp.Symbol("i", integer=True)
    block = model.block("subject", i)

    R_inv = block.model.residual.precision(block)

    assert R_inv == sp.Identity(block.size()) / sigma2


def test_woodbury_ri_shrinkage_closed_form():
    model, tau2, sigma2 = _ri_model()
    i = sp.Symbol("i", integer=True)
    block = model.block("subject", i)
    term = block.term("subject")

    Z = sp.OneMatrix(block.size(), 1)
    G = term.random.cov.dense()
    R_inv = model.residual.precision(block)
    S = woodbury_shrinkage(G, Z, R_inv)

    expected = (tau2 / (sigma2 + block.size() * tau2)) * sp.OneMatrix(1, block.size())

    assert S.shape == (1, block.size())
    assert (S - expected).equals(sp.zeros(1, block.size()))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_structured.py -q
```

Expected: fail because `symlmm.structured` and `ResidualCovariance.precision` do not exist.

- [ ] **Step 3: Implement the protocol**

In `src/symlmm/covariances.py`, add to `ResidualCovariance`:

```python
    def precision(self, block: Any) -> Any:
        """Return R^-1 for a block; default is the formal inverse."""
        from .structured import inverse_matrix

        return inverse_matrix(block.residual())
```

Add to `IID`:

```python
    def precision(self, block: Any) -> Any:
        """Return the IID precision matrix."""
        return sp.Identity(block.size()) / self.var
```

`Any` is already imported in `covariances.py`; do not add a Block import at runtime.

- [ ] **Step 4: Implement `structured.py`**

Create `src/symlmm/structured.py`:

```python
"""Structured design columns, residual precision, and Woodbury reductions."""

from __future__ import annotations

from typing import Any

import sympy as sp  # type: ignore[import-untyped]


def as_scalar(expr: Any) -> Any:
    """Return the scalar inside a 1x1 matrix, otherwise `expr`."""
    if isinstance(expr, (sp.MatrixBase, sp.MatrixExpr)) and expr.shape == (1, 1):
        return expr[0, 0]
    return expr


def inverse_matrix(matrix: Any) -> Any:
    """Return an inverse for explicit matrices and matrix expressions."""
    if isinstance(matrix, sp.MatrixBase):
        return matrix.inv()
    return matrix.inverse()


def woodbury_reduced_inverse(G: Any, Z: Any, R_inv: Any) -> Any:
    """Return (G^-1 + Z' R^-1 Z)^-1 without an n x n inverse."""
    gram = Z.T * R_inv * Z
    if isinstance(gram, (sp.MatrixBase, sp.MatrixExpr)) and gram.shape == (1, 1):
        gram = gram[0, 0]
        return 1 / (as_scalar(G) + gram)
    return inverse_matrix(inverse_matrix(G) + gram)


def woodbury_shrinkage(G: Any, Z: Any, R_inv: Any) -> Any:
    """Return G Z' (R + Z G Z')^-1."""
    return woodbury_reduced_inverse(G, Z, R_inv) * Z.T * R_inv


def woodbury_covariance(G: Any, Z: Any, R_inv: Any) -> Any:
    """Return (G^-1 + Z' R^-1 Z)^-1."""
    return woodbury_reduced_inverse(G, Z, R_inv)


def woodbury_precision(G: Any, Z: Any, R_inv: Any) -> Any:
    """Return (R + Z G Z')^-1 using Woodbury."""
    reduced = woodbury_reduced_inverse(G, Z, R_inv)
    return R_inv - R_inv * Z * reduced * Z.T * R_inv
```

Add `as_scalar`/Woodbury exports only through the module; do not add them to the package root yet.

- [ ] **Step 5: Run the focused tests**

```bash
uv run pytest tests/unit/test_structured.py -q
```

Expected: pass, including the RI closed form.

- [ ] **Step 6: Commit**

```bash
git add src/symlmm/structured.py src/symlmm/covariances.py tests/unit/test_structured.py
git commit -m "feat: add structured covariance and Woodbury core"
```

## Task 2: Structured Block Design and Block Queries

**Files:**
- Modify: `src/symlmm/blocks.py`
- Test: `tests/unit/test_blocks.py`

**Interfaces:**
- Consumes: `block_design_columns`, `block_random_design`, `woodbury_*`, `ResidualCovariance.precision`
- Produces:
  - `Block.design_columns(term=None) -> tuple[Any, ...]`
  - `Block.structured_random_design(term=None) -> Any`
  - `Block.residual_precision() -> Any`
  - `Block.precision() -> Any`
  - `Block.shrinkage(term=None) -> Any` uses Woodbury
  - `Block.blup_covariance(term=None) -> Any` uses Woodbury

- [ ] **Step 1: Update old assertions and add the failing tests**

Replace the old `Block.shrinkage`/`Block.blup_covariance` assertions that
compare against `G * Z.T * V.inverse()` with either the new structured closed
form or dense equality. Those old tests assume the opaque `MatrixSymbol` path
and will no longer be the implementation contract.

Then add to `tests/unit/test_blocks.py`:

```python
def test_block_design_columns_for_ri_and_rs():
    ri_model, i = _ri_rs_symbolic_model()
    ri_block = ri_model.block("subject", i)
    ri_term = ri_block.term("subject")
    ri_columns = ri_block.design_columns(ri_term)

    assert len(ri_columns) == 1
    assert ri_columns[0] == sp.OneMatrix(ri_block.size(), 1)

    tau2 = sp.Symbol("tau2", positive=True)
    model = LMM(
        fixed=Fixed(),
        random=[
            RandomEffects(
                group=Group("subject"),
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Unstructured.from_sd_corr(
                    [sp.Symbol("s0", positive=True), sp.Symbol("s1", positive=True)],
                    sp.Symbol("rho_b", real=True),
                ),
            )
        ],
        residual=IID(var=sp.Symbol("sigma2", positive=True)),
        coordinate="time",
    )
    block = model.block("subject", i)
    columns = block.design_columns("subject")

    assert len(columns) == 2
    assert columns[0] == sp.OneMatrix(block.size(), 1)


def test_block_precision_matches_dense_inverse():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 1, 1, 1],
        time=[0, 1, 2, 0, 1, 2],
    )
    block = dense.block("subject", 0)

    assert block.precision() == dense.matrix_covariance()[0:3, 0:3].inv()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_blocks.py::test_block_design_columns_for_ri_and_rs tests/unit/test_blocks.py::test_block_precision_matches_dense_inverse -q
```

Expected: fail because the methods do not exist.

- [ ] **Step 3: Implement the design and precision methods**

In `blocks.py`, add:

```python
    def design_columns(self, term: Any = None) -> tuple[Any, ...]:
        """Return structured random-effect design columns for one term."""
        resolved = self.term(term)
        columns: list[Any] = []
        if resolved.random.intercept:
            columns.append(sp.OneMatrix(self.size(), 1))
        for slope in resolved.random.slopes:
            within = resolved.factor.name if slope.within else None
            if self.model._dense:
                values = [
                    self.model.coordinate(row, slope.coordinate.name, within=within)
                    for row in self.rows()
                ]
                columns.append(sp.Matrix(values))
            else:
                name = (
                    f"t_{self.factor.label}:{self.level}:"
                    f"{resolved.name}:{slope.coordinate.name}"
                )
                columns.append(sp.MatrixSymbol(name, self.size(), 1))
        return tuple(columns)

    def structured_random_design(self, term: Any = None) -> Any:
        """Return the structured random-effect design block."""
        return sp.BlockMatrix([list(self.design_columns(term))])

    def residual_precision(self) -> Any:
        """Return R_A^-1 for this block."""
        return self.model.residual.precision(self)

    def precision(self) -> Any:
        """Return V_A^-1 using the factorized covariance."""
        term = self.term()
        G = term.random.cov.dense()
        Z = self.structured_random_design(term)
        return woodbury_precision(G, Z, self.residual_precision())
```

Import at the top:

```python
from .structured import woodbury_covariance, woodbury_precision, woodbury_shrinkage
```

Replace `shrinkage` and `blup_covariance`:

```python
    def shrinkage(self, term: str | None = None) -> Any:
        """Return G_t Z_t' V_A^-1 using Woodbury."""
        resolved = self.term(term)
        G = resolved.random.cov.dense()
        Z = self.structured_random_design(resolved)
        return woodbury_shrinkage(G, Z, self.residual_precision())

    def blup_covariance(self, term: str | None = None) -> Any:
        """Return Cov(b_A | Y_A) using Woodbury."""
        resolved = self.term(term)
        return woodbury_covariance(
            resolved.random.cov.dense(),
            self.structured_random_design(resolved),
            self.residual_precision(),
        )
```

Keep `blup_mean` as `self.shrinkage(resolved.name) * (self.response() - self.mean())`.

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/unit/test_blocks.py tests/unit/test_structured.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/blocks.py tests/unit/test_blocks.py
git commit -m "feat: use structured Woodbury in block queries"
```

## Task 3: Structured BlockSet Queries

**Files:**
- Modify: `src/symlmm/blocksets.py`
- Test: `tests/unit/test_blocksets.py`

**Interfaces:**
- Consumes: `Block.design_columns`, `Block.structured_random_design`, `Block.residual_precision`, `woodbury_*`
- Produces:
  - `BlockSet.structured_random_design(term=None) -> Any`
  - `BlockSet.residual_precision() -> Any`
  - `BlockSet.precision() -> Any`
  - `BlockSet.shrinkage(term=None) -> Any`
  - `BlockSet.blup_covariance(term=None) -> Any`

- [ ] **Step 1: Update old assertions and add the failing tests**

Replace old `BlockSet.shrinkage`/`blup_covariance` assertions that compare
against `V.inverse()` with structured closed forms or dense equality. The old
assertions encode the opaque `MatrixSymbol` contract.

Then add to `tests/unit/test_blocksets.py`:

```python
def test_blockset_ri_shrinkage_closed_form():
    model, i = _ri_rs_symbolic_model()
    j = sp.Symbol("j", integer=True)
    blocks = model.block_set(
        model.block("subject", i),
        model.block("subject", j),
    )
    tau2 = blocks.blocks[0].term("subject").random.cov.dense()[0, 0]
    sigma2 = model.residual.var

    S = blocks.shrinkage("subject")

    assert S.shape == (2, blocks.size())
    assert S.has(sp.OneMatrix(1, blocks.blocks[0].size()))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_blocksets.py::test_blockset_ri_shrinkage_closed_form -q
```

Expected: fail because the method still uses `V.inverse()` and missing structured helpers.

- [ ] **Step 3: Implement structured BlockSet methods**

Add to `BlockSet`:

```python
    def structured_random_design(self, term: str | None = None) -> Any:
        """Return the block-diagonal structured random design."""
        if term is None and len({block.factor for block in self.blocks}) > 1:
            raise ModelError("random_design requires a term for mixed-factor block sets")
        return _assemble(
            self.model,
            sp.BlockDiagMatrix(
                *[block.structured_random_design(term=term) for block in self.blocks]
            ),
        )

    def residual_precision(self) -> Any:
        """Return the block-diagonal residual precision."""
        return _assemble(
            self.model,
            sp.BlockDiagMatrix(
                *[block.residual_precision() for block in self.blocks]
            ),
        )

    def precision(self) -> Any:
        """Return the joint V^-1 using Woodbury."""
        resolved = self._term()
        G = sp.BlockDiagMatrix(
            *[resolved.random.cov.dense() for _ in self.blocks]
        ).as_explicit()
        Z = self.structured_random_design(resolved.name)
        return woodbury_precision(G, Z, self.residual_precision())

    def shrinkage(self, term: str | None = None) -> Any:
        """Return the block-set shrinkage operator using Woodbury."""
        resolved = self._term(term)
        G = sp.BlockDiagMatrix(
            *[resolved.random.cov.dense() for _ in self.blocks]
        ).as_explicit()
        Z = self.structured_random_design(resolved.name)
        return woodbury_shrinkage(G, Z, self.residual_precision())

    def blup_covariance(self, term: str | None = None) -> Any:
        """Return the joint conditional covariance using Woodbury."""
        resolved = self._term(term)
        G = sp.BlockDiagMatrix(
            *[resolved.random.cov.dense() for _ in self.blocks]
        ).as_explicit()
        Z = self.structured_random_design(resolved.name)
        return woodbury_covariance(G, Z, self.residual_precision())
```

Keep `blup_mean` as `self.shrinkage(resolved.name) * (self.response() - self.design() * self.model.mean?)`; use `self.mean()` to avoid empty fixed effects:

```python
    def blup_mean(self, term: str | None = None) -> Any:
        """Return the joint BLUP for this set."""
        resolved = self._term(term)
        return self.shrinkage(resolved.name) * (self.response() - self.mean())
```

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/unit/test_blocksets.py tests/unit/test_structured.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/blocksets.py tests/unit/test_blocksets.py
git commit -m "feat: use structured Woodbury in block-set queries"
```

## Task 4: General Precision Consumers

**Files:**
- Modify: `src/symlmm/blocks.py`
- Modify: `src/symlmm/blocksets.py`
- Test: `tests/unit/test_blocks.py`, `tests/unit/test_blocksets.py`

**Interfaces:**
- Produces: `Block.precision()`, `BlockSet.precision()` for any query needing \(V^{-1}\)
- Consumes: `woodbury_precision`

- [ ] **Step 1: Add precision tests**

Add:

```python
def test_block_precision_has_no_n_by_n_inverse_symbol():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    precision = block.precision()

    assert precision.shape == (block.size(), block.size())
    assert not any(
        isinstance(symbol, sp.Inverse) and symbol.shape == (block.size(), block.size())
        for symbol in precision.atoms(sp.Inverse)
    )
```

For `BlockSet`:

```python
def test_blockset_precision_has_no_n_by_n_inverse_symbol():
    model, i = _ri_rs_symbolic_model()
    j = sp.Symbol("j", integer=True)
    blocks = model.block_set(
        model.block("subject", i),
        model.block("subject", j),
    )

    precision = blocks.precision()

    assert precision.shape == (blocks.size(), blocks.size())
    assert not any(
        isinstance(symbol, sp.Inverse) and symbol.shape == (blocks.size(), blocks.size())
        for symbol in precision.atoms(sp.Inverse)
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_blocks.py::test_block_precision_has_no_n_by_n_inverse_symbol tests/unit/test_blocksets.py::test_blockset_precision_has_no_n_by_n_inverse_symbol -q
```

Expected: pass after Task 2/3; these tests lock the no-\(n\times n\)-inverse property. If they fail, fix the structured path, not the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_blocks.py tests/unit/test_blocksets.py
git commit -m "test: guard structured precision paths"
```

## Task 5: Docs, Closed-Form Guard, and Full Gate

**Files:**
- Create: `tests/integration/test_structured_woodbury.py`
- Modify: `docs/pedagogy/blackboard.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: all prior tasks
- Produces: closed-form RI guard and documentation

- [ ] **Step 1: Add the integration guard**

Create `tests/integration/test_structured_woodbury.py`:

```python
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, RandomEffects


def test_ri_iid_shrinkage_closed_form_without_dense():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    i = sp.Symbol("i", integer=True)

    model = LMM(
        fixed=Fixed(),
        random=[
            RandomEffects(
                group=Group("subject"),
                intercept=True,
                cov=Scalar(tau2),
            )
        ],
        residual=IID(var=sigma2),
    )
    block = model.block("subject", i)
    S = block.shrinkage("subject")
    expected = (tau2 / (sigma2 + block.size() * tau2)) * sp.OneMatrix(1, block.size())

    assert (S - expected).equals(sp.zeros(1, block.size()))
    assert block.precision().shape == (block.size(), block.size())
```

- [ ] **Step 2: Run the guard**

```bash
uv run pytest tests/integration/test_structured_woodbury.py -q
```

Expected: pass.

- [ ] **Step 3: Document the structured layer**

Add to `docs/pedagogy/blackboard.md`:

```markdown
### Structured covariance

Block moments use the factorized covariance

\[
V_A = R_A + Z_A G Z_A',
\]

and Woodbury for any inverse:

\[
V_A^{-1}
=
R_A^{-1}
-
R_A^{-1}Z_A
\left(G^{-1}+Z_A'R_A^{-1}Z_A\right)^{-1}
Z_A'R_A^{-1}.
\]

`Z_A` is built from the random-effect specification and `R_A^{-1}` comes from
the residual covariance. RI/IID therefore reduces automatically:

```python
A.shrinkage("subject")
# tau2/(sigma2 + n_subject:i*tau2) * 1'
```

without a dense \(n_i\times n_i\) inverse.
```

- [ ] **Step 4: Changelog**

Add:

```markdown
### Added

- Structured block designs and residual precision for factorized covariance.
- Woodbury-based `Block.precision`, `BlockSet.precision`, shrinkage, and BLUP
  moments without dense \(n\times n\) inverses.
```

- [ ] **Step 5: Full gate**

```bash
uv run pytest --cov=symlmm --cov-fail-under=90
uv run pytest -q -W error::DeprecationWarning
uv run mypy src/symlmm
uv run ruff check
uv run ruff format --check
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_structured_woodbury.py docs/pedagogy/blackboard.md CHANGELOG.md
git commit -m "docs: document structured covariance and Woodbury"
```

## Acceptance Criteria

- `Block.shrinkage` for RI/IID returns the closed form from abstract \(n_i\) with no `dense()` call.
- RI/RS designs are built from `RandomEffects`; no RI-specific branch exists in `blocks.py`.
- `ResidualCovariance.precision` has an IID override and a formal-inverse fallback; adding a covariance family does not require touching BLUP code.
- `Block.precision` and `BlockSet.precision` avoid \(n_i\times n_i\) inverse expressions.
- Random slopes keep a \(q\times q\) reduced inverse.
- Dense equality tests still pass.
- Full CI gate passes.

## Risks

- **SymPy structured simplification:** `OneMatrix`/`Identity` do not automatically apply Woodbury. The explicit `woodbury_*` helpers are the algebra layer, not `simplify()`.
- **1×1 conversion:** `Z'R^{-1}Z` returns a 1×1 matrix expression; convert with `[0, 0]` before scalar arithmetic.
- **Coordinate columns:** symbolic slope columns are `MatrixSymbol`s; Gram matrices stay symbolic. Do not force expansion; keep the reduced \(q\times q\) inverse.
- **Residual precision:** only IID is implemented initially. Other families fall back to `Inverse(R)` until they add a `precision` override.
- **BlockSet scope:** independent block sets only. Nested/crossed joint conditioning remains a follow-up.
- **Concurrency:** core is the only writer of `blocks.py`/`covariances.py`; BlockSet and docs start only after core merges.
