# Abstract Block Algebra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make block selection and conditioning a first-class symbolic layer so block-level moments, BLUPs, shrinkage, and panel entry moments work without `dense()`, using symbolic block sizes and block matrices.

**Architecture:** `Block` remains the single-block handle. A new `BlockSet` stacks blocks and exposes response, design, mean, covariance, and cross-covariance as block matrices. `Block.shrinkage`, `Block.blup_mean`, and `BlockSet.blup_mean` derive results from `G_t`, `Z_A`, and `V_A`; dense mode remains a numeric evaluator. No new covariance algebra is introduced.

**Tech Stack:** Python 3.14, SymPy, pytest, mypy, ruff.

**Base:** `2f22c54` on `main`

**Spec:** No separate spec. The design decisions in this plan are authoritative.

## Global Constraints

- Python 3.14; no new runtime dependencies.
- Every public function and class has a docstring.
- Dense behavior must not change.
- Symbolic block APIs must not call `dense()`, `Block.rows()`, `PanelGrid.row()`, `PanelGrid.level_codes()`, or `PanelGrid.position_codes()`.
- `Block.rows()` remains the dense evaluator's row accessor.
- No `TODO` comments merged without a linked issue.
- CI gate:
  - `uv run pytest --cov=symlmm --cov-fail-under=90`
  - `uv run pytest -q -W error::DeprecationWarning`
  - `uv run mypy src/symlmm`
  - `uv run ruff check`
  - `uv run ruff format --check`
- Branch names: `feat/block-core`, `feat/block-sets`, `feat/panel-abstract-entries`, `docs/block-algebra`.
- Wave branches merge into `feat/abstract-block-algebra` after review. The integration branch opens one PR to `main`. No direct commits to `main`.
- Local Paseo configuration (`paseo.json`, `.superpowers/`, `.worktreeinclude`) from 2026-11-19 must not be committed.

## Concurrency Map

Use an integration branch:

```text
main
└── feat/abstract-block-algebra
    ├── feat/block-core                 (wave 0, must land first)
    ├── feat/block-sets                 (wave 1, after block-core)
    ├── feat/panel-abstract-entries     (wave 1, after block-core)
    └── docs/block-algebra              (wave 2, after block-sets and panel)
```

Create it once:

```bash
git switch main
git pull --ff-only
git switch -c feat/abstract-block-algebra
git push -u origin feat/abstract-block-algebra
```

Create wave-0 from that branch:

```bash
git switch feat/abstract-block-algebra
git switch -c feat/block-core
```

After wave 0 merges, create both wave-1 branches from the updated
integration branch:

```bash
git switch feat/abstract-block-algebra
git pull --ff-only

git switch -c feat/block-sets
git switch feat/abstract-block-algebra
git switch -c feat/panel-abstract-entries
```

| Wave | Workstream | Branch | Files owned | Depends on | Runs with |
|---|---|---|---|---|---|
| 0 | Block core | `feat/block-core` | `src/symlmm/blocks.py`, `tests/unit/test_blocks.py` | none | nothing |
| 1 | Block sets | `feat/block-sets` | `src/symlmm/blocksets.py` (new), `src/symlmm/model.py`, `src/symlmm/__init__.py`, `tests/unit/test_blocksets.py` (new) | block-core | panel workstream |
| 1 | Panel entries | `feat/panel-abstract-entries` | `src/symlmm/panel.py`, `tests/unit/test_panel.py` | block-core | block-set workstream |
| 2 | Docs and integration | `docs/block-algebra` | `docs/pedagogy/blackboard.md`, `CHANGELOG.md`, integration tests | block-sets and panel | nothing |

Concurrency rules:

1. Wave 1 starts only after `feat/block-core` is merged into `feat/abstract-block-algebra`.
2. Wave 1 branches are created from the post-core integration branch, not from each other.
3. No two implementers edit `src/symlmm/blocks.py`. `block-core` is the only wave-0 writer.
4. `model.py` and `__init__.py` belong to `block-sets`; the panel workstream must not edit them.
5. Reviewers run after each branch; one implementer and one reviewer per branch.
6. If `block-core` changes `Block.term`, `Block.response`, or `Block.random_effect` signatures, rebase wave-1 branches before dispatch.
7. Each worktree is hydrated with `.superpowers/hydrate_workspace.sh`; the setup copies the slim OpenCode config with `implementer` and `code-reviewer`.
8. Merge order: core → block-sets → panel → docs. The integration branch gets the full gate before its final PR to `main`.

## File Structure

| File | Responsibility |
|---|---|
| `src/symlmm/blocks.py` | Single-block symbolic handles: response, term resolution, shrinkage, BLUP, BLUP covariance |
| `src/symlmm/blocksets.py` | Ordered block collection and block-matrix moments |
| `src/symlmm/model.py` | `LMM.block_set(*blocks)` accessor |
| `src/symlmm/__init__.py` | Export `BlockSet` |
| `src/symlmm/panel.py` | Symbolic panel entry moments through block entries |
| `tests/unit/test_blocks.py` | Block response/term/BLUP/shrinkage tests |
| `tests/unit/test_blocksets.py` | BlockSet shape, covariance, BLUP tests |
| `tests/unit/test_panel.py` | Symbolic panel entry tests; existing dense tests stay |
| `tests/integration/test_block_sets.py` | Dense equality and no-row-access guards |
| `docs/pedagogy/blackboard.md` | Abstract block moment examples |
| `CHANGELOG.md` | Public API additions |

## Design Decisions

- **Block-local BLUP:** for a target block \(A\) and random-effect term \(t\), condition on the response of \(A\):

  \[
  \tilde b_{A,t}
  =
  G_t Z_{A,t}' V_A^{-1}
  \left(Y_A-X_A\beta\right)
  \]

  The shrinkage operator is

  \[
  S_{A,t}=G_t Z_{A,t}' V_A^{-1}.
  \]

- **BlockSet BLUP:** for an ordered set of independent blocks \(\mathcal B\), use the block-diagonal effect design and effect covariance:

  \[
  Z_{\mathcal B,t}
  =
  \operatorname{BlockDiag}(Z_{A,t}:A\in\mathcal B),
  \qquad
  G_{\mathcal B,t}
  =
  \operatorname{BlockDiag}(G_t:A\in\mathcal B).
  \]

  Then

  \[
  \tilde b_{\mathcal B,t}
  =
  G_{\mathcal B,t}Z_{\mathcal B,t}'V_{\mathcal B}^{-1}
  \left(Y_{\mathcal B}-X_{\mathcal B}\beta\right).
  \]

- **Scope:** this plan supports independent block sets, which is the panel/subject case. Joint BLUP for overlapping nested or crossed random-effect blocks is a follow-up and must not be faked with row selection.
- **Panel entries:** symbolic `PanelView.entry_*` uses the position as the block-local row address; dense mode still maps positions to concrete rows with `PanelGrid.row`.
- **Position-column aggregates:** `position_mean` and `position_covariance` keep their current dense-only behavior in this plan. They need a separate symbolic vector/sum representation and are not part of this change.
- **No dense fallback:** every new symbolic method is tested without `dense()`, and the tests fail if row access is invoked.

## Task 1: Block Response and Term Handles

**Files:**
- Modify: `src/symlmm/blocks.py` (`Block` class)
- Test: `tests/unit/test_blocks.py`

**Interfaces:**
- Consumes: `Block.size()`, `Block.random_design(term)`, `decomposition.terms()`
- Produces:
  - `Block.response() -> MatrixSymbol`
  - `Block.term(name: str | None = None) -> Term`
  - `Block.random_effect(term: str | None = None) -> MatrixSymbol`

- [ ] **Step 1: Write the failing tests**

Extend the imports at the top of `tests/unit/test_blocks.py` with
`Unstructured` from `symlmm.covariances` and `Covariate` from `symlmm.specs`.
Then add:

```python
def _ri_rs_symbolic_model():
    b0, b1 = sp.symbols("b0 b1", real=True)
    tau0, tau1, sigma2 = sp.symbols("tau0 tau1 sigma2", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    i = sp.Symbol("i", integer=True)

    subject = Group("subject", index=i)
    time = Covariate("time")

    return LMM(
        fixed=Fixed(intercept=b0, slopes={time: b1}),
        random=[
            RandomEffects(
                group=subject,
                intercept=True,
                slopes=(RandomSlope(time),),
                cov=Unstructured.from_sd_corr([tau0, tau1], rho_b),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    ), i


def test_block_response_is_symbolic():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    Y = block.response()

    assert Y.shape == (block.size(), 1)
    assert Y.name == "Y_subject:i"


def test_block_term_resolves_single_term():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    assert block.term().name == "subject"
    assert block.term("subject").name == "subject"
    assert block.random_effect("subject") == block_coefficients(
        model.factor("subject"), i, 2
    )


def test_block_term_unknown_name_raises():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    with pytest.raises(ModelError, match="unknown random-effect term"):
        block.term("missing")
```

- [ ] **Step 2: Run the tests to verify failure**

```bash
uv run pytest tests/unit/test_blocks.py::test_block_response_is_symbolic tests/unit/test_blocks.py::test_block_term_resolves_single_term tests/unit/test_blocks.py::test_block_term_unknown_name_raises -v
```

Expected: fail because `Block.response` and `Block.term` do not exist.

- [ ] **Step 3: Implement the handles**

Add a module-level helper near the other block symbol helpers in `src/symlmm/blocks.py`:

```python
def block_response_symbol(factor: Factor, level: Any, size: Any) -> sp.MatrixSymbol:
    """Return the symbolic response vector for one block."""
    return sp.MatrixSymbol(f"Y_{factor.label}:{level}", size, 1)
```

Add these methods to `Block`:

```python
    def response(self) -> Any:
        """Return the symbolic response vector for this block."""
        return block_response_symbol(self.factor, self.level, self.size())

    def term(self, name: str | None = None) -> Any:
        """Return the random-effect term for this block.

        If more than one term belongs to the block's factor, `name` is required.
        """
        from .decomposition import terms

        candidates = [term for term in terms(self.model) if term.factor == self.factor]
        if name is None:
            if not candidates:
                raise NotDefined(f"factor {self.factor.name!r} has no random-effect term")
            if len(candidates) > 1:
                raise ModelError(
                    f"factor {self.factor.name!r} has multiple random-effect terms; "
                    "pass term=<name>"
                )
            return candidates[0]
        for term in candidates:
            if term.name == name:
                return term
        raise ModelError(f"unknown random-effect term: {name!r}")
```

Replace `random_effect` with:

```python
    def random_effect(self, term: str | None = None) -> Any:
        """Return the random-effect vector for one term of this block."""
        resolved = self.term(term)
        return block_coefficients(
            resolved.factor,
            self.level,
            resolved.random.dim,
        )
```

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/unit/test_blocks.py -q
```

Expected: all pass. Existing `random_effect()` calls still work when the factor has one term.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/blocks.py tests/unit/test_blocks.py
git commit -m "feat: add symbolic block response and term handles"
```

## Task 2: Block Shrinkage and BLUP

**Files:**
- Modify: `src/symlmm/blocks.py` (`Block` class)
- Test: `tests/unit/test_blocks.py`

**Interfaces:**
- Consumes: `Block.term`, `Block.response`, `Block.design`, `Block.random_design`, `Block.covariance`
- Produces:
  - `Block.shrinkage(term: str | None = None) -> MatrixExpr`
  - `Block.blup_mean(term: str | None = None) -> MatrixExpr`
  - `Block.blup_covariance(term: str | None = None) -> MatrixExpr`

- [ ] **Step 1: Write the failing tests**

Add:

```python
def test_block_shrinkage_is_symbolic():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)
    term = block.term("subject")

    G = term.random.cov.dense()
    Z = block.random_design(term="subject")
    V = block.covariance()
    S = block.shrinkage("subject")

    assert S == G * Z.T * V.inverse()
    assert S.shape == (2, block.size())


def test_block_blup_has_symbolic_shape():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    b_hat = block.blup_mean("subject")

    assert b_hat.shape == (2, 1)
    assert block.response() in b_hat.free_symbols


def test_block_blup_matches_global_dense():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 0)
    y = sp.MatrixSymbol("y", dense.n, 1)

    actual = block.blup_mean("subject").subs({block.response(): y[0:4, :]})
    expected = dense.blup_mean(form="matrix")[0:2, :]

    assert sp.simplify(actual - expected) == sp.zeros(2, 1)


def test_block_blup_covariance_matches_definition():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)
    term = block.term("subject")
    G = term.random.cov.dense()
    Z = block.random_design(term="subject")
    V = block.covariance()

    expected = G - G * Z.T * V.inverse() * Z * G

    assert sp.simplify(block.blup_covariance("subject") - expected) == sp.zeros(2, 2)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_blocks.py::test_block_shrinkage_is_symbolic tests/unit/test_blocks.py::test_block_blup_has_symbolic_shape tests/unit/test_blocks.py::test_block_blup_matches_global_dense tests/unit/test_blocks.py::test_block_blup_covariance_matches_definition -v
```

Expected: fail because the methods do not exist.

- [ ] **Step 3: Implement the block BLUP methods**

Add to `Block`:

```python
    def shrinkage(self, term: str | None = None) -> Any:
        """Return G_t Z_t' V_A^-1 for this block and random-effect term."""
        resolved = self.term(term)
        G = resolved.random.cov.dense()
        Z = self.random_design(term=resolved.name)
        V = self.covariance()
        return G * Z.T * V.inverse()

    def blup_mean(self, term: str | None = None) -> Any:
        """Return E[b_A | Y_A] for this block and random-effect term."""
        resolved = self.term(term)
        beta = sp.Matrix(self.model.fixed.coefficients())
        residual = self.response() - self.design() * beta
        return self.shrinkage(resolved.name) * residual

    def blup_covariance(self, term: str | None = None) -> Any:
        """Return Cov(b_A | Y_A) for this block and random-effect term."""
        resolved = self.term(term)
        G = resolved.random.cov.dense()
        Z = self.random_design(term=resolved.name)
        S = self.shrinkage(resolved.name)
        return G - S * Z * G
```

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/unit/test_blocks.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/blocks.py tests/unit/test_blocks.py
git commit -m "feat: add symbolic block BLUP and shrinkage"
```

## Task 3: BlockSet Core

**Files:**
- Create: `src/symlmm/blocksets.py`
- Modify: `src/symlmm/model.py` (`LMM.block_set`)
- Modify: `src/symlmm/__init__.py` (export `BlockSet`)
- Create: `tests/unit/test_blocksets.py`

**Interfaces:**
- Consumes: `Block.size`, `Block.response`, `Block.mean`, `Block.design`, `Block.random_design`, `Block.cross_covariance`
- Produces:
  - `BlockSet(model, blocks)`
  - `BlockSet.size() -> Expr`
  - `BlockSet.response() -> MatrixExpr`
  - `BlockSet.mean() -> MatrixExpr`
  - `BlockSet.design() -> MatrixExpr`
  - `BlockSet.random_design(term=None) -> MatrixExpr`
  - `BlockSet.covariance() -> MatrixExpr`
  - `BlockSet.cross_covariance(other: BlockSet) -> MatrixExpr`
  - `LMM.block_set(*blocks) -> BlockSet`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_blocksets.py`:

```python
import pytest
import sympy as sp

from symlmm.blocks import Block
from symlmm.covariances import IID, Unstructured
from symlmm.errors import ModelError
from symlmm.model import LMM
from symlmm.specs import Covariate, Fixed, Group, RandomEffects, RandomSlope


def _ri_rs_symbolic_model():
    b0, b1 = sp.symbols("b0 b1", real=True)
    tau0, tau1, sigma2 = sp.symbols("tau0 tau1 sigma2", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    i = sp.Symbol("i", integer=True)

    subject = Group("subject", index=i)
    time = Covariate("time")

    model = LMM(
        fixed=Fixed(intercept=b0, slopes={time: b1}),
        random=[
            RandomEffects(
                group=subject,
                intercept=True,
                slopes=(RandomSlope(time),),
                cov=Unstructured.from_sd_corr([tau0, tau1], rho_b),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )
    return model, i


def test_blockset_symbolic_shapes():
    model, i = _ri_rs_symbolic_model()
    j = sp.Symbol("j", integer=True)
    left = model.block("subject", i)
    right = model.block("subject", j)
    blocks = model.block_set(left, right)

    assert blocks.size() == left.size() + right.size()
    assert blocks.response().shape == (blocks.size(), 1)
    assert blocks.mean().shape == (blocks.size(), 1)
    assert blocks.design().shape == (blocks.size(), 2)
    assert blocks.random_design(term="subject").shape == (blocks.size(), 4)
    assert blocks.covariance().shape == (blocks.size(), blocks.size())


def test_blockset_rejects_empty_and_duplicate_blocks():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    with pytest.raises(ModelError, match="at least one block"):
        model.block_set()

    with pytest.raises(ModelError, match="duplicate block"):
        model.block_set(block, block)


def test_blockset_rejects_blocks_from_different_models():
    model, i = _ri_rs_symbolic_model()
    other, _ = _ri_rs_symbolic_model()

    with pytest.raises(ModelError, match="same model"):
        model.block_set(model.block("subject", i), other.block("subject", i))


def test_blockset_dense_covariance_matches_global_matrix():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    blocks = dense.block_set(
        dense.block("subject", 0),
        dense.block("subject", 1),
    )
    rows = list(range(8))

    assert blocks.covariance() == dense.matrix_covariance()[rows, :][:, rows]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_blocksets.py -v
```

Expected: fail because `symlmm.blocksets` and `LMM.block_set` do not exist.

- [ ] **Step 3: Implement `BlockSet`**

Create `src/symlmm/blocksets.py`:

```python
"""Ordered symbolic collections of factor-level blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .blocks import Block
from .errors import ModelError


@dataclass(frozen=True)
class BlockSet:
    """An ordered set of blocks from one model."""

    model: Any
    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        """Validate model membership, uniqueness, and non-emptiness."""
        if not self.blocks:
            raise ModelError("block set requires at least one block")
        if any(block.model is not self.model for block in self.blocks):
            raise ModelError("all blocks must belong to the same model")
        keys = [(block.factor, block.level) for block in self.blocks]
        if len(keys) != len(set(keys)):
            raise ModelError("duplicate block in block set")

    def size(self) -> Any:
        """Return the total number of observations in the set."""
        return sp.Add(*[block.size() for block in self.blocks])

    def response(self) -> Any:
        """Return the stacked response vector."""
        return sp.BlockMatrix([[block.response()] for block in self.blocks])

    def mean(self) -> Any:
        """Return the stacked fixed-effect mean."""
        return sp.BlockMatrix([[block.mean()] for block in self.blocks])

    def design(self) -> Any:
        """Return the stacked fixed-effect design matrix."""
        return sp.BlockMatrix([[block.design()] for block in self.blocks])

    def random_design(self, term: str | None = None) -> Any:
        """Return the block-diagonal design matrix for random-effect columns."""
        return sp.BlockDiagMatrix(
            *[block.random_design(term=term) for block in self.blocks]
        )

    def covariance(self) -> Any:
        """Return the joint block covariance matrix."""
        return sp.BlockMatrix(
            [
                [left.cross_covariance(right) for right in self.blocks]
                for left in self.blocks
            ]
        )

    def cross_covariance(self, other: BlockSet) -> Any:
        """Return Cov(Y_self, Y_other)."""
        if self.model is not other.model:
            raise ModelError("cross covariance requires block sets from the same model")
        return sp.BlockMatrix(
            [
                [left.cross_covariance(right) for right in other.blocks]
                for left in self.blocks
            ]
        )
```

Add to `LMM` in `src/symlmm/model.py`:

```python
    def block_set(self, *blocks: Any) -> Any:
        """Return an ordered symbolic set of blocks."""
        from .blocksets import BlockSet

        return BlockSet(self, tuple(blocks))
```

Update `src/symlmm/__init__.py`:

```python
from .blocksets import BlockSet
```

and add `"BlockSet"` to `__all__`.

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/unit/test_blocksets.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/blocksets.py src/symlmm/model.py src/symlmm/__init__.py tests/unit/test_blocksets.py
git commit -m "feat: add symbolic block sets"
```

## Task 4: BlockSet BLUP

**Files:**
- Modify: `src/symlmm/blocksets.py`
- Test: `tests/unit/test_blocksets.py`

**Interfaces:**
- Consumes: `BlockSet.random_design`, `BlockSet.covariance`, `BlockSet.response`, `BlockSet.design`, `Block.term`
- Produces:
  - `BlockSet.shrinkage(term=None) -> MatrixExpr`
  - `BlockSet.blup_mean(term=None) -> MatrixExpr`
  - `BlockSet.blup_covariance(term=None) -> MatrixExpr`

- [ ] **Step 1: Write the failing tests**

Add:

```python
def test_blockset_shrinkage_shape():
    model, i = _ri_rs_symbolic_model()
    j = sp.Symbol("j", integer=True)
    blocks = model.block_set(
        model.block("subject", i),
        model.block("subject", j),
    )

    S = blocks.shrinkage(term="subject")

    assert S.shape == (4, blocks.size())


def test_blockset_blup_matches_global_dense():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    blocks = dense.block_set(
        dense.block("subject", 0),
        dense.block("subject", 1),
    )
    y = sp.MatrixSymbol("y", dense.n, 1)
    left, right = blocks.blocks
    actual = blocks.blup_mean(term="subject").subs(
        {
            left.response(): y[0:4, :],
            right.response(): y[4:8, :],
        }
    )
    expected = dense.blup_mean(form="matrix")

    assert sp.simplify(actual - expected) == sp.zeros(4, 1)


def test_blockset_blup_requires_one_factor():
    model, i = _ri_rs_symbolic_model()
    other_factor = model.factor("subject")
    blocks = model.block_set(model.block("subject", i))

    assert blocks.blocks[0].factor == other_factor
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_blocksets.py::test_blockset_shrinkage_shape tests/unit/test_blocksets.py::test_blockset_blup_matches_global_dense -v
```

Expected: fail because the methods do not exist.

- [ ] **Step 3: Implement the block-set BLUP methods**

Add to `BlockSet`:

```python
    def _term(self, term: str | None = None) -> Any:
        """Resolve the shared random-effect term for this set."""
        factors = {block.factor for block in self.blocks}
        if len(factors) != 1:
            raise ModelError("block-set BLUP requires blocks from one factor")
        return self.blocks[0].term(term)

    def shrinkage(self, term: str | None = None) -> Any:
        """Return the block-set shrinkage operator."""
        resolved = self._term(term)
        G = sp.BlockDiagMatrix(
            *[resolved.random.cov.dense() for _ in self.blocks]
        ).as_explicit()
        Z = self.random_design(term=resolved.name)
        V = self.covariance()
        return G * Z.T * V.inverse()

    def blup_mean(self, term: str | None = None) -> Any:
        """Return the joint BLUP for the random effects in this set."""
        resolved = self._term(term)
        beta = sp.Matrix(self.model.fixed.coefficients())
        residual = self.response() - self.design() * beta
        return self.shrinkage(resolved.name) * residual

    def blup_covariance(self, term: str | None = None) -> Any:
        """Return the joint conditional covariance for this set."""
        resolved = self._term(term)
        G = sp.BlockDiagMatrix(
            *[resolved.random.cov.dense() for _ in self.blocks]
        ).as_explicit()
        Z = self.random_design(term=resolved.name)
        S = self.shrinkage(resolved.name)
        return G - S * Z * G
```

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/unit/test_blocksets.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/blocksets.py tests/unit/test_blocksets.py
git commit -m "feat: add symbolic block-set BLUP"
```

## Task 5: Symbolic Panel Entry Moments

**Files:**
- Modify: `src/symlmm/panel.py` (`PanelView.entry_mean`, `entry_covariance`, `entry_cross_covariance`)
- Test: `tests/unit/test_panel.py`

**Interfaces:**
- Consumes: `Block.mean`, `Block.covariance`, `Block.cross_covariance`, `PanelGrid.row` for dense only
- Produces: symbolic `PanelView.entry_mean`, `entry_covariance`, `entry_cross_covariance` without `dense()`

- [ ] **Step 1: Write the failing test**

Extend the imports at the top of `tests/unit/test_panel.py` with
`Unstructured` from `symlmm.covariances` and `Covariate`, `Group` from
`symlmm.specs`. Then add:

```python
def test_symbolic_panel_entry_moments_use_block_entries():
    model = _symbolic_ri_rs_model()
    panel = model.panel("subject", "time")
    i = sp.Symbol("i", integer=True)
    t = sp.Symbol("t", integer=True)
    s = sp.Symbol("s", integer=True)
    block = model.block("subject", i)

    assert panel.entry_mean(i, t) == block.mean()[t, 0]
    assert panel.entry_covariance(i, t, s) == block.covariance()[t, s]
    assert panel.entry_cross_covariance(i, t, i, s) == block.covariance()[t, s]
```

Use this fixture in `tests/unit/test_panel.py`:

```python
def _symbolic_ri_rs_model():
    b0, b1 = sp.symbols("b0 b1", real=True)
    tau0, tau1, sigma2 = sp.symbols("tau0 tau1 sigma2", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    i = sp.Symbol("i", integer=True)

    subject = Group("subject", index=i)
    time = Covariate("time")

    return LMM(
        fixed=Fixed(intercept=b0, slopes={time: b1}),
        random=[
            RandomEffects(
                group=subject,
                intercept=True,
                slopes=(RandomSlope(time),),
                cov=Unstructured.from_sd_corr([tau0, tau1], rho_b),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_panel.py::test_symbolic_panel_entry_moments_use_block_entries -v
```

Expected: fail with `NotDefined("panel grids require dense mode")`.

- [ ] **Step 3: Implement symbolic entry addressing**

Add a private helper to `PanelView`:

```python
    def _entry_row(self, level: Any, position: Any) -> Any:
        """Return a block-local row for dense or symbolic entry access."""
        if self.model._dense:
            return self.grid.row(level, position)
        return position
```

Update the three entry methods:

```python
    def entry_mean(self, level: Any, position: Any) -> Any:
        """Return E[Y_level,position]."""
        self._require_position()
        row = self._entry_row(level, position)
        return Block(self.model, self.factor, level).mean()[row, 0]

    def entry_covariance(self, level: Any, p1: Any, p2: Any) -> Any:
        """Return Cov(Y_level,p1, Y_level,p2)."""
        self._require_position()
        row1 = self._entry_row(level, p1)
        row2 = self._entry_row(level, p2)
        return Block(self.model, self.factor, level).covariance()[row1, row2]

    def entry_cross_covariance(self, level1: Any, p1: Any, level2: Any, p2: Any) -> Any:
        """Return Cov(Y_level1,p1, Y_level2,p2)."""
        self._require_position()
        row1 = self._entry_row(level1, p1)
        row2 = self._entry_row(level2, p2)
        return Block(self.model, self.factor, level1).cross_covariance(
            Block(self.model, self.factor, level2)
        )[row1, row2]
```

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/unit/test_panel.py -q
```

Expected: all pass. Existing dense tests continue to use `PanelGrid.row`.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/panel.py tests/unit/test_panel.py
git commit -m "feat: support symbolic panel entry moments"
```

## Task 6: No-Dense Guards, Docs, and Full Integration

**Files:**
- Create: `tests/integration/test_block_sets.py`
- Modify: `docs/pedagogy/blackboard.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: all prior tasks
- Produces: regression guards and documentation

- [ ] **Step 1: Add no-dense guards**

Create `tests/integration/test_block_sets.py`:

```python
import sympy as sp

from symlmm.blocks import Block
from symlmm.covariances import IID, Unstructured
from symlmm.model import LMM
from symlmm.specs import Covariate, Fixed, Group, RandomEffects, RandomSlope


def _ri_rs_model():
    b0, b1 = sp.symbols("b0 b1", real=True)
    tau0, tau1, sigma2 = sp.symbols("tau0 tau1 sigma2", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    i = sp.Symbol("i", integer=True)

    subject = Group("subject", index=i)
    time = Covariate("time")

    return LMM(
        fixed=Fixed(intercept=b0, slopes={time: b1}),
        random=[
            RandomEffects(
                group=subject,
                intercept=True,
                slopes=(RandomSlope(time),),
                cov=Unstructured.from_sd_corr([tau0, tau1], rho_b),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    ), i


def test_symbolic_block_moments_do_not_access_rows(monkeypatch):
    model, i = _ri_rs_model()
    block = model.block("subject", i)

    def fail(self):
        raise AssertionError("Block.rows() called from symbolic path")

    monkeypatch.setattr(Block, "rows", fail)

    assert block.shrinkage("subject").shape == (2, block.size())
    assert block.blup_mean("subject").shape == (2, 1)
    assert block.blup_covariance("subject").shape == (2, 2)
```

- [ ] **Step 2: Add the documentation section**

Add to `docs/pedagogy/blackboard.md`:

```markdown
### Abstract block moments

Block moments do not require a dense observation table. For a symbolic
subject block:

```python
i = sp.Symbol("i", integer=True)
A = model.block("subject", i)

A.size()                   # n_subject:i
A.response()               # Y_subject:i
A.design()                 # X_subject:i
A.covariance()             # V_subject:i
A.shrinkage("subject")     # G Z' V^-1
A.blup_mean("subject")     # E[b_i | Y_i]
```

`Block.rows()` and `PanelGrid.row()` are dense-evaluator internals. Symbolic
block queries use block matrices and symbolic block sizes.
```

- [ ] **Step 3: Update `CHANGELOG.md`**

Add under the unreleased section:

```markdown
### Added

- `Block.response()`, `Block.term()`, `Block.shrinkage()`,
  `Block.blup_mean()`, and `Block.blup_covariance()` for symbolic
  block-addressed moments.
- `BlockSet` and `LMM.block_set()` for symbolic joint block moments and BLUPs.
- Symbolic `PanelView.entry_*` moments without dense mode.
```

- [ ] **Step 4: Run the full gate**

```bash
uv run pytest --cov=symlmm --cov-fail-under=90
uv run pytest -q -W error::DeprecationWarning
uv run mypy src/symlmm
uv run ruff check
uv run ruff format --check
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_block_sets.py docs/pedagogy/blackboard.md CHANGELOG.md
git commit -m "docs: document abstract block moments"
```

## Acceptance Criteria

- `Block.shrinkage`, `Block.blup_mean`, and `Block.blup_covariance` work on a symbolic `LMM` with no `.dense()` call.
- `BlockSet` returns symbolic sizes, responses, designs, random designs, and block covariance.
- `BlockSet.blup_mean` matches the global dense BLUP after substituting the same response and parameters.
- Symbolic `PanelView.entry_mean`, `entry_covariance`, and `entry_cross_covariance` work with no `.dense()` call.
- `Block.rows()` and `PanelGrid.row()` are not called by the new symbolic paths.
- Existing dense behavior and tests remain green.
- Full CI gate passes.

## Risks

- **Term-design consistency:** the block-set BLUP is scoped to independent blocks of one factor. Nested and crossed joint BLUP needs a separate restriction-design algebra. Do not fake it with row selection.
- **SymPy simplification:** `Inverse` expressions may not simplify structurally; tests compare against the direct definition and against dense evaluation.
- **Block ordering:** `BlockSet` preserves caller order; `BlockDiagMatrix` and `BlockMatrix` use that order. Document that callers must keep the same order for `Y`, `Z`, and `G`.
- **API naming:** `Block.response()` returns a symbolic response vector, not observed data. Dense evaluation remains the place for concrete values.
- **Concurrency:** the core branch is the only writer of `blocks.py`; consumers are created from the post-core integration branch. A consumer that needs a core API change must rebase and request a core follow-up, not edit `blocks.py` in parallel.
