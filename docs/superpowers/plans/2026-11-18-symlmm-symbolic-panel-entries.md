# Symbolic Panel Entry Moments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `PanelView.entry_mean`, `entry_covariance`, and `entry_cross_covariance` work for symbolic models, while keeping `position_mean` and `position_covariance` dense-only with an explicit error.

**Architecture:** Treat the position code as the symbolic block-row address. `PanelGrid.row(level, position)` returns `position` in symbolic mode; the existing symbolic `Block.mean()`, `Block.covariance()`, and `Block.cross_covariance()` matrices are then indexed by that position. No new covariance algebra is introduced.

**Tech Stack:** Python 3.14, SymPy, pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-11-18-symlmm-factor-first-design.md`  
**Base:** `562a6e4`

## Global Constraints

- Python 3.14; no new runtime dependencies.
- Public functions/classes have docstrings.
- No new hardcoded `time` or `occasion` identifiers in the new public API.
- Dense behavior must not change.
- CI:
  - `uv run pytest --cov=symlmm --cov-fail-under=90`
  - `uv run mypy src/symlmm`
  - `uv run ruff check`
  - `uv run ruff format --check`
- Worktree/branch: create a new worktree from the factor-first branch; do not edit the completed cleanup worktree directly.

## Design Decision

For a symbolic balanced panel, a block’s rows are assumed ordered by position code. Therefore:

```text
entry (level, position) -> Block(model, factor, level).mean()[position, 0]
```

This is the minimum change: `PanelGrid.row` already computes the dense mapping; in symbolic mode it returns the position itself. The alternative — a separate per-entry matrix symbol — is rejected as over-engineering until a concrete use case requires non-contiguous symbolic rows.

Symbolic position moments (`position_mean`, `position_covariance`) remain out of scope because they would require a symbolic level summation/matrix, not a single entry.

## Task 1: Symbolic Row Addressing

**Files:**
- Modify: `src/symlmm/panel.py` (`PanelGrid.row`, `PanelView.entry_*`)
- Create: `tests/unit/conftest.py`
- Test: `tests/unit/test_panel.py`

**Interfaces:**
- Consumes: `Block.mean()`, `Block.covariance()`, `Block.cross_covariance()`
- Produces: `PanelGrid.row(level, position)` returning `position` for symbolic models

- [ ] **Step 1: Write the failing test**

Create `tests/unit/conftest.py`:

```python
import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


@pytest.fixture
def symbolic_panel_model():
    b0, b1 = sp.symbols("b0 b1", real=True)
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0, slopes={"p": b1}),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
    )
```

Add to `tests/unit/test_panel.py`:

```python
from symlmm.blocks import Block


def test_symbolic_entry_moments_use_position_as_row(symbolic_panel_model):
    model = symbolic_panel_model
    panel = PanelView(model, model.factor("F"), Position("p"))
    block = Block(model, panel.factor, 0)

    assert panel.entry_mean(0, 0) == block.mean()[0, 0]
    assert panel.entry_covariance(0, 0, 1) == block.covariance()[0, 1]
    assert panel.entry_cross_covariance(0, 0, 1, 1) == (
        block.cross_covariance(Block(model, panel.factor, 1))[0, 1]
    )
```

- [ ] **Step 2: Run the test to verify failure**

```bash
uv run pytest tests/unit/test_panel.py::test_symbolic_entry_moments_use_position_as_row -v
```

Expected: `NotDefined` from `PanelGrid.row`.

- [ ] **Step 3: Implement symbolic row addressing**

In `PanelGrid.row`:

```python
def row(self, level: Any, position: Any) -> Any:
    """Return the block-local row for a level and position."""
    if not self.model._dense:
        return position
    ordered = self.rows_for_level(level)
    target = next(
        (row for row in ordered if self.position_codes()[row] == position),
        None,
    )
    if target is None:
        raise NotDefined(f"position {position!r} is not present for level {level!r}")
    block_rows = [row for row, code in enumerate(self.level_codes()) if code == level]
    return block_rows.index(target)
```

Keep `rows_for_level` dense-only.

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/unit/test_panel.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/panel.py tests/unit/test_panel.py
git commit -m "feat: support symbolic panel entry moments"
```

## Task 2: Explicit Dense Guard for Position Moments

**Files:**
- Modify: `src/symlmm/panel.py`
- Test: `tests/unit/test_panel.py`

**Interfaces:**
- Produces: `NotDefined` for symbolic `position_mean`/`position_covariance`

- [ ] **Step 1: Write failing tests**

```python
def test_symbolic_position_moments_raise_not_defined(symbolic_panel_model):
    model = symbolic_panel_model
    panel = PanelView(model, model.factor("F"), Position("p"))
    with pytest.raises(NotDefined, match="dense"):
        panel.position_mean(0)
    with pytest.raises(NotDefined, match="dense"):
        panel.position_covariance(0, 1)
```

- [ ] **Step 2: Run and confirm failure**

```bash
uv run pytest tests/unit/test_panel.py::test_symbolic_position_moments_raise_not_defined -v
```

- [ ] **Step 3: Add the guard**

Add a small private helper and call it at the start of both methods:

```python
def _require_dense(self) -> None:
    """Raise when a moment needs concrete observation rows."""
    if not self.model._dense:
        raise NotDefined("position moments require dense mode")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_panel.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/symlmm/panel.py tests/unit/test_panel.py
git commit -m "fix: guard dense-only position moments"
```

## Task 3: Symbolic Blackboard Entry Coverage

**Files:**
- Test: `tests/unit/test_blackboard.py`

**Interfaces:**
- Consumes: `panel.symbols()`, `syms.E`, `syms.Var`, symbolic `PanelView.entry_*`

- [ ] **Step 1: Write the failing test**

```python
def test_symbolic_blackboard_entry_moments_use_panel_entries(symbolic_panel_model):
    model = symbolic_panel_model
    panel = model.panel("F", "p")
    syms = panel.symbols()
    g = panel.factor.index
    t = panel.position.index

    assert syms.E(syms.Y[g, t]) == panel.entry_mean(g, t)
    assert syms.Var(syms.Y[g, t]) == panel.entry_covariance(g, t, t)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_blackboard.py::test_symbolic_blackboard_entry_moments_use_panel_entries -v
```

- [ ] **Step 3: Fix only if the failure is in the entry path**

Expected implementation: `PanelAdapter._entry_row` delegates to `PanelGrid.row`, which now returns the symbolic position. Do not add a second mapping layer.

- [ ] **Step 4: Run the focused tests**

```bash
uv run pytest tests/unit/test_blackboard.py tests/unit/test_panel.py -q
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_blackboard.py
git commit -m "test: cover symbolic blackboard entry moments"
```

## Task 4: Documentation and Full Verification

**Files:**
- Modify: `docs/pedagogy/blackboard.md`
- Modify: `src/symlmm/panel.py` docstrings if needed

- [ ] **Step 1: Update the docs**

Change the “entry moments are dense-only” note to:

```text
Entry moments work for symbolic balanced panels: the position code is used as
the block-row address. Position-column moments (`position_mean`,
`position_covariance`) remain dense-only because they require a concrete
level count and concrete rows.
```

Document the balanced-panel assumption in `PanelGrid.row` and `PanelView.entry_*`.

- [ ] **Step 2: Run the full gate**

```bash
uv run pytest --cov=symlmm --cov-fail-under=90
uv run mypy src/symlmm
uv run ruff check
uv run ruff format --check
```

- [ ] **Step 3: Commit**

```bash
git add docs/pedagogy/blackboard.md src/symlmm/panel.py
git commit -m "docs: clarify symbolic panel entry semantics"
```

## Acceptance Criteria

- Symbolic `entry_mean`, `entry_covariance`, and `entry_cross_covariance` match the corresponding `Block` matrix entries.
- Symbolic `position_mean` and `position_covariance` raise `NotDefined` with a clear dense-mode message.
- Dense panel behavior is unchanged.
- Symbolic blackboard entry moments work through `panel.symbols()`.
- Full suite, coverage, mypy, and ruff are green.

## Risks

- **Row-order assumption:** symbolic entries assume block rows are ordered by position. Document this; if non-contiguous or unbalanced symbolic rows are needed later, introduce an explicit row symbol instead of reusing the position code.
- **Position index:** `PanelView.position_code` still keys off the panel factor for symbolic within-coordinate consistency; `Position.index` is not the row address. Document the distinction.
- **Scope:** position-column moments remain dense-only; do not attempt symbolic level summation in this plan.
