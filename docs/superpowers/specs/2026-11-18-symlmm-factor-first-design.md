# Factor-First Model API Design

**Status:** accepted for implementation
**Date:** 2026-11-18
**Scope:** replace the primary-group/occasion panel core with a factor-first model API; keep a panel view as an optional adapter.

## 1. Context

`symlmm` derives symbolic response moments for linear mixed, marginal, and GEE models. The stacked observation-row engine already supports multiple random-effect terms, nested and crossed factors, and arbitrary residual covariance structures. Row-level `covariance_entry(i, j)` returns correct moments for hierarchical models when factor codes are distinct.

The public canonical layer does not share that generality:

- It assumes one primary grouping factor and one within-group occasion axis.
- `group_covariance` and related methods reject models with more than one grouping factor.
- `Nested` labels include only the immediate parent; ancestry above that parent is lost.
- Panel terminology (`occasion`, `time_*`) leaks into the core API.
- The word “canonical” is used for default selection, not canonicalization.

The result is that hierarchical models are representable at the row level but not first-class at the block level, and deep nesting can produce incorrect covariance.

## 2. Problem

### 2.1 Incomplete nested ancestry

`Nested(G, F)` labels levels `F:G`. If `F` is itself nested in an ancestor `E`, the resolved label remains `F:G`; it does not include `E`. Dense codes therefore collide when the same child labels appear under different ancestors.

Generic reproduction:

```text
factors: E, F, G
E = [0, 0, 1, 1]
F = [0, 0, 0, 0]
G = [0, 1, 0, 1]
```

`G` codes become `[0, 1, 0, 1]`, so `G0@E0` and `G0@E1` share a level code. Their covariance includes the `G` variance component instead of zero. This is a correctness bug, not only a naming issue.

### 2.2 Panel assumptions in the core

`group_count`, `occasion_count`, `response_matrix`, `time_response`, `time_mean`, and `time_covariance` all assume:

- one primary group;
- one global within-group axis;
- a balanced panel.

A hierarchical model may have no within-group position axis and may have several equally important grouping factors. Treating one as primary is arbitrary.

### 2.3 Single-factor block covariance

`group_covariance` raises when random terms use more than one factor. For a hierarchy, the covariance of a block must include all random terms whose factors are equal to, ancestors of, or descendants of that block’s factor. The current method includes only the primary term and the residual.

### 2.4 Overloaded vocabulary

The same axis is called `coordinate`, `occasion`, `time`, and `canonical` in different places. The constructor argument `occasion` names a raw label column, while `occasion()` returns an integer code. `occasion_definition` is a `Covariate`, not an occasion. The vocabulary obscures the model.

## 3. Goals and non-goals

### Goals

- Make grouping factors the primary public abstraction.
- Support arbitrary nested, crossed, and deep hierarchies.
- Provide block moments for any `(factor, level)`.
- Support term-level covariance decomposition and partial covariance.
- Preserve longitudinal and spatial modeling through an explicit panel view and existing covariance structures.
- Remove domain-specific `time` and `canonical` wording from the core API.
- Provide a migration path from 0.5.x.

### Non-goals

- Model fitting.
- Bayesian hierarchical models.
- Group-level predictors.
- Arbitrary observed-data conditioning beyond existing `conditional_*` and `blup_*`.
- Automatic expansion of symbolic block matrices.
- Multi-membership models beyond crossed factors.

## 4. Design decisions

| Decision | Choice |
|----------|--------|
| Primary abstraction | Factor-first: blocks addressed by `(factor, level)`. |
| Nested ancestry | Hybrid: `within: str | FactorSpec`; strings resolve recursively through registered factors. |
| Panel view | Thin `PanelView` adapter; panel methods leave `LMM`. |
| Term identity | Explicit `RandomEffects.name`; default resolved factor label. |
| Partial covariance | Term-only exclusion; block restriction is separate. |
| Crossed conditioning | Uniform term exclusion; no node-conditioning special case. |
| Observed-data conditioning | Remains `conditional_*` / `blup_*`. |
| Vocabulary | `factor`, `level`, `block`, `node`, `position`, `coordinate`, `term`, `panel`. |
| Compatibility | 0.6.0 adds new API and deprecation warnings; 1.0 removes old names. |

## 5. Core concepts

### Factor

A resolved grouping dimension owned by the model.

```python
@dataclass(frozen=True)
class Factor:
    name: str                      # lookup name, e.g. "H"
    label: str                     # unique label, e.g. "F:G:H"
    components: tuple[str, ...]    # simple factor names, e.g. ("F", "G", "H")
    index: sp.Symbol
    parent: Factor | None
    kind: Literal["group", "nested", "crossed"]
```

`Group`, `Nested`, and `Crossed` remain user declarations. The public declaration type is `FactorSpec = Group | Nested | Crossed`. The model resolves declarations at construction and stores `Factor` objects. The resolved `Factor.label` is the source of truth for symbols, dense codes, and diagnostics.

### Level

A concrete value of a factor.

- Dense mode: integer code over the factor’s label space.
- Symbolic mode: an index expression such as `g_F_G_H[i]`.

### Block

The observation set belonging to one factor level.

```python
@dataclass(frozen=True)
class Block:
    model: LMM
    factor: Factor
    level: Any
```

A block is the unit for group means, covariance, cross-covariance, term decomposition, and partial covariance. A block is not a random-effect vector; `random_effect()` returns the random-effect vector associated with the block when one exists.

### Term

A named random-effect component.

```python
RandomEffects(
    group=H,
    intercept=True,
    slopes=(...),
    cov=...,
    name="H",             # optional
)
```

- `name` is unique across the model.
- Default name is the resolved factor label.
- `"residual"` is reserved and refers to the residual covariance.
- Term order is the model’s random-term order; public keys do not depend on list position.
- If a factor ever has more than one block, explicit names become mandatory.

### Position

An optional within-block index used only by `PanelView`.

```python
@dataclass(frozen=True)
class Position:
    name: str                 # coordinate column used to order observations
    column: str | None = None # optional explicit position-label column
    index: sp.Symbol | None = None
```

- If `column` is provided, position codes are assigned by first appearance within each block.
- Otherwise, positions are ranks of `name` values sorted within each block.
- `index` is the symbolic position index; default derived from `name`.
- `Position` is not a core model concept. A model without a panel has no position axis.

### Coordinate

A measured variable used in fixed-effect designs, random slopes, and covariance functions. A coordinate is never an index axis.

## 6. Public API

### Model

```python
model.factor(ref)                    # resolved Factor by name, path, spec, or Factor
model.factors()                      # tuple[Factor, ...]
model.levels(factor)                 # symbolic or dense level count
model.level_code(factor, i)          # replaces group(i, factor)

model.block(factor, level)           # Block
model.block_mean(factor, level)
model.block_covariance(factor, level)
model.block_cross_covariance(factor1, level1, factor2, level2)

model.covariance_terms(a, b)
model.partial_covariance(a, b, exclude=[...])
model.partial_correlation(a, b, exclude=[...])

model.panel(factor, position)        # PanelView
```

`ref` accepts a factor name, a component path tuple, a factor declaration, or a resolved `Factor`. `a` and `b` accept row indices or blocks. Results are scalar for rows and matrices for blocks. `partial_correlation` accepts row indices only.

### Block

```python
block = model.block("H", h)

block.size()
block.mean()
block.covariance()
block.cross_covariance(other)
block.design()
block.random_design()
block.random_design(term="H")
block.residual()
block.random_effect()
block.covariance_terms()
block.partial_covariance(other, exclude=[...])
```

### PanelView

```python
panel = model.panel("F", "p")

panel.level_count()
panel.position_count()
panel.response_matrix()
panel.response(level)
panel.response_column(position)
panel.entry_mean(level, position)
panel.entry_covariance(level, p1, p2)
panel.entry_cross_covariance(level1, p1, level2, p2)
panel.position_mean(position)
panel.position_covariance(p1, p2)
panel.position_code(i)
panel.symbols()
```

## 7. Factor resolution

### Group

```python
Group("F")
```

- `components = ("F",)`
- `label = "F"`
- `parent = None`
- Index defaults to `Symbol("F", integer=True)` unless provided.

### Nested

```python
F = Group("F")
G = Nested("G", F)
H = Nested("H", G)          # path ("F", "G", "H")
```

`Nested.within` accepts `str | FactorSpec`.

1. If `within` is a factor declaration, resolve it recursively and prepend its components.
2. If `within` is a string and a registered factor with that name exists, resolve that factor and prepend its components.
3. If the string is not registered, keep the legacy shallow interpretation: components `(within, child)`.
4. Cycles, duplicate names with different component paths, and ambiguous lookups raise `ModelError`.

The declaration-level `Nested.label` may be shallow for legacy strings. The resolved `Factor.label` is authoritative.

### Crossed

`Crossed` accepts two or more components, each `str | FactorSpec`.

- Components are the concatenation of the component factors’ simple names.
- `label = ":".join(components)`.
- The existing two-argument form remains valid.
- `Crossed` gains `name: str | None = None`; the default preserves the existing lookup name (the final component). An explicit name is recommended when crossing more than two components or when names would collide.
- Duplicate component names with different definitions raise `ModelError`.

### Dense codes

For a resolved factor with components `(k1, ..., km)`:

1. If an observation column named `"k1:...:km"` exists, use it.
2. Otherwise build the composite key from the component columns.
3. Map composite keys to integer codes in sorted order.

`model.level_code(factor, i)` must use the resolved components. `PanelView` position codes are derived separately from the chosen position axis.

### Validation

- Nested ancestry must be acyclic.
- A factor name maps to exactly one resolved path.
- `Nested("H", "missing")` is legacy shallow nesting and is documented as such.
- Repeated child labels under different ancestors produce distinct composite codes.
- `Crossed` components must be resolvable.
- Conflicting definitions for the same lookup name raise `ModelError`.

## 8. Moment algebra

### Row-level decomposition

For rows `i`, `j`:

```text
Cov(Y_i, Y_j) = R_ij + Σ_t z_t(i)' G_t z_t(j)
```

API:

```python
model.covariance_terms(i, j)
# ordered mapping: {"term_a": ..., "residual": ...}
```

The mapping is ordered by model term order, with `"residual"` last.

### Block moments

For blocks `A` and `B`:

```text
Cov(Y_A, Y_B) = R_AB + Σ_t Z_{A,t} G_t Z_{B,t}'
```

where `Y_A` is the stacked response vector for rows in `A` and `Z_{A,t}` is term `t`’s design matrix restricted to those rows.

This formula applies uniformly to:

- nested and non-nested factors;
- ancestor and descendant blocks;
- crossed factors;
- disjoint and overlapping blocks.

The single-grouping-factor restriction is removed from block covariance.

### Term contributions

```text
C_t(A, B) = Z_{A,t} G_t Z_{B,t}'
```

`covariance_terms(A, B)` returns one matrix per term plus `"residual"`. `covariance_terms(i, j)` returns scalars.

### Partial covariance

```text
partial_cov(A, B | exclude) =
    Cov(Y_A, Y_B) - Σ_{t in exclude} Z_{A,t} G_t Z_{B,t}'
```

API:

```python
model.partial_covariance(A, B, exclude=["t"])
model.partial_covariance(i, j, exclude=["t", "residual"])
```

Rules:

- `exclude` accepts term names only.
- `"residual"` is valid.
- Unknown or duplicate names raise `ModelError`.
- Excluding every term and the residual returns a zero matrix/scalar.
- Block restriction is expressed by selecting blocks first, not by `exclude=`.

### Partial correlation

Defined for scalar entries only:

```text
partial_corr(i, j | exclude) =
    partial_cov(i, j) / sqrt(partial_cov(i, i) * partial_cov(j, j))
```

Zero partial variance raises `NotDefined`. Block correlation is derived by the caller.

### Observed-data conditioning

`conditional_*` and `blup_*` keep their existing meaning: conditional distribution given observed data or random effects. They are not merged with `partial_*`.

## 9. Symbolic representation

For symbolic blocks:

```text
X_A       MatrixSymbol("X_A", n_A, p)
Z_{A,t}   MatrixSymbol("Z_A:t", n_A, q_t)
R_AB      MatrixSymbol("R_AB", n_A, n_B)
n_A       Symbol("n_A", positive=True, integer=True)
```

Term contributions are matrix expressions. Dense mode computes them from actual row subsets. Block matrix symbols are not automatically expanded; scalar entry moments remain fully expanded and exact.

## 10. PanelView semantics

### Construction

```python
panel = model.panel(factor, position)
```

- `factor` selects one grouping factor.
- `position` accepts `str | Position`.
- A string is shorthand for `Position(name)`.

### Balanced panels

A panel is balanced when every factor level has the same number of position codes.

- Positions are compared by **position code**, not raw coordinate value.
- Different raw coordinate values that rank to the same codes are balanced.
- Explicit position labels are assigned by first appearance within each block; codes, not raw labels, define balance.
- Unbalanced blocks raise `NotDefined` for `response_matrix`, `position_mean`, and `position_covariance`.
- The factor-first block API remains available for unbalanced data.

### Response matrix

For a balanced panel:

```text
Y = MatrixSymbol("Y", level_count, position_count)
```

`response(level)` returns a row; `response_column(position)` returns a column; `entry_*(level, position)` returns entries.

### Blackboard

- `panel.symbols()` returns the existing `ModelSymbols` bound to a `PanelView`.
- `LMM.symbols()` becomes a deprecated wrapper around a default panel view.
- Block-level symbolic handles are available through `Block`; a separate `BlockSymbols` facade is out of scope for 0.6.0.

## 11. Migration and deprecation

### 0.6.0

- Add factor-first model and `Block` API.
- Add `PanelView`.
- Add term decomposition and partial covariance.
- Add deprecated wrappers for old methods.
- Rename `canonical.py` to `blocks.py`.
- Remove “canonical” from public vocabulary.

### 1.0

- Remove `group_*`, `entry_*`, `time_*`, `response_matrix`, and `symbols` from `LMM`.
- Remove `occasion`/`time` naming. Remove legacy shallow nesting; nested factors must resolve through a registered factor or an explicit `FactorSpec`.

### Migration table

| 0.5.x | 0.6.x |
|-------|-------|
| `model.group_count()` | `model.levels(factor)` or `panel.level_count()` |
| `model.occasion_count()` | `panel.position_count()` |
| `model.group_size(level)` | `model.block(factor, level).size()` |
| `model.group_mean(level)` | `model.block(factor, level).mean()` |
| `model.group_covariance(level)` | `model.block(factor, level).covariance()` |
| `model.group_cross_covariance(a, b)` | `block_a.cross_covariance(block_b)` |
| `model.entry_mean(level, t)` | `panel.entry_mean(level, position)` |
| `model.entry_covariance(level, t, s)` | `panel.entry_covariance(level, p1, p2)` |
| `model.time_response(t)` | `panel.response_column(position)` |
| `model.time_mean(t)` | `panel.position_mean(position)` |
| `model.time_covariance(t, s)` | `panel.position_covariance(p1, p2)` |
| `model.symbols()` | `panel.symbols()` |
| `model.response_matrix()` | `panel.response_matrix()` |
| `model.group(i, factor)` | `model.level_code(factor, i)` |
| `model.occasion(i, factor)` | `panel.position_code(i)` |

## 12. Implementation phases

### Phase 1 — Factor resolution and hierarchy correctness

- Add resolved `Factor` and component-path resolution.
- Extend `Nested.within` to `str | FactorSpec`.
- Extend `Crossed` to two or more components.
- Fix dense codes to use full ancestry.
- Validate cycles, duplicate names, ambiguous paths.
- Regression tests for repeated child labels under different ancestors.

### Phase 2 — Term identity and row-level decomposition

- Add optional `RandomEffects.name`.
- Reserve `"residual"`; reject duplicate/unknown names.
- Add `covariance_terms(i, j)`.
- Add `partial_covariance(i, j, exclude=[...])` and `partial_correlation(i, j, exclude=[...])`.
- Update conditional/BLUP internals to use term names instead of positional group indices.

### Phase 3 — Block API

- Add `model.block(factor, level)` and `Block`.
- Add block size, mean, covariance, cross-covariance, design, random design, residual, random effect, term decomposition, and partial methods.
- Remove the single-grouping-factor restriction.
- Add symbolic block matrices.
- Tests for nested, deep-nested, crossed, ancestor/descendant, overlapping blocks; dense vs symbolic; sum-of-terms identity.

### Phase 4 — PanelView and deprecation

- Add `Position` and `model.panel`.
- Move panel methods onto `PanelView`.
- Fix balanced-panel logic to compare position codes.
- Bind blackboard to `PanelView`.
- Add deprecated wrappers for old methods.
- Tests for differing raw coordinates, explicit position columns, unbalanced panels, and deprecation warnings.

### Phase 5 — Vocabulary and documentation

- Rename `canonical.py` to `blocks.py`.
- Remove “canonical” from public vocabulary.
- Update `CHANGELOG.md`, `README.md`, pedagogy docs, and migration table.
- Update `__init__.py` exports and internal imports.

### Delivery split

Two PRs:

1. **Hierarchy correctness and term algebra** — Phases 1–2.
2. **Factor-first block API, PanelView, and vocabulary cleanup** — Phases 3–5.

## 13. Testing

- Unit tests for factor resolution, component paths, dense codes, term names, decomposition, and exclusion.
- Property checks:
  - `Cov(A, B) == Σ covariance_terms(A, B)`.
  - `partial_covariance(A, B, exclude=T) + Σ_{t∈T} term_t(A,B) == Cov(A,B)`.
  - `partial_correlation` in `[-1, 1]` where defined.
  - Dense block covariance equals the corresponding submatrix of `matrix_covariance()`.
- Integration tests:
  - deep nesting with repeated child labels;
  - crossed factors;
  - deep ancestor/descendant blocks;
  - balanced and unbalanced panels;
  - differing raw coordinates across levels.
- Regression tests:
  - nested ancestry collision;
  - position counting using codes rather than raw coordinates.
- Commands: `pytest`, `mypy --strict`, `ruff`.

## 14. Risks

| Risk | Mitigation |
|------|------------|
| Symbolic block matrices are opaque and may not simplify as users expect. | Document the representation; keep scalar entries fully expanded; dense mode exact. |
| Large breaking API change. | 0.6.0 adds new API with deprecation warnings; 1.0 removes old names; migration table. |
| String-based `Nested.within` remains ambiguous. | Explicit `FactorSpec` is preferred and authoritative; strings resolve through registered factors; unresolved strings are documented as legacy shallow nesting until 1.0. |
| Crossed-factor naming/path composition gets complex. | Unique resolved labels; reject ambiguous duplicate names; preserve two-component compatibility. |
| Blackboard is tightly coupled to the panel model. | Bind blackboard symbols to `PanelView`; keep block API independent. |
| Existing tests and docs assume primary group/occasion. | Update in Phases 4–5; keep deprecated wrappers covered until removal. |

## 15. Alternatives considered

- **Panel-first**: keep the current primary-group/occasion core and add hierarchy beside it. Rejected because hierarchy remains second-class and the vocabulary stays mixed.
- **Index-tuple-first**: generic queries over arbitrary index tuples. Rejected as over-engineered for symbolic moments; factor-first covers the required generality.
- **Node conditioning**: allow arbitrary block/node conditioning in partial covariance. Rejected because “condition on a descendant node” is ambiguous and often degenerate; term exclusion is precise.
- **Single-factor block covariance**: keep the current restriction. Rejected because it cannot answer hierarchical covariance questions.
- **Explicit ancestry only**: require `Nested` to receive a factor declaration. Rejected as a breaking change without a migration path; hybrid keeps legacy strings.

## 16. Acceptance criteria

- A deep hierarchy produces distinct codes for repeated child labels under different ancestors.
- `block_covariance` and `block_cross_covariance` include all contributing random terms and equal dense covariance submatrices.
- `covariance_terms` sums to the total covariance for rows and blocks.
- `partial_covariance` and `partial_correlation` satisfy the stated identities.
- `PanelView` handles differing raw coordinates across levels correctly and rejects genuinely unbalanced panels.
- Crossed factors work under block moments, term decomposition, and term exclusion.
- Deprecated 0.5.x methods remain functional with warnings.
- `pytest`, `mypy --strict`, and `ruff` pass.
