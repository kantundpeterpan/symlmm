# Blackboard-style moments for factor blocks and panel positions

`symlmm` derives symbolic moments; it does not fit models. This page shows the
blackboard facade built on the factor-first API. A `Factor` indexes group-level
blocks, and an optional `Position` indexes positions within each block. The
same model can be read in textbook notation: `g` indexes the factor level,
`p` indexes the position, `Y_g` is the block response vector, and `Y_{gp}` is
the scalar response at position `p` in block `g`.

The notation used here is:

```text
g = factor level
p = position within a block

Y_g     = response vector for block g
Y_{gp}  = scalar response at level g, position p
b_g     = random-effect vector for block g
X_g     = fixed design block for block g
Z_g     = random design block for block g
R_g     = residual covariance block for block g
G       = random-effect covariance
V_g     = Z_g G Z_g' + R_g

Cov(Y_g, Y_h) = 0              for g != h
Cov(Y_{gp}, Y_{gq}) = [V_g]pq
```

For a balanced panel, `Y` is the `m x P` matrix with factor levels as rows and
positions as columns. `Y_g = Y[g, :]` is a row vector, `Y[:, p]` is the
position column, and `Y_{gp}` is the scalar entry. For an unbalanced panel,
`Y_g` remains the block response vector but is not a row of a rectangular `Y`
matrix; `Y_{gp}` remains defined, while position columns are not.

Python examples use 0-based indices. Rendered mathematical formulas use 1-based
bounds and state the conversion when they show index ranges.

The stacked observation-row index used by the matrix engine is an
implementation detail. It is not `Y_g` and it is not the public block index.

## Model definitions carry the index symbols

Index symbols belong to `Group`, `Nested`, `Crossed`, `Covariate`, and
`Position` definitions. The model resolves a `Factor` and a `Position` from
them; nothing hardcodes `g` or `p`.

```python
import sympy as sp
import symlmm as sl

g = sp.Symbol("g", integer=True, nonnegative=True)
p = sp.Symbol("p", integer=True, nonnegative=True)

subject = sl.Group("subject", index=g)
visit = sl.Position("time", index=p)

tau2 = sp.Symbol("tau2", positive=True)
sigma2 = sp.Symbol("sigma2", positive=True)
rho = sp.Symbol("rho", positive=True)
beta0, beta1 = sp.symbols("beta0 beta1", real=True)

model = sl.LMM(
    fixed=sl.Fixed(intercept=beta0, slopes={sl.Covariate("time", index=p): beta1}),
    random=[sl.RandomEffects(group=subject, intercept=True, cov=sl.Scalar(tau2))],
    residual=sl.AR1(
        group=subject,
        coordinate=sl.Covariate("time", index=p),
        var=sigma2,
        rho=rho,
    ),
    coordinate="time",
    n=4,
)

dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
panel = dense.panel(subject, visit)

assert panel.factor.index == g
assert panel.position.index == p
```

Passing `index=` is optional. When it is omitted, a `Group` derives its index
from its label and a `Position` or `Covariate` derives it from its name.
Examples pass it explicitly so the intended model notation is visible.

`RandomSlope(coordinate=...)` takes a `Covariate` definition, not a bare
string. A bare string is accepted only as a compatibility shim and normalized
to `Covariate(name)`; examples pass the definition.

## Choosing a factor and position

`model.panel(factor, position)` returns a `PanelView`.

- The factor defaults to the primary grouping factor: the first
  `RandomEffects` block's `Group`, otherwise the residual covariance's group
  definition, otherwise the first registered grouping definition.
- The position defaults to the model's coordinate name. Pass a `Position` or a
  column name for an explicit axis. `None` only omits the axis when the model
  has no coordinate; for block-only work use `model.block(factor, level)`.
- Position codes are a 0-based rank in sorted position order within each
  block, not the raw coordinate value. A `Position` with `column=` supplies
  the codes by first appearance within each block instead.

```python
panel = dense.panel(subject, visit)      # explicit factor and position
panel = dense.panel(subject)             # position from coordinate="time"
block = dense.block(subject, 0)          # one block, no position axis
```

Only the selected factor defines `Y_g` and its design/residual blocks. With
multiple grouping factors, block covariance and entry covariance raise
`NotDefined` rather than dropping the other factors' contributions.

## Block and panel API

`Block` exposes moments for one factor level. `PanelView` adds a position axis
and returns SymPy expressions directly; neither takes `form`:

```python
block = dense.block(subject, 0)

block.size()                 # n_g
block.rows()                 # dense observation rows
block.mean()                 # E[Y_g] = X_g beta
block.design()               # X_g
block.random_design()        # Z_g
block.residual()             # R_g
block.random_effect()        # b_g
block.covariance()           # V_g
block.cross_covariance(other)

panel.response_matrix()      # Y, m x P
panel.level_count()          # m
panel.position_count()       # P
panel.response(0)            # Y_g row
panel.response_column(0)     # Y[:, p] column
panel.entry_mean(0, 1)       # E[Y_{g,p}]
panel.entry_covariance(0, 1, 1)
panel.position_mean(0)       # E[Y[:, p]]
panel.position_covariance(0, 1)
```

Position-column moments (`position_mean`, `position_covariance`) require a
concrete level count; the fully symbolic arbitrary-`m` form raises
`NotDefined`.

## Blackboard symbols

`panel.symbols()` returns `ModelSymbols` bound to the panel's factor and
position. Its indices are the definitions' own indices.

```python
syms = panel.symbols()

assert syms.i == panel.factor.index
assert syms.t == panel.position.index

syms.Y[:, :]                # Y, m x P
syms.Y[0, :]                # Y_g row
syms.Y[:, 0]                # Y[:, p] column
syms.Y[0, 1]                # Y_{g,p} scalar

syms.b[syms.i]              # b_g vector
syms.X(syms.i)              # X_g
syms.Z(syms.i)              # Z_g
syms.R(syms.i)              # R_g
syms.G                      # G
syms.beta                   # beta
```

A symbolic-model panel can leave positions symbolic. Entry moments require
dense mode; position-column moments additionally require a concrete level
count. A bare integer index is ambiguous, so use `Y[0, :]` for a block row or
`Y[:, 0]` for a position column.

## Block-level moments

The facade delegates block moments to the factor-first model API:

```python
Y_g = syms.Y[0, :]

assert syms.E(Y_g) == dense.block(subject, 0).mean().T
assert syms.Var(Y_g) == dense.block(subject, 0).covariance()
assert syms.Cov(Y_g, Y_g) == dense.block(subject, 0).covariance()

Y_gp = syms.Y[0, 1]

assert syms.E(Y_gp) == panel.entry_mean(0, 1)
assert syms.Var(Y_gp) == panel.entry_covariance(0, 1, 1)
```

In matrix terms, `Cov(Y_g) = V_g = Z_g G Z_g' + R_g`. Across blocks,
`Cov(Y_g, Y_h) = 0` for `g != h`:

```python
assert syms.Cov(syms.Y[0, :], syms.Y[1, :]) == sp.zeros(2, 2)
assert dense.block(subject, 0).cross_covariance(dense.block(subject, 1)) == sp.zeros(2, 2)
```

## Conditional moments and the law of total variance

Condition on the block random-effect vector `b_g`:

```python
b_g = syms.b[syms.i]

E_Y_given_b = syms.E(syms.Y[0, :], given=b_g)
V_Y_given_b = syms.Var(syms.Y[0, :], given=b_g)

# X_g beta + Z_g b_g
# R_g
```

The law of total variance gives the marginal block covariance:

```python
V_Y = syms.Var(syms.Y[0, :])
between = syms.Var(syms.E(syms.Y[0, :], given=b_g))
within = syms.E(syms.Var(syms.Y[0, :], given=b_g))

assert between + within == V_Y
assert syms.total_variance(syms.Y[0, :], given=b_g) == V_Y
```

`between` is the random-effect contribution `Z_g G Z_g'`; `within` is the
residual contribution `R_g`.

## What remains stacked-engine internal

The engine still stacks all observations into one vector `y` for matrix
assembly, dense evaluation, variograms, and BLUP. The following methods take
the stacked row cursor and are not block notation:

```python
row = 0    # 0-based stacked row
other = 1  # 0-based stacked row

model.mean(row, form="sum")                  # row-level fixed mean
model.covariance(row, other, form="sum")     # row-level covariance
model.variance(row, form="sum")
model.variogram(row, other, kind="residual", form="sum")
model.covariogram(row, other, form="sum")
model.blup_cross_covariance(row, form="matrix")
```

Do not read `row = 0` as factor level `0`, and do not describe `model.mean(row)`
as `E[Y_g]`. The block result is `model.block(factor, level).mean()`; the panel
entry result is `model.panel(factor, position).entry_mean(level, position)`.

## Deprecated helpers

The former `group_*` and `time_*` model helpers are still available as
deprecated adapters that forward to `model.panel(...)`. New code should use
`model.block(...)` and `model.panel(...)`; the legacy helpers emit
`DeprecationWarning` and will be removed in a later release.

## Limitations

- The moment engine supports linear expressions in model atoms. Nonlinear or
  unsupported vector combinations raise `UnsupportedMoment`.
- `ModelSymbols` returns ordinary SymPy expressions. It is not a
  `sympy.stats` random-variable system and does not construct distributions.
- Block covariances, entry covariances, and `Cov(Y_g, Y_h) = 0` require a
  single grouping factor. Crossed random effects remain additional terms. With
  multiple grouping factors, block and entry covariance raise `NotDefined`
  rather than dropping other factors' random terms.
- A rectangular `Y` and position columns require a balanced panel. `Y_g`
  remains a block vector for unbalanced panels but is not a row of a
  rectangular matrix; scalar entries `Y_{gp}` remain defined, and ragged
  position columns are rejected with `NotDefined` rather than filled with
  invented positions.
