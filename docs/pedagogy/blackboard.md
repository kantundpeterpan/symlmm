# Blackboard-style moments with canonical indexing

`symlmm` derives symbolic moments; it does not fit models. This page shows the
blackboard facade after the canonical group/occasion indexing change. The same
model can be read in textbook notation: `i` indexes the
group/cluster/subject, `t` indexes the occasion, `Y_i` is the group response
vector, and `Y_{it}` is the scalar response at occasion `t` in group `i`.

The canonical notation is:

```text
i = group/cluster/subject
t = occasion

Y_i     = response vector for group i
Y_{it}  = scalar response at group i, occasion t
b_i     = random-effect vector for group i
X_i     = fixed design block for group i
Z_i     = random design block for group i
R_i     = residual covariance block for group i
G       = random-effect covariance
V_i     = Z_i G Z_i' + R_i

Cov(Y_i, Y_j) = 0              for i != j
Cov(Y_{it}, Y_{is}) = [V_i]ts
```

For a balanced panel, `Y` is the `m x T` matrix with groups as rows and
occasions as columns. `Y_i = Y[i, :]` is a row vector, `Y[:, t]` (also written
`Y[t]`) is the occasion column, and `Y_{it}` is the scalar entry. For an
unbalanced panel, `Y_i` remains the group response vector but is not a row of a
rectangular `Y` matrix; `Y_{it}` remains defined, while occasion columns are
not.

Python examples use 0-based indices. Rendered mathematical formulas use 1-based
bounds and state the conversion when they show index ranges.

The stacked observation-row index used by the matrix engine is an
implementation detail. It is not `Y_i` and it is not the canonical public
index.

## Model definitions carry the index symbols

Index symbols belong to `Group`, `Nested`, `Crossed`, and `Covariate`
definitions. The model derives `group_index` and `occasion_index` from them;
nothing hardcodes `i` or `t`.

```python
import sympy as sp
import symlmm as sl

i = sp.Symbol("i", integer=True, nonnegative=True)
t = sp.Symbol("t", integer=True, nonnegative=True)

subject = sl.Group("subject", index=i)
visit = sl.Covariate("time", index=t)

tau0 = sp.Symbol("tau0", positive=True)
tau1 = sp.Symbol("tau1", positive=True)
rho_b = sp.Symbol("rho_b", real=True)
sigma2 = sp.Symbol("sigma2", positive=True)
rho = sp.Symbol("rho", positive=True)
beta0, beta1 = sp.symbols("beta0 beta1", real=True)

model = sl.LMM(
    fixed=sl.Fixed(intercept=beta0, slopes={visit: beta1}),
    random=[
        sl.RandomEffects(
            group=subject,
            intercept=True,
            slopes=(sl.RandomSlope(visit),),
            cov=sl.Unstructured.from_sd_corr(sd=[tau0, tau1], corr=rho_b),
        )
    ],
    residual=sl.AR1(group=subject, coordinate=visit, var=sigma2, rho=rho),
)

assert model.group_index == i
assert model.occasion_index == t
```

Passing `index=` is optional. When it is omitted, a `Group` derives its index
from its label and a `Covariate` derives it from its name. Canonical examples
pass it explicitly so the intended model notation is visible. A model rejects
equal group and occasion index symbols with `ModelError`.

`RandomSlope(coordinate=...)` takes a `Covariate` definition, not a bare
string. A bare string is accepted only as a compatibility shim and normalized
to `Covariate(name)`; canonical examples pass the definition.

## Which group and occasion are canonical?

The model resolves the canonical indices by fixed precedence:

1. The primary group is the first `RandomEffects` block's `Group`; otherwise
   the residual covariance's group definition; otherwise the first registered
   grouping definition.
2. For the selected coordinate name, the canonical occasion is the residual
   covariance's coordinate `Covariate`; otherwise the fixed-effect slope
   `Covariate` with the same name; otherwise a random-slope `Covariate` when
   exactly one distinct coordinate exists; otherwise, when `coordinate=` was
   supplied, a synthesized `Covariate(name)`; otherwise there is no canonical
   occasion (`occasion_index is None`).

The coordinate name is selected from an explicit `LMM(coordinate=...)` when
supplied, otherwise from the residual coordinate, otherwise from the single
distinct random-slope coordinate. An explicit `coordinate=` that conflicts
with a residual coordinate raises `ModelError`. Supplying `coordinate=` never
produces "no occasion": when no residual, fixed, or random-slope definition
matches, the model synthesizes `Covariate(coordinate_name)`.

Only the primary group defines `Y_i`, `V_i`, and the canonical group block.
With multiple grouping factors, `Y_i` is still the primary group's response,
but block covariance that would combine factors is `NotDefined`.

## Canonical model API

These methods return SymPy expressions directly and do not take `form`:

```python
j = sp.Symbol("j", integer=True)
s = sp.Symbol("s", integer=True)

model.response_matrix()              # Y, m x T
model.group_count()                  # m
model.occasion_count()               # T
model.group_size(i)                  # n_i

model.response(i)                    # Y_i row vector (balanced)
model.time_response(t)               # Y[:, t] column vector
model.response(i, t)                 # Y_it scalar

model.fixed_design(i)                # X_i
model.random_design(i)               # Z_i
model.residual_covariance(i)          # R_i
model.random_effect(i)               # b_i

model.group_mean(i)                  # E[Y_i] = X_i beta
model.group_covariance(i)            # V_i = Z_i G Z_i' + R_i
model.group_cross_covariance(i, j)   # zero for i != j

model.entry_mean(i, t)
model.entry_covariance(i, t, s)              # [V_i]ts
model.entry_cross_covariance(i, t, j, s)     # zero for i != j

model.time_mean(t)                   # E[Y[:, t]]
model.time_covariance(t, s)          # Cov(Y[:, t], Y[:, s])
```

`j` and `s` above are ordinary SymPy symbols for a second group and a second
occasion. Occasion-column moments (`time_mean`, `time_covariance`) require a
concrete group count or dense mode; the fully symbolic arbitrary-`m` form raises
`NotDefined`.

## Blackboard symbols

`model.symbols()` returns `ModelSymbols`. Its indices are the model's own
indices: the facade sets `syms.i = model.group_index` and
`syms.t = model.occasion_index` when it is constructed.

```python
syms = model.symbols()

assert syms.i == model.group_index
assert syms.t == model.occasion_index

syms.Y[syms.i]              # Y_i row vector (balanced)
syms.Y[syms.t]              # Y[:, t] occasion column
syms.Y[syms.i, syms.t]      # Y_it scalar

syms.Y[syms.i, :]           # row slice
syms.Y[:, syms.t]           # column slice

syms.b[syms.i]              # b_i vector
syms.X(syms.i)              # X_i
syms.Z(syms.i)              # Z_i
syms.R(syms.i)              # R_i
syms.G                      # G
syms.beta                   # beta
```

`Y[t]` is unambiguously the occasion column when `t` is the model-defined
occasion symbol. A bare integer such as `Y[0]` is ambiguous and raises
`ModelError`; use `Y[0, :]` for a group row or `Y[:, 0]` for an occasion
column.

## Group-level moments

The facade delegates group blocks to the canonical model API:

```python
Y_i = syms.Y[syms.i]

assert syms.E(Y_i) == model.group_mean(syms.i)
assert syms.Var(Y_i) == model.group_covariance(syms.i)
assert syms.Cov(Y_i, Y_i) == model.group_covariance(syms.i)

Y_it = syms.Y[syms.i, syms.t]

assert syms.E(Y_it) == model.entry_mean(syms.i, syms.t)
assert syms.Var(Y_it) == model.entry_covariance(syms.i, syms.t, syms.t)
assert syms.Cov(Y_it, syms.Y[syms.i, s]) == model.entry_covariance(
    syms.i, syms.t, s
)
```

In matrix terms, `Cov(Y_i) = V_i = Z_i G Z_i' + R_i`. Across groups,
`Cov(Y_i, Y_j) = 0` for `i != j`:

```python
j = sp.Symbol("j", integer=True)
assert model.group_cross_covariance(syms.i, j).is_zero_matrix
```
The blackboard facade itself is built around one model group index. Use the
model's `group_cross_covariance` when a second group symbol is needed.

## Conditional moments and the law of total variance

Condition on the group random-effect vector `b_i`:

```python
b_i = syms.b[syms.i]

E_Y_given_b = syms.E(syms.Y[syms.i], given=b_i)
V_Y_given_b = syms.Var(syms.Y[syms.i], given=b_i)

# X_i beta + Z_i b_i
# R_i
```

The law of total variance gives the marginal group covariance:

```python
V_Y = syms.Var(syms.Y[syms.i])
between = syms.Var(syms.E(syms.Y[syms.i], given=b_i))
within = syms.E(syms.Var(syms.Y[syms.i], given=b_i))

assert sp.simplify(V_Y - between - within).is_zero_matrix
assert sp.simplify(syms.total_variance(syms.Y[syms.i], given=b_i) - V_Y).is_zero_matrix
```

`between` is the random-effect contribution `Z_i G Z_i'`; `within` is the
residual contribution `R_i`.

## What remains stacked-engine internal

The engine still stacks all observations into one vector `y` for matrix
assembly, dense evaluation, variograms, and BLUP. The following methods take
the stacked row cursor and are not canonical `Y_i` notation:

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

Do not read `row = 0` as group `0`, and do not describe `model.mean(row)` as
`E[Y_i]`. The canonical group result is `model.group_mean(i)`; the canonical
entry result is `model.entry_mean(i, t)`.

## Limitations

- The moment engine supports linear expressions in model atoms. Nonlinear or
  unsupported vector combinations raise `UnsupportedMoment`.
- `ModelSymbols` returns ordinary SymPy expressions. It is not a
  `sympy.stats` random-variable system and does not construct distributions.
- Group blocks `V_i` and `Cov(Y_i, Y_j) = 0` require a single primary grouping
  factor. Crossed random effects remain additional terms. With multiple
  grouping factors, block covariance that would combine factors is
  `NotDefined`.
- A rectangular `Y` and occasion columns require a balanced panel. `Y_i`
  remains a group vector for unbalanced panels but is not a row of a
  rectangular matrix; scalar entries `Y_{it}` remain defined, and ragged
  occasion columns are rejected with `NotDefined` rather than filled with
  invented occasions.
