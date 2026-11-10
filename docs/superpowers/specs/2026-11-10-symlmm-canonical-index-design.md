# Canonical Group/Occasion Index Design

**Status:** proposed
**Resolves:** issue #1 — public canonical indexing for mixed models
**Supersedes:** row-index notation in `2026-11-10-symlmm-blackboard-design.md`

## Problem

The current public model uses one stacked observation row index `i`:

```python
model.mean(i)          # E[Y_i] for one row
model.covariance(i, j) # Cov(Y_i, Y_j) for two rows
```

In mixed-model notation that is wrong. `Y_i` is the response **vector** for
group/cluster/subject `i`; `Y_{it}` is the scalar response at occasion `t`
within group `i`. The stacked row index is an implementation detail.

## Canonical notation

The canonical object is a response matrix with groups as rows and occasions as
columns:

- `i` (or `g`) indexes the group/cluster/subject.
- `t` (or `j`) indexes the occasion.
- `Y` is the `m x T` response matrix for a balanced panel.
- `Y_i` is the row vector for group `i`.
- `Y_{.t}` is the column vector over groups at occasion `t`.
- `Y_{it}` is the scalar component.
- `b_i` is the group-level random-effect vector.
- `X_i`, `Z_i`, `R_i` are group blocks.
- `G` is the random-effect covariance.
- `V_i = Z_i G Z_i' + R_i`.
- `Cov(Y_i, Y_j) = 0` for `i != j`.
- `Cov(Y_{it}, Y_{is}) = [V_i]_{ts}`.
- For a balanced panel, `Cov(Y_{.t}, Y_{.s}) = diag_i([V_i]_{ts})`.

For unbalanced panels the matrix `Y` is not rectangular. Group rows `Y_i` and
scalar entries `Y_{it}` remain defined; occasion columns are defined over the
groups that actually have that occasion, or are rejected as ragged.

## Model-defined index symbols

The model owns the canonical index symbols; they are not hardcoded in the
response accessor:

```python
IndexSpec(group=sp.Symbol("i", integer=True), occasion=sp.Symbol("t", integer=True))
```

`LMM(..., index=IndexSpec(...))` stores:

- `model.group_index`
- `model.occasion_index`

If omitted, the model uses default index symbols, but `Y[...]` always dispatches
against the stored symbols, never against a hardcoded `"t"`. Group and occasion
symbols must be distinct; equal symbols raise `ModelError`.

## Public model API

Add canonical group-level and occasion-level methods while retaining the
stacked engine internally:

```python
model.group_size(i)
model.occasion_count()              # T, balanced panels
model.response_matrix()             # Y, m x T, balanced panels
model.response(i)                   # Y_i row vector
model.time_response(t)              # Y_{.t} column vector
model.response(i, t)                # Y_{it} scalar

model.group_mean(i)                 # X_i beta
model.group_covariance(i)           # V_i
model.group_cross_covariance(i, j)  # zero for i != j
model.time_mean(t)                  # E[Y_{.t}]
model.time_covariance(t, s)         # Cov(Y_{.t}, Y_{.s})
model.entry_mean(i, t)
model.entry_covariance(i, t, s)
model.entry_cross_covariance(i, t, j, s)

model.random_effect(i)              # b_i vector
model.fixed_design(i)               # X_i
model.random_design(i)              # Z_i
model.residual_covariance(i)        # R_i
```

The existing `mean(i)`, `covariance(i, j)` observation-row methods remain
available as stacked implementation/query methods but are no longer the
canonical public notation. They will be documented as row-level internals or
deprecated in a later release.

## Symbolic representation

For a balanced panel, `Y` is represented by an `m x T` `MatrixSymbol`:

```python
Y = sp.MatrixSymbol("Y", m, T)
Y_i = Y[i, :]
Y_t = Y[:, t]
Y_it = Y[i, t]
```

`Y_i` is the group row vector; `Y_{.t}` is the occasion column vector. For
unbalanced panels, group rows are represented by `MatrixSymbol(f"Y_{i}", n_i, 1)`;
occasion columns are rejected or represented over the observed groups only.

`b_i` is a `MatrixSymbol` of length `q`. `X_i`, `Z_i`, `R_i` are matrix symbols
of the appropriate group shapes. Dense mode returns concrete matrices.

## Blackboard symbols

`ModelSymbols` is rebuilt around canonical indexing:

```python
syms = model.symbols()

syms.i                # model.group_index
syms.t                # model.occasion_index

syms.Y[syms.i]        # Y_i row vector
syms.Y[syms.t]        # Y_{.t} column vector
syms.Y[i]             # row when i == model.group_index
syms.Y[t]             # column when t == model.occasion_index
syms.Y[i, t]          # Y_{it} scalar
syms.Y[i, :]          # row slice
syms.Y[:, t]          # column slice

syms.b[syms.i]        # b_i vector
syms.b[syms.i][a]     # component
syms.X(syms.i)        # X_i matrix
syms.Z(syms.i)        # Z_i matrix
syms.R(syms.i)        # R_i matrix
syms.G                # G
syms.beta             # beta vector
```

`Y[t]` is robust when `t` is the model-defined occasion symbol. `Y[0]` is
ambiguous and raises `ModelError`; use `Y[:, 0]` for a column or `Y[0, :]` for a
row.

Moments:

```python
syms.E(syms.Y[i])                  # X_i beta (row vector)
syms.Var(syms.Y[i])                # V_i (matrix)
syms.Cov(syms.Y[i], syms.Y[j])     # zero matrix for i != j
syms.E(syms.Y[:, t])               # E[Y_{.t}] (column vector)
syms.Var(syms.Y[:, t])             # diag_i([V_i]_{tt})
syms.Cov(syms.Y[:, t], syms.Y[:, s])  # diag_i([V_i]_{ts})
syms.E(syms.Y[i, t])               # scalar
syms.Var(syms.Y[i, t])             # V_i[t, t]
syms.Cov(syms.Y[i, t], syms.Y[i, s])  # V_i[t, s]
```

Conditional and total variance use matrices:

```python
syms.Var(syms.Y[i], given=syms.b[i])       # R_i
syms.E(syms.Y[i], given=syms.b[i])         # X_i beta + Z_i b_i
syms.Var(syms.E(syms.Y[i], given=syms.b[i]))  # Z_i G Z_i'
syms.E(syms.Var(syms.Y[i], given=syms.b[i]))  # R_i
syms.total_variance(syms.Y[i], given=syms.b[i])  # V_i
```

## Compatibility

- The stacked observation engine remains for matrix assembly, dense mode,
  BLUP, variograms, and existing internal tests.
- New canonical methods are the public teaching/API surface.
- The row-level methods are marked as implementation-oriented; deprecation is a
  later change.
- No `Y[i]` row-scalar behavior in the blackboard facade.

## Testing

- Group sizes and canonical response vectors.
- Group mean/covariance blocks and zero cross-group covariance.
- Entry moments `Y_{it}` and `Y_{is}`.
- Dense group sizes and matrices.
- Blackboard vector/matrix moments.
- Total variance decomposition for vectors.
- Existing stacked tests remain green.

## Risks

- Scope is large: canonical indexing touches model symbols, moments, dense mode,
  blackboard, and docs.
- Group-size symbols can make matrix expressions heavier; tests should use small
  concrete groups for numerical checks.
- The old row API must remain functional until a deliberate deprecation.
- Multiple grouping factors need a primary grouping definition for `Y_i`;
  crossed random effects remain additional terms.
