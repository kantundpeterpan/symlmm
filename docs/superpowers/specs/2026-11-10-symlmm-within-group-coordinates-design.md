# Within-Group Coordinates, Delta Resolution, and Nugget Effects

**Status:** accepted for implementation
**Issue:** #1 — full two-level `(group, occasion)` index model is future work
**Scope:** M2.6; removes the remaining row-index/level-index conflation before M3.

## Goal

Make the observation row, group level, within-group occasion, and coordinate
distinct concepts. Expressions must show `X[g, k]`, not `X[i]` or a parallel
`subject[i]`/`time[i]` pair. `KroneckerDelta` stays in the general expression
and resolves correctly for same-group and cross-group queries. Coordinate names
are generic, and residual structures gain optional nuggets.

## Index model A

`i` is only the observation-row cursor and the arity index for queries and
matrices. It is not a group level and not a coordinate index.

```python
g = model.group(i, "subject")        # level index g_subject[i]
k = model.occasion(i, "subject")     # within-group occasion k_subject[i]
x = model.coordinate(i, "visit_time", within="subject")
# visit_time[g_subject[i], k_subject[i]]
```

The distinction is deliberate and temporary:

- Group levels are a separate index space from row positions.
- Occasions are a separate index space within each group.
- A coordinate is indexed by `(group, occasion)`, never by the row index.
- Global row-level coordinates remain available as
  `model.coordinate(i, "timestamp")` -> `timestamp[i]`.

`src/symlmm/model.py` will carry a module-level note naming index model A and
pointing at issue #1 for the full two-level response index
`y[g, k]`, `X[g, k]`, `b[g]`.

## Accessors

| Accessor | Symbolic | Dense |
|---|---|---|
| `group(i, factor)` | `g_<label>[i]` | integer level code |
| `occasion(i, factor)` | `k_<label>[i]` | integer occasion code |
| `coordinate(i, name)` | `name[i]` | row value |
| `coordinate(i, name, within=factor)` | `name[group(i,factor), occasion(i,factor)]` | row value, validated as nested |

Dense occasion codes are taken from an explicit occasion column when supplied;
otherwise they are derived by ranking unique coordinate values within each
group, falling back to row order when no coordinate is declared. Dense
within-group coordinates validate that every `(group, occasion)` pair maps to
one value and raise `ModelError` otherwise.

`coordinate_name` replaces `time_name`. Calling `coordinate(i)` without a
canonical coordinate raises `ModelError`; there is no hardcoded `"t"` fallback.

## Coordinate rename

The public concept is `coordinate`, not `time`:

- `model.time` -> `model.coordinate`
- `time_name` -> `coordinate_name`
- `LMM(time=...)` -> `LMM(coordinate=...)`
- `RandomSlope(group, time, cov)` -> `RandomSlope(group, coordinate, cov)`
- `AR1(group, time, ...)` / `Toeplitz(group, time, ...)` -> `coordinate`
- `icc(at={"time": ...})` -> `icc(at={"coordinate": ...})`

No `time` compatibility alias. This is a 0.2.0 breaking rename; the package is
pre-1.0 and the old name is the source of the hardcoded coordinate assumption.

## Delta resolution

The delta is not removed from the model:

```text
default:  rho**Abs(X[g_i,k_i] - X[g_j,k_j]) * sigma2 * delta(g_i, g_j)
within:   rho**Abs(X[g,k_i] - X[g,k_j]) * sigma2
```

- Default queries keep `KroneckerDelta` for the cross-group expression.
- `within="subject"` collapses the factor's level index for `i` and `j` to one
  common level, so the delta resolves to `1`. It is resolved, not deleted from
  the general IR.
- Dense concrete indices resolve deltas to `1`/`0` from actual labels.
- Cross-group symbolic expressions remain available through the default query;
  cross-group dense queries evaluate to `0` for within-group structures.

`within=` is accepted by `covariance`, `variance`, `correlation`, `variogram`,
and `covariogram`. `icc` uses within-group random-slope coordinates directly.

## Random slopes

`RandomSlope(group, coordinate, cov, within=True)` uses
`coordinate[group(i), occasion(i)]` by default. `within=False` preserves a
global-coordinate slope. The group gate remains.

## Spatial structures

`Exponential`, `Gaussian`, and `Matern` gain:

```python
Exponential(var=sigma2, range=phi, group="subject", coords=("x", "y"), nugget=omega2)
```

- `group=None` preserves global spatial covariance.
- `group="subject"` gates the correlated part with
  `KroneckerDelta(group(i), group(j))`.
- `coords` removes the hardcoded `x`/`y` names.
- Within-group spatial coordinates use `coordinate(..., within=group)`.

## Nugget effects

Additive `nugget: sp.Expr = 0` to `AR1`, `Toeplitz`, `CompoundSymmetry`,
`Exponential`, `Gaussian`, and `Matern`:

```text
C(0)   = var + nugget
C(h>0) = var * correlation(h)
```

The nugget is `nugget * KroneckerDelta(i, j)` and is not group-gated. Add
`nugget >= 0` when the parameter is present. `var` is the partial variance;
`nugget` is measurement error.

## Testing

- Accessor tests: group and occasion are distinct index spaces; within
  coordinate is two-level; no `"t"` fallback.
- Dense tests: occasion derivation, explicit occasion column, nesting
  validation, same-subject adjacent vs cross-subject.
- Covariance tests: AR(1)/Toeplitz use within coordinate; default delta; within
  resolution; dense cross-group zero.
- Spatial tests: group gate; cross-subject zero; generic coordinate names;
  nugget at zero lag.
- Nugget tests: zero-lag includes nugget; nonzero lag excludes it; constraint.
- Full gate: `uv run pytest -q && uv run ruff check &&
  uv run ruff format --check && uv run mypy src/symlmm`.

## Risks

- Index model A still uses a row-to-level mapping. It is a containment step, not
  the final index model; issue #1 tracks the full redesign.
- The rename is breaking. Update `PLAN.md`, the M3 plan, tests, and examples in
  the same change; bump the changelog to 0.2.0.
- Dense occasion derivation is order-dependent by nature. An explicit occasion
  column is the deterministic escape hatch.
- Nugget changes the zero-lag covariance for existing structures only when the
  parameter is supplied; default `0` preserves current behavior.
