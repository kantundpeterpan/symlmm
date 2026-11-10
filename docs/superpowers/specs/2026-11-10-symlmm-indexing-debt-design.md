# M2.5 Indexing-Debt Design

**Status:** accepted for implementation
**Scope:** M2 residual debt that is load-bearing for M3 marginal/GEE work.
**Source:** `docs/superpowers/plans/2026-11-09-symlmm-core.md` review findings and
`/tmp/opencode/symlmm-indexing-mechanics-2026-11-09.md`.

## Goal

Remove the remaining silent inconsistencies in index, design, time, level, and
ICC semantics before marginal and GEE models reuse the same machinery.

## Decisions

### D1 — `design()` derives from `Fixed`

`LMM.design(i)` returns the fixed-effect design row in `Fixed.names()` order:
`1` for the intercept when present, then `model.covariate(i, name)` for each
slope. It no longer unconditionally inserts the canonical time. `IndexedModel.design`
keeps its existing illustrative `[1, time]` behavior for models without a
`Fixed` specification.

Rejected alternative: remove `design()` from `LMM`. It is part of the public
surface in `PLAN.md` §3.0 and §5.0.

### D2 — one canonical time coordinate

`LMM` accepts an optional `time: str | None` keyword. Resolution is deterministic:

1. If `time` is supplied, it is canonical; any residual or random-term time
   coordinate that disagrees raises `ModelError`.
2. Otherwise, collect the distinct time coordinates declared by the residual and
   random terms. Exactly one is canonical. More than one raises `ModelError`.
   None leaves `time_name` as `None`.

Fixed slope names are covariates, not time declarations, and are not validated
against the canonical time. `Fixed(slopes={"time": ...})` intentionally remains
usable with a differently named residual time; the fixed design uses the name
the user declared.

Rejected alternative: always prefer the residual time and silently ignore a
different random-term time. That is the current order-dependent defect.

### D3 — levels are label-space cardinalities

`levels(factor)` is the number of distinct labels returned by
`group(i, factor)`. For symbolic models this is `m_<label>` with `:` sanitized
to `_`. For dense models it is an `int` computed from the actual labels.

Dense composite factors (`Nested`, `Crossed`) are converted to integer codes
over the composite label space before use in `matrix_z`, `matrix_g`, and
`dimensions`. This keeps `KroneckerDelta` numeric and makes dense nested/crossed
assembly agree with the symbolic composite count.

### D4 — ICC requires a time value for time-varying terms

If any random term is a `RandomSlope`, `icc` requires `at={"time": value}`.
Without it, `ModelError` is raised instead of silently evaluating at observation
`0`. Random-intercept-only ICC is unchanged and still needs no `at`.

The value may also be keyed by the canonical time name. The substitution covers
the random-slope design and the residual entry at the representative index.

Rejected alternative: return an expression in a fresh time symbol. The public
`icc(at=...)` contract is scalar evaluation.

### D5 — AR(1) time semantics are explicit

`AR1` keeps the plan-mandated covariance
`var * rho**Abs(t_i - t_j)`. Its docstring states that this is discrete-time
AR(1) for numeric unit-spaced time and the continuous-time exponential
covariance for irregular or non-unit spacing. Conflicting time declarations are
rejected by D2, so the residual cannot silently use a different time array.

A `step` parameter is deferred; it would add API surface not present in the
plan.

### D6 — minor consistency fixes

- Constraint messages use the actual parameter expression, e.g.
  `Scalar(s0).constraints()[0].message == "s0 > 0"`; no hardcoded `tau2` or
  `sigma2[k]` when the parameter is a different symbol.
- `covariogram` delegates to `moments.covariance`; it is the same quantity.
- `Heteroscedastic.entry(i, j, model)` raises `ModelError` when a concrete
  index is outside `0..len(variances)-1`. Symbolic indices still return the
  piecewise expression.

## Testing

- `LMM.design` column count and order match `Fixed.coefficients()` for
  intercept/no-intercept and arbitrary slope names.
- Dense `design(i)` matches `matrix_x()[i, :].T`.
- Conflicting residual/random time coordinates raise `ModelError`; a single
  declared coordinate is inferred; an explicit `time=` wins and validates.
- `levels` matches `dimensions` for plain, nested, and crossed factors in
  symbolic and dense modes.
- Dense nested/crossed `matrix_z` uses composite level codes and has the
  composite column count.
- ICC on a random-slope model raises without `at` and is time-free with `at`.
- Constraint messages name the actual symbols.
- Heteroscedastic out-of-range concrete indices raise `ModelError`.
- Full gate: `uv run pytest -q`, `uv run ruff check`,
  `uv run ruff format --check`, `uv run mypy src/symlmm`.

## Risks

- D2 is a behavior change for models that currently declare conflicting time
  coordinates. That conflict was previously silent and is the defect being
  removed.
- D4 is a behavior change for random-slope ICC without `at`. The old value was
  arbitrary; raising is the safer contract.
- D3 changes dense `group()` values for nested/crossed factors from raw child
  columns to integer composite codes. This is required for correct
  `KroneckerDelta` arithmetic and matches the integer-label contract.
