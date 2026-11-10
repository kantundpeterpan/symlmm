# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-10

### Breaking

- Removed `RandomIntercept` and the covariance-bearing `RandomSlope`. Random
  effects are now declared with one joint `RandomEffects` block per grouping
  factor; `LMM.random` accepts `RandomEffects` blocks.
- A grouping factor may appear in at most one block. Duplicate-group blocks
  raise `ModelError`; combine intercepts and slopes into one block instead.

### Added

- `RandomEffects(group, cov, intercept=False, slopes=())` with `dim`,
  `coordinates()`, and one `Z G Z^T` block per group. Intercepts and slopes in
  the same block share a joint covariance, including intercept–slope and
  slope–slope terms.
- `RandomSlope(coordinate, within=True)` as a pure design definition with no
  covariance. Multiple coordinates per block are supported.
- `icc(at=...)` accepts a coordinate-name mapping such as
  `at={"time": 2, "dose": 3}` for multi-coordinate random effects.

### Changed

- A model with multiple random-slope coordinates has no single canonical
  coordinate; coordinates are inferred from all slope definitions.
- Old independent intercept and slope terms on one group must be combined into
  a joint block; unrelated grouping factors remain separate `RandomEffects`
  blocks.

### Fixed

- `design(i)` no longer double-counts a fixed slope whose name equals the
  declared coordinate.
- Derived dense occasion codes sort mixed value types deterministically
  instead of raising `TypeError`.
- Within-group symbols are cached per factor instead of being recreated on
  every query.
- Nugget constraints are applied consistently across AR(1), Toeplitz,
  compound-symmetry, and spatial covariances; zero nuggets emit no constraint.
- Added covariance docstrings plus coverage for grouped-spatial cross-group
  zeros and per-factor `occasion` mappings.

### Migration

| 0.2.0 | 0.3.0 |
|---|---|
| `RandomIntercept(group=G, cov=C)` | `RandomEffects(group=G, intercept=True, cov=C)` |
| `RandomSlope(group=G, coordinate=X, cov=C)` | `RandomEffects(group=G, intercept=True, slopes=(RandomSlope(X),), cov=C)` |
| `RandomSlope(group=G, coordinate=X, cov=C, within=W)` | `RandomEffects(group=G, intercept=True, slopes=(RandomSlope(X, within=W),), cov=C)` |

`RandomSlope` previously carried a 2×2 covariance for its implicit intercept
and slope, so the mapping above keeps both columns. When a 0.2.0 model used a
separate `RandomIntercept` and `RandomSlope` on the same group, fold the
standalone intercept variance into the joint block's intercept variance; the
old blocks cannot be reproduced by concatenating two new blocks.

## [0.2.0] - 2026-11-10

### Added

- `model.occasion(i, factor)` and two-level `model.coordinate(i, name, within=factor)`.
- `within=` resolution for covariance, variance, correlation, variogram, and covariogram.
- Grouped spatial covariances with `group=` and generic `coords=`.
- Additive `nugget=` on AR(1), Toeplitz, compound-symmetry, and spatial covariances.

### Changed

- Public `time` names replaced by `coordinate`; `model.time` and `time=` are removed.
- `RandomSlope` uses a within-group coordinate by default.
- Coordinate access without a canonical name raises `ModelError` instead of falling back to `t`.
- `LMM.design(i)` follows `Fixed` column order and count instead of unconditionally inserting the canonical coordinate.
- Conflicting residual/random coordinates raise `ModelError`; a single declared coordinate is inferred.
- Random-slope `icc` requires `at={"coordinate": ...}` instead of silently evaluating at observation 0.
- Dense-mode group labels are integer codes over the factor's label space, including nested and crossed composite factors; occasion codes are ranked within each group.
- Constraint messages name the actual parameter expression instead of using fixed `tau2`/`sigma2` text.

### Fixed

- Spatial covariances no longer return nonzero cross-group covariance when `group=` is supplied.
- Group levels, occasions, and coordinates no longer share the observation row index.
- `Heteroscedastic.entry` raises `ModelError` for concrete out-of-range indices instead of silently returning zero.
- `covariogram` delegates to `covariance` to keep the two quantities in sync.

## [0.1.0] - 2026-11-09

### Added

- Symbolic derivation of response moments for linear mixed models (LMMs),
  marginal models, and GEE working-covariance models.
- Conditional and marginal LMMs exposing both sum and matrix forms.
- GEE support for Gaussian/identity, Poisson/log, and Bernoulli/logit with
  configurable working correlations.
- First-class methods for mean, variance, covariance, correlation, ICC,
  variogram, covariogram, and BLUP with documented return types.
- Covariance structures: random intercept, random slope, AR(1), compound
  symmetry, unstructured, heteroscedastic, Toeplitz, spatial, nested, and
  crossed, including multiple random terms (`K > 1`).
- PD-safe parameterizations with `validate_parameters()` and tri-state
  positive-definiteness checks.
- Conditional → marginal conversion for all LMMs and marginal → conditional
  conversion for registered exact decompositions, with `NotDecomposable`
  otherwise.
- Evaluation (`subs`, `evaluate`, `lambdify`) and a dense mode used by
  numerical validation.
- Monte Carlo and reference-fixture validation of the core formulas.
- LaTeX and Markdown printers tested for sum, matrix, and combined output.
- Typed errors from `symlmm.errors`, including `MissingFormError`.
- Extension protocols for post-v0.1 work; no GLMM implementation.

[Unreleased]: https://github.com/kantundpeterpan/symlmm/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/kantundpeterpan/symlmm/releases/tag/v0.3.0
[0.2.0]: https://github.com/kantundpeterpan/symlmm/releases/tag/v0.2.0
[0.1.0]: https://github.com/kantundpeterpan/symlmm/releases/tag/v0.1.0
