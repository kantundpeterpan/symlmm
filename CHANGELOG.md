# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/kantundpeterpan/symlmm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kantundpeterpan/symlmm/releases/tag/v0.1.0
