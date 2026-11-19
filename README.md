# symlmm

Symbolic derivation of response moments for linear mixed models, marginal
models, and GEE working-covariance models. The package does not fit models; it
produces reproducible analytic formulas in sum and matrix notation.

The implemented package provides:

- A factor-first layer: `Group`, `Nested`, and `Crossed` declarations resolve
  to `Factor` objects; `model.block(factor, level)` returns a `Block` with
  design, residual, random-effect, mean, and covariance quantities.
- Panel views: `model.panel(factor, position)` returns a `PanelView` over a
  factor and an optional within-block `Position` axis, with entry and
  position-column moments.
- Term-level decomposition: `model.terms()`, `model.covariance_terms(a, b)`,
  and partial covariance/correlation helpers.
- A blackboard facade: `panel.symbols()` returns `ModelSymbols` for
  textbook-style expressions such as `Y_g`, `b_g`, `E(...)`, and `Var(...)`.
- Covariance structures, dense evaluation, BLUPs, variograms, and symbolic or
  numeric moment evaluation.

The former `group_*` and `time_*` model helpers remain deprecated adapters that
forward to the factor-first API. See [CHANGELOG.md](CHANGELOG.md) for the
migration notes.

A worked tutorial with a hierarchical model, a longitudinal model, symbolic
math, and plots is in [docs/tutorials/factor-first.md](docs/tutorials/factor-first.md).

## Example

```python
import sympy as sp
import symlmm as sl

subject = sl.Group("subject")
visit = sl.Position("time")

b0, b1 = sp.symbols("b0 b1")
tau2 = sp.Symbol("tau2", positive=True)
sigma2 = sp.Symbol("sigma2", positive=True)
rho = sp.Symbol("rho", positive=True)

model = sl.LMM(
    fixed=sl.Fixed(intercept=b0, slopes={sl.Covariate("time"): b1}),
    random=[sl.RandomEffects(group=subject, intercept=True, cov=sl.Scalar(tau2))],
    residual=sl.AR1(
        group=subject,
        coordinate=sl.Covariate("time"),
        var=sigma2,
        rho=rho,
    ),
    coordinate="time",
    n=4,
)

dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
panel = dense.panel(subject, visit)
block = dense.block(subject, 0)

assert panel.response_matrix().shape == (2, 2)
assert panel.entry_mean(0, 1) == b0 + b1
assert block.mean() == sp.Matrix([[b0], [b0 + b1]])
assert block.covariance()[0, 1] == tau2 + rho * sigma2

syms = panel.symbols()
assert syms.E(syms.Y[0, :]) == block.mean().T
assert syms.Var(syms.Y[0, :]) == block.covariance()
```

Run `pytest` for the test suite and `mypy src/symlmm` for static checks.
