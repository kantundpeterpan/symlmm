# symlmm — Implementation Plan

**Status:** plan revised after gap review; M0.5 feasibility spike pending; implementation not started  
**Date:** 2026-11-09  
**Package:** `symlmm`  
**Repository:** `/workspaces/symlmm`  
**Target runtime:** Python 3.14.5 (3.15 stable is not available in this environment; 3.15.0b1 is beta)  
**Documentation:** Quarto  
**Scope:** symbolic LMMs, marginal models, and GEE working-correlation models

---

## 1. Vision

`symlmm` is a SymPy-based Python package for deriving closed-form expressions for
response moments under linear mixed models (LMMs), marginal/population-averaged
models, and GEE working-covariance specifications.

Given a symbolic model specification, it returns:

- expectation;
- variance;
- covariance and correlation;
- ICC / variance partition coefficients;
- variograms and covariograms;
- conditional and marginal versions of the above;
- BLUP moments.

Every quantity is available in **sum notation**, **matrix notation**, or both.

The package does **not** fit models. It is a symbolic derivation and documentation
engine intended for statisticians, methodologists, teachers, and authors who need
reproducible analytic formulas.

---

## 2. Locked design decisions

1. **Symbolic matrix dimensions.** Core matrix expressions use symbolic shapes
   `(n, p, q)` via `MatrixSymbol`. Concrete matrices are opt-in through an
   explicit dense mode.
2. **`form` is required.** Every public moment, covariance, variogram, summary,
   and LaTeX call must receive `form="sum"`, `form="matrix"`, or `form="both"`.
   There are no defaults. Missing `form` raises `MissingFormError`.
3. **Dual notation.** Sum notation and matrix notation are generated from one
   shared model IR. `form="both"` means sum + matrix, not conditional + marginal.
4. **Conditional → marginal is always available for LMMs.** The induced marginal
   covariance is `V = Z G Z^T + R`.
5. **Marginal → conditional only when exact.** A direct marginal model returns a
   conditional representation only when an exact decomposition exists
   (provenance or a known exact registry match). Otherwise it raises
   `NotDecomposable`. No silent or approximate conversion.
6. **GEE is in v0.1.** GEE models provide marginal response moments under a
   working covariance. They have no conditional counterpart and no fitting or
   sandwich-estimator inference in v0.1.
7. **GLMM marginalization is future work.** Define extension protocols only.
8. **Observation index.** Every model exposes one observation index `i` over
   the stacked response. Indices are 0-based internally and 1-based in rendered
   documentation. Group labels, times, coordinates, and covariates are read
   through accessors (`model.group`, `model.time`, `model.covariate`,
   `model.design`); users never build them by hand.
9. **Two-layer matrix representation.** Matrix form is dimensionally abstract:
   `V`, `G`, `R`, `Z`, `X` are `MatrixExpr` objects with symbolic shapes.
   Structure is exposed only through entry/sum form or dense mode. This avoids
   unsupported symbolic block-diagonal and concatenation operations while
   keeping both notations derived from one IR.
10. **PD-safe parameterization.** Covariance structures default to
    positive-definite-by-construction parameterizations (Cholesky or
    SD+correlation). Raw matrix constructors are dense-mode only. Every
    structure exposes `constraints()`, and `is_positive_definite` is explicit
    tri-state (`True`, `False`, `None`).
11. **Decomposition is bounded.** `is_decomposable` matches a finite template
    registry and solves parameter maps; it never attempts general symbolic
    equivalence. Failure returns `NotDecomposable` with a reason and a
    suggested alternative.
12. **Evaluation contract.** Models expose `subs`, `evaluate`, `lambdify`, and
    `dense`. Validation and examples use these; no ad-hoc `lambdify` in tests.
13. **Complete public surface.** Correlation, ICC, covariogram, BLUP, and
    conditional moments are first-class methods, not implied by `summary`.
    Variograms require an explicit `kind`.
14. **Namespace discipline.** One public name per concept; covariance and
    working-correlation classes are distinct (`AR1` vs `WorkingAR1`); `__all__`
    is exhaustive and snapshot-tested.
15. **Form/arity legality.** Each query declares which arities accept which
    forms. Invalid combinations raise `FormArityError`; invalid values raise
    `InvalidFormError`.

---

## 3. Mathematical core

### 3.0 Index and data model

The response vector is stacked over observations. A single index `i` addresses a
row; all model quantities are functions of `i` and `j`. Internal sums are
0-based to match SymPy; rendered documentation converts bounds to 1-based.

Every model provides:

```python
i, j = model.indices()
model.group(i, "subject")     # integer group label g(i)
model.time(i)                 # t_i
model.covariate(i, "time")    # x_i component
model.design(i)               # [1, t_i]
```

Grouping factors are declared explicitly:

- `sl.Group("subject")` — one integer label per observation;
- `sl.Nested("visit", within="subject")` — composite label `subject:visit`;
- `sl.Crossed("subject", "item")` — composite label `subject:item`.

Each composite factor yields one integer label, so the sum formula in §3.4 uses a
single `KroneckerDelta` per factor. GEE models add `model.cluster(i)` and
`model.occasion(i)`; these are aliases over the same observation index, not a
second index space.

### 3.1 Conditional LMM

\[
y = X\beta + Zb + \varepsilon,\qquad
b \sim N(0,G),\quad \varepsilon \sim N(0,R),\quad b \perp \varepsilon
\]

Symbolic matrices:

\[
X:n\times p,\quad \beta:p\times1,\quad
Z:n\times q,\quad G:q\times q,\quad
R:n\times n,\quad y:n\times1
\]

Moments:

\[
E[y\mid b]=X\beta+Zb,\qquad \operatorname{Cov}(y\mid b)=R
\]

\[
E[y]=X\beta,\qquad \operatorname{Cov}(y)=ZGZ^\top+R
\]

For multiple random terms:

\[
Z=[Z_1\ \cdots\ Z_K],\qquad
G=\operatorname{BlockDiag}(G_1,\dots,G_K)
\]

### 3.2 Marginal LMM

Two routes:

1. **Direct marginal model**

   \[
   y \sim N(X\beta,\;V^m(\alpha))
   \]

   with `V^m` built from a `MarginalCovSpec`.

2. **Induced marginal model**

   \[
   V^m = ZGZ^\top + R
   \]

Both expose the same moment API.

The variance decomposition is made explicit:

\[
V^m=\underbrace{\operatorname{Var}(E[y\mid b])}_{ZGZ^\top}
+\underbrace{E[\operatorname{Var}(y\mid b)]}_{R}
\]

### 3.3 GEE marginal model

GEE specifies a marginal mean and a working covariance directly:

\[
\mu_{ij}=g^{-1}(x_{ij}^\top\beta)
\]

\[
\operatorname{Var}(y_{ij})=\phi\,v(\mu_{ij})
\]

\[
V_i=\phi\,A_i^{1/2}R_i(\alpha)A_i^{1/2},\qquad
A_i=\operatorname{diag}\!\big(v(\mu_{ij})\big)
\]

\[
V=\operatorname{BlockDiag}(V_1,\dots,V_m)
\]

Working correlations:

- independence;
- exchangeable / compound symmetry;
- AR(1);
- Toeplitz;
- unstructured;
- spatial exponential, Gaussian, and Matérn.

GEE in v0.1 supports symbolic response moments for:

- Gaussian / identity;
- Poisson / log;
- Bernoulli / logit.

Fitting, estimating equations, and sandwich covariance are out of scope for
v0.1 and remain explicit future work.

### 3.4 Sum notation

For a conditional LMM:

\[
\operatorname{Cov}(y_i,y_j)
=
\sum_{k=1}^{K}
\sum_{a=1}^{q_k}
\sum_{b=1}^{q_k}
z_{k,i,a}\,G_{k,ab}\,z_{k,j,b}\,
\delta\!\left(g_k(i),g_k(j)\right)
+
R_{ij}
\]

For a direct marginal model:

\[
\operatorname{Cov}(y_i,y_j)=V^m_{ij}
\]

For GEE:

\[
\operatorname{Cov}(y_{ij},y_{ik})
=
\phi\,
\sqrt{v(\mu_{ij})}\,
R_{i,jk}(\alpha)\,
\sqrt{v(\mu_{ik})}
\]

and zero across clusters.

Implementation uses `IndexedBase`, `Sum`, `KroneckerDelta`, and Euclidean
distance for spatial coordinates. All sums are 0-based internally; rendered
output converts bounds to 1-based. Group and time symbols come from model
accessors, so the same formula derives for nested and crossed factors after
label construction. Spatial distance is
\(\lVert s_i-s_j\rVert\), not a scalar absolute difference; `Abs` is used only
for one-dimensional time/coordinate lags.

Special cases must simplify automatically:

- random intercept:
  \[
  \tau^2\delta(g_i,g_j)+\sigma^2\delta(i,j)
  \]
- random slope:
  \[
  \big[\tau_{00}+\tau_{01}(t_i+t_j)+\tau_{11}t_it_j\big]\delta(g_i,g_j)
  +\sigma^2\delta(i,j)
  \]
- AR(1) residual:
  \[
  \sigma^2\rho^{|t_i-t_j|}\delta(g_i,g_j)
  \]

### 3.5 Matrix-to-sum bridge

SymPy already maps symbolic matrix entries to explicit sums:

```python
V = Z * G * Z.T + R
V[i, j]
# R[i,j] + Sum(
#     G[_i_1,_i_2] * Z[i,_i_1] * Z[j,_i_2],
#     (_i_1, 0, q - 1),
#     (_i_2, 0, q - 1),
# )
```

`matrix_to_sum` canonicalizes this result. `sum_to_matrix` assembles `X,Z,G,R`
from the model IR. Both directions share the same model specification.

`as_explicit()` is never called on symbolic dimensions. It is only used in the
explicit dense mode, which requires concrete observations.

### 3.6 Variograms

Semantics must be explicit.

Residual (mean-adjusted) variogram:

\[
\gamma^{\text{res}}_{ij}
=
\tfrac12\left(V_{ii}+V_{jj}-2V_{ij}\right)
\]

Matrix form:

\[
\gamma^{\text{res}}_{ij}
=
\tfrac12(e_i-e_j)^\top V(e_i-e_j)
\]

Raw variogram:

\[
\gamma^{\text{raw}}_{ij}
=
\tfrac12\left(\mu_i-\mu_j\right)^2
+
\tfrac12\left(V_{ii}+V_{jj}-2V_{ij}\right),
\qquad
\mu_i=g^{-1}\!\left(x_i^\top\beta\right)
\]

For the identity link this reduces to
\(\tfrac12[(x_i-x_j)^\top\beta]^2+\tfrac12(V_{ii}+V_{jj}-2V_{ij})\).
The non-identity form is required for Poisson/log and Bernoulli/logit GEE
variograms, which are in scope (§8.4).

Conditional variogram:

\[
\gamma^{\text{cond}}_{ij}
=
\tfrac12\left(R_{ii}+R_{jj}-2R_{ij}\right)
\]

Conditional raw form also includes the fixed/random mean difference:

\[
\tfrac12\left[
(x_i-x_j)^\top\beta+(z_i-z_j)^\top b
\right]^2
+
\tfrac12\left(R_{ii}+R_{jj}-2R_{ij}\right)
\]

Covariogram (the covariance itself, retained as a named quantity):

\[
C_{ij}=\operatorname{Cov}(y_i,y_j)=V_{ij}
\]

Variogram selection is explicit: `kind="raw"`, `kind="residual"`, or
`kind="conditional"`. There is no default.

Stationary specialization substitutes `t_i = t`, `t_j = t + h`.

### 3.7 BLUP moments

\[
E[b\mid y]=GZ^\top V^{-1}(y-X\beta)
\]

\[
\operatorname{Cov}(b\mid y)
=
\left(G^{-1}+Z^\top R^{-1}Z\right)^{-1}
\]

\[
\operatorname{Cov}(y_i,b_k)=z_{k,i}^\top G_k
\]

Dense mode exposes explicit `V`, `G`, `R`, and the BLUP expressions for small
concrete instances.

BLUP support is form-bounded:

- `blup_mean`, `blup_covariance`, and `blup_cross_covariance` are available in
  matrix form for every LMM;
- dense mode evaluates them for concrete instances;
- sum form is available only for structures with a closed-form inverse (RI,
  IID, and explicitly registered block structures); otherwise
  `UnsupportedForm` is raised.

---

## 4. Package architecture

```text
/workspaces/symlmm/
  pyproject.toml
  uv.lock
  .python-version
  README.md
  PLAN.md
  LICENSE
  CHANGELOG.md
  CITATION.cff
  src/symlmm/
    __init__.py         # exhaustive __all__
    py.typed
    errors.py           # public exception hierarchy
    symbols.py          # observation indices, parameter symbols
    specs.py            # Fixed, Random*, Group, Nested, Crossed
    constraints.py      # Constraint, PD tri-state checks
    covariances.py      # random/residual covariance structures
    forms.py            # Form, DualForm, EntryMap, bridges
    model.py            # LMM, index accessors, form dispatch
    moments.py          # mean/var/cov/corr/ICC, conditional moments
    variogram.py        # raw/residual/conditional + covariogram
    blup.py             # BLUP moments and form support
    evaluate.py         # subs, evaluate, lambdify, dense
    algebra.py          # compare_specs
    simplify.py         # canonicalization, rewrite rules
    printers.py         # LaTeX/Markdown for sum/matrix/both
    marginal/
      __init__.py
      model.py          # MarginalLMM
      covariance.py     # direct marginal specs
      induce.py         # InducedCovariance
      decompose.py      # exact decomposition registry + NotDecomposable
    gee/
      __init__.py
      model.py          # GEE
      families.py       # Family, Link, VarianceFunction
      correlation.py    # WorkingCorrelation classes
      covariance.py     # V_i assembly and block-diagonal V
  tests/
    unit/
    integration/
    fixtures/
  docs/
    _quarto.yml
    index.qmd
    getting-started.qmd
    model-specification.qmd
    covariance-structures.qmd
    moments.qmd
    variograms.qmd
    marginal-vs-conditional.qmd
    gee.qmd
    forms.qmd
    validation.qmd
    api/
  examples/
    *.qmd
  scripts/
  .github/workflows/ci.yml
```

### Module responsibilities

- `errors.py`: `SymlmmError` hierarchy; every public failure has a typed error.
- `symbols.py`: observation indices and parameter symbols with assumptions.
- `specs.py`: fixed effects, random terms, grouping factors, design vectors.
- `constraints.py`: `Constraint`, constraint collection, PD tri-state checks.
- `covariances.py`: covariance structures used by conditional and marginal
  models; PD-safe constructors.
- `forms.py`: `Form`, `DualForm`, `EntryMap`, arity validation, sum/matrix
  conversion.
- `model.py`: `LMM`, index accessors, required-form dispatch.
- `moments.py`: mean/variance/covariance/correlation/ICC, conditional moments.
- `variogram.py`: raw, residual, and conditional variograms; covariogram.
- `blup.py`: BLUP mean, covariance, and cross-covariance with form gating.
- `evaluate.py`: substitution, numeric evaluation, `lambdify`, dense mode.
- `marginal/`: direct and induced marginal models, exact decomposition.
- `gee/`: GEE families, links, working correlations, symbolic marginal moments.
- `algebra.py`: `compare_specs` and comparison results.
- `simplify.py`: canonicalization and rewrite rules.
- `printers.py`: LaTeX and Markdown output.
- Validation harnesses and reference fixtures live in `tests/validation/` and
  `scripts/`; they are not shipped in the package.

---

## 5. Public API

### 5.0 Conventions and contracts

**Index model.** Every model exposes a single observation index `i` over the
stacked response. Indices are 0-based internally; documentation displays
1-based sums and states the mapping. Group labels, times, coordinates, and
covariates are accessed through the model, never constructed by the user:

```python
i, j = model.indices()
model.group(i, "subject")     # integer group label g(i)
model.time(i)                 # t_i
model.covariate(i, "time")    # x_i component
model.design(i)               # [1, t_i]
```

Nested and crossed factors are declared in the spec (`sl.Nested(...)`,
`sl.Crossed(...)`). GEE models use the same observation index and add
`model.cluster(i)` and `model.occasion(i)`.

**Form and arity.** `form` selects notation, not conditioning. Valid
combinations are fixed:

| Query | No index | One index | Two indices |
|---|---|---|---|
| `mean` | matrix | sum or matrix element | invalid |
| `covariance` | matrix | invalid | sum or matrix element |
| `variance` | matrix | sum or matrix element | invalid |
| `correlation` | matrix | invalid | sum or matrix element |
| `variogram` | invalid | invalid | sum or matrix element |
| `covariogram` | invalid | invalid | sum or matrix element |
| `summary` | both | invalid | invalid |

`form="both"` on a whole-matrix query returns
`DualForm(sum=EntryMap(...), matrix=MatrixExpr)`, where `EntryMap` is a lazy
mapping `(i, j) -> expression`. Invalid arity raises `FormArityError`; an
invalid value raises `InvalidFormError`; a missing value raises
`MissingFormError`.

**Conditioning.** Marginal moments are the default (`mean`, `covariance`).
Conditional moments are explicit (`conditional_mean`, `conditional_variance`,
`conditional_covariance`). `form="both"` never means conditional + marginal.

### 5.1 Conditional LMM

```python
import sympy as sp
import symlmm as sl

tau0 = sp.Symbol("tau0", positive=True)
tau1 = sp.Symbol("tau1", positive=True)
rho_b = sp.Symbol("rho_b", real=True)      # |rho_b| < 1, enforced by constraints
sigma2 = sp.Symbol("sigma2", positive=True)
rho = sp.Symbol("rho", positive=True)      # 0 < rho < 1, enforced by constraints
beta0, beta1 = sp.symbols("beta0 beta1", real=True)

model = sl.LMM(
    fixed=sl.Fixed(intercept=beta0, slopes={"time": beta1}),
    random=[
        sl.RandomSlope(
            group="subject",
            time="time",
            cov=sl.Unstructured.from_sd_corr(sd=[tau0, tau1], corr=rho_b),
        )
    ],
    residual=sl.AR1(group="subject", time="time", var=sigma2, rho=rho),
)

i, j = model.indices()

model.mean(form="matrix")                        # X beta
model.mean(i, form="sum")                        # x_i^T beta
model.mean(i, form="both")                       # DualForm
model.conditional_mean(form="matrix")            # X beta + Z b
model.conditional_mean(i, form="sum")            # x_i^T beta + z_i^T b
model.covariance(form="matrix")                  # V
model.covariance(i, j, form="sum")               # expanded V_ij
model.covariance(i, j, form="matrix")            # V[i, j]
model.variance(form="matrix")                    # DiagonalOf(V)
model.variance(i, form="sum")                    # V_ii
model.correlation(form="matrix")                 # Corr(V)
model.correlation(i, j, form="sum")              # Corr(V)_ij
model.icc(at={"time": 0}, form="both")           # conditional VPC
model.variogram(i, j, kind="residual", form="both")
model.covariogram(i, j, form="both")
model.blup_mean(form="matrix")
model.blup_covariance(form="matrix")
model.blup_cross_covariance(i, form="matrix")
model.summary(form="both")
model.to_latex(form="sum")
```

### 5.2 Marginal LMM

```python
marg = sl.MarginalLMM(
    fixed=sl.Fixed(intercept=beta0, slopes={"time": beta1}),
    covariance=sl.AR1(group="subject", time="time", var=sigma2, rho=rho),
)

marg.mean(form="matrix")
marg.covariance(i, j, form="both")
marg.correlation(i, j, form="sum")
marg.variogram(i, j, kind="residual", form="sum")
marg.covariogram(i, j, form="sum")
marg.icc(at={"time": 0}, form="both")   # Gaussian identity only
marg.summary(form="both")
```

Induced marginal:

```python
induced = model.marginal(form="both")
induced.covariance(form="matrix")       # Z G Z^T + R
induced.covariance(i, j, form="sum")    # entrywise sum
```

Marginal → conditional:

```python
marg.is_decomposable()                  # True | False; bounded template match
marg.decomposition_reason()             # reason when False
marg.conditional(form="both")           # LMM, or raises NotDecomposable
```

### 5.3 GEE

```python
gee = sl.GEE(
    fixed=sl.Fixed(intercept=beta0, slopes={"time": beta1}),
    family=sl.Poisson(link="log"),
    correlation=sl.WorkingAR1(group="subject", time="time", rho=rho),
    dispersion=sp.Symbol("phi", positive=True),
)

gee.mean(form="matrix")
gee.mean(i, form="sum")
gee.variance(form="matrix")
gee.covariance(i, j, form="both")
gee.variogram(i, j, kind="raw", form="sum")
gee.covariogram(i, j, form="sum")
gee.summary(form="both")
gee.conditional()                       # raises NoConditionalModel
gee.icc(at={"time": 0}, form="both")    # raises NotDefined
```

### 5.4 Required `form`

Every public method and function requires `form`. Missing `form` raises
`MissingFormError`. Valid values:

- `"sum"` — indexed expression with `i`, `j`, deltas, and `Sum`;
- `"matrix"` — a `MatrixExpr` for whole-matrix queries or a `MatrixElement` for
  entry queries;
- `"both"` — `DualForm(sum=..., matrix=...)`.

An invalid string raises `InvalidFormError`; an invalid arity/form combination
raises `FormArityError` (table in §5.0). `form="both"` does not mean conditional
+ marginal; conditional/marginal comparison uses `compare_specs`.

### 5.5 Evaluation and dense mode

```python
model.subs({sigma2: 1.0, rho: 0.5, tau0: 1.0, tau1: 1.0, rho_b: 0.2})
model.evaluate({sigma2: 1.0, rho: 0.5}, form="matrix")
model.lambdify(args=["sigma2", "rho"], form="sum")
model.dense(observations=observations)
```

Rules:

- `subs` returns a new model and never mutates the original.
- `evaluate` requires every free symbol to be resolved; otherwise it raises
  `UnresolvedSymbols`.
- `lambdify` returns a callable over the declared arguments.
- `dense` requires concrete observation count, group labels, times, and
  covariates. It returns concrete `Matrix` objects and is the only mode where
  `as_explicit()` is called.
- Validation, examples, and notebooks use these methods; ad-hoc `lambdify`
  calls in tests are not allowed.

### 5.6 Summary and comparison

```python
@dataclass(frozen=True)
class Summary:
    mean: DualForm
    variance: DualForm
    covariance: DualForm
    correlation: DualForm | None
    icc: DualForm | None
    variogram: DualForm | None
    covariogram: DualForm | None
    blup_mean: DualForm | None
    blup_covariance: DualForm | None

sl.compare_specs(a, b, form="both", assumptions=None)
# -> ComparisonResult(equal: bool | None,
#                     residual: Expr | None,
#                     notes: tuple[str, ...])
```

`summary` populates only quantities defined for the model; absent fields are
`None`. GEE summaries omit conditional and BLUP fields. Marginal summaries omit
conditional fields unless the model is decomposable.

### 5.7 Errors

```python
class SymlmmError(Exception): ...
class FormError(SymlmmError): ...
class MissingFormError(FormError, TypeError): ...
class InvalidFormError(FormError, ValueError): ...
class FormArityError(FormError, ValueError): ...
class UnsupportedForm(FormError): ...
class ModelError(SymlmmError): ...
class NotDecomposable(ModelError): ...       # .reason, .suggestion
class NoConditionalModel(ModelError): ...
class NotDefined(ModelError): ...
class UnresolvedSymbols(ModelError): ...
class ParameterConstraintError(ModelError): ...  # .violations
```

---

## 6. Covariance structures

### 6.1 Random effects `G`

Random terms are specified in `specs.py`; covariance structure is separate:

- `RandomIntercept(group, cov=Scalar(tau2))`.
- `RandomSlope(group, time, cov=...)`, with `cov` a 2×2 `Unstructured`.

Covariance building blocks:

- `Scalar(tau2)`: 1×1 covariance, \(\tau^2>0\).
- `Diagonal(q)`: `IndexedBase("d", positive=True)`, \(d_k>0\).
- `Unstructured(q)`: default `from_cholesky(L)`; for symbolic `q`, an
  `IndexedBase("L", (q, q))` with lower-triangular structure and positive
  diagonal.
- `Kronecker`: product of positive-definite factors.
- `FactorAnalytic(k)`: future work; not in v0.1.

### 6.2 Residual / marginal covariance `R`

- `IID`: \(\sigma^2\delta(i,j)\), \(\sigma^2>0\).
- `CompoundSymmetry`: \(\rho>-1/(m_{\min}-1)\), where \(m_{\min}\) is the
  minimum cluster size.
- `AR1`: \(-1<\rho<1\).
- `Toeplitz`: `from_cholesky` or checked dense construction.
- `Unstructured`: `from_cholesky`.
- `Heteroscedastic`: `IndexedBase("sigma2", positive=True)`.
- `Exponential`: range \(>0\).
- `Gaussian`: range \(>0\).
- `Matern`: range \(>0\), \(\nu\in\{\tfrac12,\tfrac32,\tfrac52\}\), Euclidean
  distance \(\lVert s_i-s_j\rVert\).
- `BlockDiag`: explicit blocks or group-index blocks.
- `KroneckerProduct`: separable group × time structures.

Each structure exposes `constraints()`; `model.validate_parameters()` checks
them and raises `ParameterConstraintError` with the violated constraints.

### 6.3 GEE working correlations

Working correlations are separate public classes; they are not residual
covariances and carry no variance parameter:

- `WorkingIndependence`
- `WorkingExchangeable`
- `WorkingAR1`
- `WorkingToeplitz`
- `WorkingUnstructured`
- `WorkingSpatialExponential`
- `WorkingSpatialGaussian`
- `WorkingMatern`

All structures provide:

- `.entry(i, j, model)` for symbolic arbitrary-\(n\) work;
- `.dense(model)` for explicit small-\(n\) matrices;
- `.constraints()` and `.validate_parameters()`;
- parameter symbols with domain assumptions and PD-safe constructors.

---

## 7. Conditional ↔ marginal rules

### 7.1 Conditional → marginal: always

For any LMM, `model.marginal(form=...)` returns an induced marginal model with

\[
V^m = ZGZ^\top+R.
\]

Both sum and matrix forms are available.

### 7.2 Marginal → conditional: bounded exact decomposition

`MarginalLMM.is_decomposable()` runs a bounded algorithm; it never attempts
general symbolic equivalence:

1. If the covariance carries `InducedCovariance` provenance, return the stored
   source model.
2. Canonicalize \(V^m\) into group-block form: diagonal blocks plus per-factor
   block structure.
3. Match against the template registry.
4. Solve the parameter-map equations with `sympy.solve`; require a unique
   solution.
5. Check all constraints on the candidate conditional parameters.
6. Return `True` only if steps 1–5 succeed; otherwise record the first failure
   reason and return `False`.

Registry templates in v0.1:

- CS \((\rho\ge0)\) → RI + IID:
  \(\tau^2=\sigma^2\rho,\ \sigma_e^2=\sigma^2(1-\rho)\);
- block RI → RI + IID;
- block RS → RS + IID;
- diagonal → IID.

Refused cases:

- direct AR(1) marginal;
- general unstructured marginal;
- CS with \(\rho<0\);
- non-unique or non-identifiable variance splits;
- parameter solutions that violate constraints.

`NotDecomposable` carries `.reason` and `.suggestion`; the suggestion names
`InducedCovariance` or direct conditional specification. `decomposition_reason()`
returns the recorded reason without raising.

### 7.3 GEE

GEE models are inherently marginal. `GEE.conditional()` always raises
`NoConditionalModel`. GEE working covariance may be compared with an LMM-induced
marginal covariance when structures match, but it is not presented as induced by
a conditional model.

---

## 8. GEE module (v0.1)

### 8.1 Responsibilities

- Symbolic marginal mean, variance, covariance, correlation, and variogram under
  a working covariance.
- Family/link/variance-function protocol.
- Working-correlation structures.
- Block-diagonal cluster covariance.
- Equivalence checks against marginal LMMs for Gaussian identity.

### 8.2 Families and links

v0.1:

- Gaussian / identity;
- Poisson / log;
- Bernoulli / logit.

The protocol is extensible:

```python
class Family:
    def variance(self, mu): ...
    def link(self, mu): ...
    def inverse_link(self, eta): ...
```

### 8.3 Working covariance

\[
V_i=\phi A_i^{1/2}R_i(\alpha)A_i^{1/2}
\]

For Gaussian identity:

\[
A_i=I,\qquad V_i=\phi R_i(\alpha)
\]

This matches marginal CS and AR(1) LMMs when parameters align.

### 8.4 Scope boundary

Included in v0.1:

- response moments under working covariance;
- variograms;
- sum/matrix forms;
- equivalence tests.

Excluded from v0.1:

- fitting estimating equations;
- sandwich covariance for \(\hat\beta\);
- model selection;
- missing-data theory.

These are documented as future work.

---

## 9. Documentation with Quarto

### 9.1 Structure

```text
docs/
  _quarto.yml
  index.qmd
  getting-started.qmd
  model-specification.qmd
  covariance-structures.qmd
  moments.qmd
  variograms.qmd
  marginal-vs-conditional.qmd
  gee.qmd
  forms.qmd
  validation.qmd
  api/
```

### 9.2 Quarto configuration

- Project type: website.
- Sidebar navigation and search.
- `format: html` with a clean theme and LaTeX math.
- Executable code cells with the project environment.
- `quartodoc` for API reference pages generated from docstrings.
- Output directory: `docs/_site` (ignored by git).
- `quarto check` and `quarto render docs` in CI.

### 9.3 Required pages

- **Getting started:** install, import, first RI model, first marginal model.
- **Model specification:** fixed/random/residual/marginal/GEE specs.
- **Covariance structures:** formulas and parameter constraints.
- **Moments:** expectation, variance, covariance, correlation, ICC.
- **Variograms:** raw vs residual vs conditional; same-group vs cross-group.
- **Marginal vs conditional:** derivation, equivalence, decomposition limits.
- **GEE:** working covariance, families, equivalence with marginal LMMs.
- **Forms:** sum notation vs matrix notation, required `form`, conversion.
- **Validation:** Monte Carlo and reference fixtures.
- **API:** generated reference.

### 9.4 Examples

Quarto example pages:

1. Random intercept + IID: ICC and variogram.
2. Random slope: variance function and variogram.
3. RI + AR(1): conditional vs induced marginal.
4. Direct marginal AR(1) and CS.
5. CS marginal → RI conditional decomposition.
6. Nested and crossed random effects.
7. GEE exchangeable and AR(1) working correlations.
8. Dense mode and BLUP moments.

### 9.5 CI documentation job

- Install Quarto.
- Install Python 3.14 and project dependencies with `uv`.
- Run `quarto render docs`.
- Optionally publish to GitHub Pages when a remote is configured.

---

## 10. Testing and validation

### 10.1 Unit tests

- Forms: missing, invalid, and invalid-arity errors; `DualForm` consistency;
  `EntryMap` shape and lookup.
- Errors: hierarchy, stable `.reason`/`.suggestion`/`.violations` fields.
- Index model: accessors, nested/crossed labels, GEE cluster/occasion aliases.
- Constraints: PD tri-state (`True`, `False`, `None`), constraint collection,
  `ParameterConstraintError`.
- Covariance structures: `entry`/`dense` consistency for every G/R structure;
  PD-safe constructors; constraint checks.
- Shapes and assumptions: symbolic dimensions; `as_explicit()` is never called
  outside dense mode (monkeypatched guard).
- Sum/matrix consistency for every model and covariance structure.
- Closed forms: RI, RS, RI+RS, AR(1), CS, IID.
- Variograms: \(\gamma_{ii}=0\), symmetry, non-negativity for valid parameters,
  raw non-identity formula.
- Correlation bounds and positive variance.
- ICC: RI scalar, RS covariate-dependent, GEE raises `NotDefined`.
- Covariogram equals covariance.
- BLUP: matrix vs dense equality; sum-form gating raises `UnsupportedForm`.
- Printers: LaTeX/Markdown snapshot tests for sum, matrix, and both.
- `Summary` schema and `compare_specs` tri-state result.
- Evaluation: `subs` immutability, `evaluate` unresolved-symbol errors,
  `lambdify` callable behavior.

### 10.2 Integration tests

- Dense explicit mode equals `Z G Z^T + R` and `V^m`.
- Conditional → marginal covariance equals `Z G Z^T + R`.
- Marginal → conditional succeeds for CS/RI provenance and registry templates.
- Marginal → conditional raises for AR(1), negative-\(\rho\) CS, general US,
  non-unique splits; `NotDecomposable` reason/suggestion asserted.
- GEE Gaussian/identity matches marginal LMM working covariance.
- GEE Poisson/log and Bernoulli/logit moment expressions.
- BLUP moments match explicit dense formulas for small instances.
- Nested and crossed random effects match hand-derived covariance entries.
- Spatial exponential, Gaussian, and Matérn entries match hand-derived values.
- Multiple random terms (K>1) assemble correctly.
- Dense mode plus BLUP runs end to end.

### 10.3 Numerical validation

- Monte Carlo simulation with NumPy for every covariance structure.
- Fixed seeds; tolerance calibrated for sample size; tolerance recorded in the
  fixture file.
- Reference fixtures from `lme4`, `nlme`, and `geepack`; `glmmTMB` is dropped
  because GLMM is out of scope.
- Fixtures are JSON with a versioned schema: package version, R version,
  parameter values, observation design, and expected matrix/entries.
- Generator scripts live in `scripts/fixtures/`; no runtime R dependency in CI.
- Optional R integration tests run locally only.

### 10.4 Traceability matrix

| DoD item | Test file |
|---|---|
| Required `form` and error taxonomy | `tests/unit/test_forms.py`, `tests/unit/test_errors.py` |
| Sum/matrix consistency | `tests/unit/test_forms_bridge.py` |
| RI, RS, AR(1), CS, IID closed forms | `tests/unit/test_closed_forms.py` |
| Variograms and covariogram | `tests/unit/test_variograms.py` |
| Correlation, ICC | `tests/unit/test_moments.py` |
| BLUP | `tests/unit/test_blup.py`, `tests/integration/test_blup_dense.py` |
| Spatial structures | `tests/unit/test_spatial_covariances.py` |
| Nested and crossed factors | `tests/integration/test_nested_crossed.py` |
| Multiple random terms | `tests/integration/test_multiple_terms.py` |
| Marginal decomposition | `tests/unit/test_decompose.py`, `tests/unit/test_decompose_refusals.py` |
| GEE families and correlations | `tests/integration/test_gee.py` |
| Printers | `tests/unit/test_printers.py` |
| Evaluation and dense mode | `tests/unit/test_evaluate.py`, `tests/unit/test_dense.py` |
| Docs examples execute | `quarto render docs` in CI |

### 10.5 Quality gates

- `uv run pytest`
- `uv run ruff check`
- `uv run ruff format --check`
- `uv run mypy src/symlmm`
- coverage target ≥ 90%
- `quarto check`
- `quarto render docs`

---

## 11. Milestones and task plans

The executable plan is split by subsystem. Each task has exact files,
interfaces, failing-test-first steps, and a verification command.

- Core (M0–M2): `docs/superpowers/plans/2026-11-09-symlmm-core.md`
- Marginal and GEE (M3–M4):
  `docs/superpowers/plans/2026-11-09-symlmm-marginal-gee.md`
- Release (M5–M7): `docs/superpowers/plans/2026-11-09-symlmm-release.md`

| Milestone | Deliverable | Exit criterion | Est. |
|---|---|---|---|
| M0 | Scaffold, `pyproject.toml`, `uv`, Quarto skeleton, errors, forms, index accessors, constraints | `uv run pytest tests/unit/test_forms.py tests/unit/test_indices.py tests/unit/test_constraints.py` passes | 1 d |
| M0.5 | SymPy capability spike: `MatMul` entries, block-diagonal limits, elementwise functions, diagonal shapes, sum simplification, PD tri-state | `uv run pytest tests/unit/test_sympy_capabilities.py` passes; `docs/design/2026-11-09-sympy-capabilities.md` records outcomes and fallbacks | 1 d |
| M1 | Conditional LMM sum/matrix; RI, RS, diagonal, unstructured; matrix↔sum bridge; moments | `uv run pytest tests/unit/test_lmm_matrix.py tests/unit/test_forms_bridge.py tests/unit/test_moments.py` passes | 3 d |
| M2 | Residual/spatial structures; variograms; BLUP; dense mode; multiple terms; nested/crossed | `uv run pytest tests/unit tests/integration` passes for M2 modules | 3 d |
| M3 | Marginal LMM; induced covariance; bounded decomposition + refusal path | `uv run pytest tests/unit/test_decompose.py tests/unit/test_decompose_refusals.py` passes | 2 d |
| M4 | GEE families, working correlations, moments, equivalence tests | `uv run pytest tests/integration/test_gee.py` passes | 3 d |
| M5 | Canonicalization, parameter validation, LaTeX/Markdown printers, summary, `compare_specs` | printer and comparison snapshot tests pass | 2 d |
| M6 | Evaluation API, Monte Carlo validation, fixture schema/scripts, Quarto docs and examples | evaluation tests pass; `quarto render docs` succeeds | 3 d |
| M7 | CI matrix, packaging, typing, v0.1 release | CI green; `uv build`; tagged release | 1 d |

**Total: ~19 working days.** Estimates are provisional until the task plan is
reviewed and the M0.5 spike closes the SymPy assumptions.

---

## 12. Risks and mitigations

- **Python 3.14 target.** 3.15 stable is not available in this environment
  (only 3.15.0b1). Target 3.14.5; revisit 3.15 when a stable release and
  dependency wheels are available.
- **Sum canonicalization.** `MatrixElement` produces internal dummy indices.
  Normalize index names and bounds; collect `KroneckerDelta` terms.
- **Decomposition ambiguity.** Only exact or provenance-based decompositions.
  Never infer a conditional model silently.
- **GEE interpretation.** Working covariance is not a generative model. Document
  this distinction prominently.
- **Symbolic matrix limits.** No densification of symbolic shapes. `dense()`
  requires explicit observations.
- **Quarto/quartodoc complexity.** Keep the API reference generated from
  docstrings; pin Quarto and quartodoc versions.
- **Scope creep to GLMM.** Keep the `Family`/`Link`/`Marginalizer` protocols but
  do not implement GLMM marginalization in v0.1.
- **Required `form` verbosity.** Accepted tradeoff; no hidden defaults.

---

## 13. Definition of done (v0.1)

- `uv run pytest` passes with coverage ≥ 90%.
- `uv run ruff check`, `uv run ruff format --check`, and `uv run mypy` pass.
- `quarto render docs` succeeds and includes all required pages.
- Conditional LMM, marginal LMM, and GEE models expose sum and matrix forms.
- All public calls require `form`; form/arity errors are typed and tested.
- Mean, variance, covariance, correlation, ICC, variogram, covariogram, and
  BLUP are exposed as first-class methods with documented return types.
- Conditional moments are available from LMMs; marginal moments are the default.
- Variograms require `kind` and support raw, residual, and conditional forms.
- RI, RS, AR(1), CS, US, heteroscedastic, Toeplitz, spatial, nested, and crossed
  structures are supported and tested.
- Multiple random terms (K>1) assemble correctly.
- PD-safe parameterizations and `validate_parameters()` are enforced.
- Conditional → marginal always works for LMMs.
- Marginal → conditional works only for registered exact decompositions and
  raises `NotDecomposable` with reason/suggestion otherwise.
- GEE supports Gaussian/identity, Poisson/log, and Bernoulli/logit with working
  correlations; raw variograms use the link-general mean difference.
- Evaluation (`subs`, `evaluate`, `lambdify`) and dense mode are implemented and
  used by validation.
- Monte Carlo and reference fixtures validate the core formulas.
- LaTeX and Markdown printers are tested for sum, matrix, and both.
- No GLMM implementation; extension protocols only.

---

## 14. Future work (post-v0.1)

- GLMM marginalization: exact kernels for log/Poisson and probit/Bernoulli,
  Laplace and Gauss–Hermite approximations for other links.
- GEE estimating equations and sandwich covariance.
- Formula parser (`y ~ 1 + x + (1 + x | group)`) via an optional adapter.
- Additional covariance structures: factor analytic, reduced rank, pedigrees.
- Code generation targets: Stan, JAX, PyMC.
- Bayesian extensions: posterior moments and marginal likelihood expressions.

---

## 15. Gap-resolution log

| Gap | Resolution |
|---|---|
| No task breakdown | §11 points to three executable task plans with TDD steps and exit criteria. |
| Clustered index model | §3.0 and §5.0 define one observation index, accessors, nested/crossed labels, and GEE aliases. |
| Matrix-form representation | §2 decision 9 defines an abstract matrix layer plus a structured entry layer; M0.5 validates SymPy capabilities. |
| PD/parameter constraints | §2 decision 10, §5.7, §6.1–6.2 define PD-safe constructors, `constraints()`, and tri-state PD checks. |
| Decomposition algorithm | §7.2 replaces "registry match" with a bounded canonicalize/match/solve/validate algorithm. |
| No evaluation API | §2 decision 12 and §5.5 define `subs`, `evaluate`, `lambdify`, and `dense`. |
| Incomplete public surface | §2 decision 13 and §5.1–5.3 add correlation, ICC, covariogram, BLUP, conditional moments, and `variogram(kind=)`. |
| Untested SymPy assumptions | §11 M0.5 adds a feasibility spike with acceptance tests and a decision record. |
| Raw variogram correctness | §3.6 uses the link-general mean difference. |
| Naming collisions | §2 decision 14 and §6.3 separate covariance and working-correlation classes. |
| Test traceability | §10.4 adds a DoD-to-test matrix; §13 expands the definition of done. |
