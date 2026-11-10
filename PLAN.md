# symlmm — Implementation Plan

**Status:** implementation in progress; v0.5.0 adds canonical group/occasion
indexing and the blackboard facade<br>
**Date:** 2026-11-09 (canonical indexing update: 2026-11-10)<br>
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
8. **Canonical group/occasion index (model B).** `i` indexes the
   group/cluster/subject and `t` indexes the occasion. For a balanced panel the
   response is a matrix `Y` with groups as rows and occasions as columns:
   `Y_i = Y[i, :]` is the group row vector, `Y[:, t]` (also written `Y[t]`) is
   the occasion column, and `Y_{it}` is a scalar. The group covariance is
   `V_i = Z_i G Z_i' + R_i`, with `Cov(Y_i, Y_j) = 0` for `i != j` and
   `Cov(Y_{it}, Y_{is}) = [V_i]_{ts}`. Index symbols belong to the `Group`,
   `Nested`, `Crossed`, and `Covariate` definitions; the model derives
   `group_index` and `occasion_index` from them and never hardcodes `i` or `t`.
   The stacked observation row used by the matrix engine is an implementation
   detail, not public model notation. This resolves issue
   [#1](https://github.com/kantundpeterpan/symlmm/issues/1).
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

### 3.0 Canonical index and data model

The public response object is a matrix `Y` with groups as rows and occasions as
columns. For a balanced panel with `m` groups and `T` occasions, `Y` is
`m x T`. Across both balanced and unbalanced panels, `Y_i` denotes the response
vector for group `i`; only in a balanced panel is it a row of the rectangular
matrix, `Y_i = Y[i, :]`. In an unbalanced panel `Y_i` remains a group vector,
but it is not a row of a rectangular `Y` matrix, and occasion columns `Y[:, t]`
are not defined for the whole panel.

Python code and internal storage are 0-based; rendered mathematical formulas
are 1-based and state the conversion. Every integer group, occasion, or row
index in a Python example is 0-based.

- `i` indexes the group/cluster/subject; `t` indexes the occasion.
- `Y_i` is the group response vector. In a balanced panel it is the row vector
  `Y_i = Y[i, :]`; in an unbalanced panel it is a group vector, not a row of a
  rectangular `Y` matrix.
- `Y[:, t]`, also written `Y[t]` when `t` is the model-defined occasion
  symbol, is the occasion column vector for a balanced panel.
- `Y_{it}` is the scalar response at group `i`, occasion `t`; it is defined for
  both balanced and unbalanced panels.
- `b_i` is the group random-effect vector.
- `X_i`, `Z_i`, and `R_i` are group design and residual blocks; `G` is the
  random-effect covariance.
- `V_i = Z_i G Z_i' + R_i`.
- `Cov(Y_i, Y_j) = 0` for `i != j`.
- `Cov(Y_{it}, Y_{is}) = [V_i]_{ts}`.
- For a balanced panel, `Cov(Y[:, t], Y[:, s]) = diag_i([V_i]_{ts})`.

**Primary group and canonical occasion.** The model resolves the canonical
indices by fixed precedence:

- primary group = the first `RandomEffects` block's `Group`; otherwise the
  residual covariance's group definition; otherwise the first registered
  grouping definition.
- For the selected coordinate name, canonical occasion = the residual
  covariance's coordinate `Covariate`; otherwise the fixed-effect slope
  `Covariate` with the same name; otherwise a random-slope `Covariate` when
  exactly one distinct coordinate exists; otherwise, when `coordinate=` was
  supplied, a synthesized `Covariate(name)`; otherwise no occasion
  (`occasion_index is None`).

The coordinate name is selected from an explicit `LMM(coordinate=...)` when
supplied, otherwise from the residual coordinate, otherwise from the single
distinct random-slope coordinate. An explicit `coordinate=` that conflicts
with a residual coordinate raises `ModelError`. Supplying `coordinate=` never
produces "no occasion": when no residual, fixed, or random-slope definition
matches, the model synthesizes `Covariate(coordinate_name)`.

Derived canonical occasion `t` is a 0-based rank in sorted occasion order
within each group, not the raw coordinate value. An explicit `occasion=`
column supplies the codes by first appearance within each group.

Only the primary group defines the canonical `Y_i` block, its design, and
residual covariance. Group and entry covariance require a single grouping
factor; with multiple grouping factors, `group_covariance`,
`entry_covariance`, and cross-factor block covariance raise `NotDefined`.

Index symbols belong to the structural definitions, not to a separate index
object:

```python
subject = sl.Group("subject", index=sp.Symbol("i", integer=True))
visit = sl.Covariate("time", index=sp.Symbol("t", integer=True))
tau2 = sp.Symbol("tau2", positive=True)
beta0, beta1 = sp.symbols("beta0 beta1", real=True)
sigma2 = sp.Symbol("sigma2", positive=True)
rho = sp.Symbol("rho", positive=True)

model = sl.LMM(
    fixed=sl.Fixed(intercept=beta0, slopes={visit: beta1}),
    random=[
        sl.RandomEffects(
            group=subject,
            intercept=True,
            cov=sl.Scalar(tau2),
        )
    ],
    residual=sl.AR1(group=subject, coordinate=visit, var=sigma2, rho=rho),
)

model.group_index is subject.index       # i
model.occasion_index is visit.index      # t

j = sp.Symbol("j", integer=True)
s = sp.Symbol("s", integer=True)
```

- `Group(name, index=...)`, `Nested(child, within, index=...)`, and
  `Crossed(left, right, index=...)` own their group index symbol. When omitted,
  the index is derived from the factor label (`subject`, `subject:visit`,
  `subject:item`).
- `Covariate(name, index=...)` owns its covariate/occasion index symbol. When
  omitted, the index is derived from the name.
- `Fixed.slopes` maps `Covariate` definitions to coefficients.
  `RandomSlope(coordinate=...)` takes a `Covariate` definition, not a bare
  string. `RandomEffects`, `AR1`, and `Toeplitz` also take `Group` and
  `Covariate` definitions. A bare string is accepted only as a compatibility
  shim and is normalized to `Covariate(name)`; canonical code passes the
  definition.
- A model rejects equal group and occasion index symbols with `ModelError`.
- No response accessor hardcodes an index name. `Y[t]` dispatches on the
  model-defined occasion symbol.

Every model exposes the canonical group/occasion API:

```python
model.group_count()                      # m, symbolic or dense
model.occasion_count()                   # T, balanced panels
model.group_size(i)                      # n_i
model.response_matrix()                  # Y, m x T
model.response(i)                        # Y_i row vector (balanced)
model.time_response(t)                   # Y[:, t] column vector
model.response(i, t)                     # Y_it scalar

model.fixed_design(i)                    # X_i
model.random_design(i)                   # Z_i
model.residual_covariance(i)             # R_i
model.random_effect(i)                   # b_i
model.group_mean(i)                      # X_i beta
model.group_covariance(i)                # V_i
model.group_cross_covariance(i, j)       # zero for i != j
model.entry_mean(i, t)
model.entry_covariance(i, t, s)
model.entry_cross_covariance(i, t, j, s)
model.time_mean(t)                       # E[Y[:, t]]
model.time_covariance(t, s)              # Cov(Y[:, t], Y[:, s])
```

Occasion-column moments require a balanced panel with a concrete group count
or dense mode; the symbolic arbitrary-`m` form raises `NotDefined`. With
multiple grouping factors, the canonical `Y_i` block uses the primary group
only; block moments that would combine factors, including
`group_cross_covariance`, raise `NotDefined`.

**Stacked engine internals.** Matrix assembly still stacks observations into a
vector `y` and addresses rows with a 0-based observation-row cursor. The
methods `mean(row)`, `covariance(row, row)`, `variance(row)`,
`correlation(row, row)`, `variogram(row, row)`, `covariogram(row, row)`, and
`blup_cross_covariance(row)` operate on that stacked engine. They remain
available for matrix assembly, dense evaluation, variograms, and BLUP, but
they are not canonical model notation. Accessors such as
`model.group(row, factor)`, `model.occasion(row, factor)`,
`model.coordinate(row, ...)`, and `model.design(row)` read structural indices
for stacked rows and are likewise engine/diagnostic internals. Use the
canonical group/occasion methods above for model work, and never describe a
stacked row as `Y_i`.

### 3.1 Conditional LMM

The canonical conditional model is written group by group:

\[
Y_i = X_i\beta + Z_i b_i + \varepsilon_i,\qquad
b_i \sim N(0,G),\quad \varepsilon_i \sim N(0,R_i),\quad b_i \perp \varepsilon_i
\]

with independent groups. Moments:

\[
E[Y_i\mid b_i]=X_i\beta+Z_i b_i,\qquad \operatorname{Cov}(Y_i\mid b_i)=R_i
\]

\[
E[Y_i]=X_i\beta,\qquad \operatorname{Cov}(Y_i)=V_i=Z_i G Z_i^\top+R_i
\]

\[
\operatorname{Cov}(Y_i,Y_j)=0\quad(i\ne j),
\qquad \operatorname{Cov}(Y_{it},Y_{is})=[V_i]_{ts}
\]

For multiple random terms, `Z_i` stacks the columns of the terms that apply
to group `i`, and `G` is block diagonal:

\[
Z_i=[Z_{i,1}\ \cdots\ Z_{i,K}],\qquad
G=\operatorname{BlockDiag}(G_1,\dots,G_K)
\]

Each block is declared as one `RandomEffects(group, cov, intercept, slopes)`.
The block owns its grouping factor, the intercept flag, the `RandomSlope`
design definitions, and the joint covariance \(G_k\). Intercepts and slopes
that share a group therefore share one covariance block, including their
covariances. Different grouping factors use separate blocks, and a grouping
factor appears in at most one block.

The matrix engine assembles the equivalent stacked form
\(y = X\beta + Zb + \varepsilon\) with \(b\sim N(0,G)\),
\(\varepsilon\sim N(0,R)\), and
\(\operatorname{Cov}(y)=ZGZ^\top+R\). That stacked form is an implementation
detail; canonical model statements use `Y_i`, `Y_{it}`, and `V_i`.

### 3.2 Marginal LMM

Two routes:

1. **Direct marginal model**

   \[
   Y_i \sim N(X_i\beta,\;V^m_i(\alpha))
   \]

   with `V^m_i` built from a `MarginalCovSpec`.

2. **Induced marginal model**

   \[
   V^m_i = Z_i G Z_i^\top + R_i
   \]

Both expose the same moment API. The variance decomposition is made explicit
group by group:

\[
V^m_i=\underbrace{\operatorname{Var}(E[Y_i\mid b_i])}_{Z_i G Z_i^\top}
+\underbrace{E[\operatorname{Var}(Y_i\mid b_i)]}_{R_i}
\]

### 3.3 GEE marginal model

GEE specifies a marginal mean and a working covariance directly, with `i` the
cluster and `t` the within-cluster occasion:

\[
\mu_{it}=g^{-1}(x_{it}^\top\beta)
\]

\[
\operatorname{Var}(Y_{it})=\phi\,v(\mu_{it})
\]

\[
V_i=\phi\,A_i^{1/2}R_i(\alpha)A_i^{1/2},\qquad
A_i=\operatorname{diag}\!\big(v(\mu_{it})\big)
\]

\[
\operatorname{Cov}(Y_i,Y_j)=0\quad(i\ne j)
\]

The stacked engine assembles the full covariance as
\(V=\operatorname{BlockDiag}(V_1,\dots,V_m)\); that stacked `V` is an
implementation detail.

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

For a conditional LMM, canonical entries are indexed by group and occasion:

\[
\operatorname{Cov}(Y_{it},Y_{is})
=
\sum_{k=1}^{K}
\sum_{a=1}^{q_k}
\sum_{b=1}^{q_k}
z_{k,it,a}\,G_{k,ab}\,z_{k,is,b}
+
[R_i]_{ts}
\]

Equivalently, with \(V_i=Z_iGZ_i^\top+R_i\),
\(\operatorname{Cov}(Y_{it},Y_{is})=[V_i]_{ts}\). Across groups,
\(\operatorname{Cov}(Y_{it},Y_{js})=0\) for \(i\ne j\).

For a direct marginal model:

\[
\operatorname{Cov}(Y_{it},Y_{is})=[V^m_i]_{ts}
\]

For GEE:

\[
\operatorname{Cov}(Y_{it},Y_{is})
=
\phi\,
\sqrt{v(\mu_{it})}\,
R_{i,ts}(\alpha)\,
\sqrt{v(\mu_{is})}
\]

and zero across clusters.

Implementation uses `IndexedBase`, `Sum`, `KroneckerDelta`, and Euclidean
distance for spatial coordinates. All sums are 0-based internally; rendered
output converts bounds to 1-based. Group-level and occasion symbols come from
the model definitions, so the same formula derives for nested and crossed
factors after label construction. Spatial distance is
\(\lVert s_{it}-s_{is}\rVert\), not a scalar absolute difference; `Abs` is used
only for one-dimensional coordinate lags.

Canonical entry queries place `t` and `s` within one group. The stacked engine
also accepts `within=factor` on row queries to place two stacked rows in one
common level of that factor, resolving the factor's `KroneckerDelta` to `1`.
The general row expression keeps the delta; resolution is a query operation,
not a change to the model IR.

Special cases must simplify automatically:

- random intercept:
  \[
  \operatorname{Cov}(Y_{it},Y_{is})=\tau^2+\sigma^2\delta(t,s)
  \]
  within a group, and zero across groups.
- random slope:
  \[
  \big[\tau_{00}+\tau_{01}(X_{it}+X_{is})
  +\tau_{11}X_{it}X_{is}\big]+\sigma^2\delta(t,s)
  \]
- AR(1) residual:
  \[
  \sigma^2\rho^{|X_{it}-X_{is}|}
  \]
  within a group, and zero across groups.

### 3.5 Matrix-to-sum bridge

SymPy maps canonical group matrix entries to explicit sums:

```python
V_i = Z_i * G * Z_i.T + R_i
V_i[t, s]
# R_i[t,s] + Sum(
#     G[_a,_b] * Z_i[t,_a] * Z_i[s,_b],
#     (_a, 0, q - 1),
#     (_b, 0, q - 1),
# )
```

`matrix_to_sum` canonicalizes this result. `sum_to_matrix` assembles
`X, Z, G, R` from the model IR. Both directions share the same model
specification. The stacked engine builds the same expression over rows and
uses `V[row, row]` internally for row-level queries; that row form is not
canonical model notation.

`as_explicit()` is never called on symbolic dimensions. It is only used in the
explicit dense mode, which requires concrete observations.

### 3.6 Variograms

Semantics must be explicit. Variograms are defined for canonical entries
`Y_{it}` and `Y_{is}`; cross-group covariances are zero.

Residual (mean-adjusted) variogram within group `i`:

\[
\gamma^{\text{res}}(Y_{it},Y_{is})
=
\tfrac12\left([V_i]_{tt}+[V_i]_{ss}-2[V_i]_{ts}\right)
\]

For `i != j` the cross-covariance is zero, so

\[
\gamma^{\text{res}}(Y_{it},Y_{js})
=
\tfrac12\left([V_i]_{tt}+[V_j]_{ss}\right).
\]

Raw variogram:

\[
\gamma^{\text{raw}}(Y_{it},Y_{is})
=
\tfrac12\left(\mu_{it}-\mu_{is}\right)^2
+
\tfrac12\left([V_i]_{tt}+[V_i]_{ss}-2[V_i]_{ts}\right),
\qquad
\mu_{it}=g^{-1}\!\left(x_{it}^\top\beta\right)
\]

For the identity link this reduces to
\(\tfrac12[(x_{it}-x_{is})^\top\beta]^2\) plus the residual-variogram term.
The non-identity form is required for Poisson/log and Bernoulli/logit GEE
variograms, which are in scope (§8.4).

Conditional variogram:

\[
\gamma^{\text{cond}}(Y_{it},Y_{is})
=
\tfrac12\left([R_i]_{tt}+[R_i]_{ss}-2[R_i]_{ts}\right)
\]

Conditional raw form also includes the fixed/random mean difference:

\[
\tfrac12\left[
(x_{it}-x_{is})^\top\beta+(z_{it}-z_{is})^\top b_i
\right]^2
+
\tfrac12\left([R_i]_{tt}+[R_i]_{ss}-2[R_i]_{ts}\right)
\]

Covariogram (the covariance itself, retained as a named quantity):

\[
C(Y_{it},Y_{is})=\operatorname{Cov}(Y_{it},Y_{is})=[V_i]_{ts},
\qquad C(Y_{it},Y_{js})=0\quad(i\ne j)
\]

Variogram selection is explicit: `kind="raw"`, `kind="residual"`, or
`kind="conditional"`. There is no default. The public `variogram(row, row)`
and `covariogram(row, row)` methods evaluate these expressions through the
stacked row cursor; that row API is an engine internal, not canonical notation.

Stationary specialization substitutes `X_{it} = x` and `X_{is} = x + h`.

### 3.7 BLUP moments

For group `i` under a single primary grouping factor, the canonical BLUP
moments are

\[
E[b_i\mid Y_i]=GZ_i^\top V_i^{-1}(Y_i-X_i\beta)
\]

\[
\operatorname{Cov}(b_i\mid Y_i)
=
\left(G^{-1}+Z_i^\top R_i^{-1}Z_i\right)^{-1}
\]

\[
\operatorname{Cov}(Y_{it},b_i)=z_{i,t}^\top G
\]

where `z_{i,t}` is the row of `Z_i` for occasion `t`. Across groups the
conditional cross-covariance is zero under independent random effects. Dense
mode exposes explicit `V_i`, `G`, `R_i`, and the BLUP expressions for small
concrete instances.

The engine evaluates the equivalent stacked form
\(E[b\mid y]=GZ^\top V^{-1}(y-X\beta)\),
\(\operatorname{Cov}(b\mid y)=(G^{-1}+Z^\top R^{-1}Z)^{-1}\), and
\(\operatorname{Cov}(y_r,b_k)=z_{k,r}^\top G_k\) with the stacked row cursor.
That row form is an implementation detail; `blup_cross_covariance(row)` remains
a stacked-engine query.

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
    symbols.py          # structural indices, parameter symbols
    canonical.py        # canonical group/occasion symbol helpers
    specs.py            # Fixed, Random*, Group, Nested, Crossed, Covariate
    constraints.py      # Constraint, PD tri-state checks
    covariances.py      # random/residual covariance structures
    forms.py            # Form, DualForm, EntryMap, bridges
    model.py            # LMM, canonical group/occasion API, form dispatch
    moments.py          # mean/var/cov/corr/ICC, conditional moments
    blackboard.py       # ModelSymbols and blackboard moment engine
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
    pedagogy/
      blackboard.md
    api/
  examples/
    *.qmd
  scripts/
  .github/workflows/ci.yml
```

### Module responsibilities

- `errors.py`: `SymlmmError` hierarchy; every public failure has a typed error.
- `symbols.py`: structural indices and parameter symbols with assumptions.
- `canonical.py`: symbolic group size, design, residual, and random-effect
  block helpers for canonical `Y_i` indexing.
- `specs.py`: fixed effects, random terms, grouping factors, covariate
  definitions, design vectors.
- `constraints.py`: `Constraint`, constraint collection, PD tri-state checks.
- `covariances.py`: covariance structures used by conditional and marginal
  models; PD-safe constructors.
- `forms.py`: `Form`, `DualForm`, `EntryMap`, arity validation, sum/matrix
  conversion.
- `model.py`: `LMM`, canonical group/occasion methods, stacked row accessors,
  and required-form dispatch.
- `moments.py`: mean/variance/covariance/correlation/ICC, conditional moments.
- `blackboard.py`: `ModelSymbols` with canonical `Y_i`, `Y[:, t]`, and `Y_{it}`
  atoms plus the linear `E`/`Var`/`Cov` engine.
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

**Canonical index (model B).** `i` indexes the group/cluster/subject and `t`
indexes the occasion. `Y_i` is the group response vector; for a balanced panel
`Y_i = Y[i, :]` is a row of the `m x T` response matrix `Y`, while for an
unbalanced panel `Y_i` remains a group vector but is not a row of a rectangular
`Y` matrix. `Y[:, t]` (also `Y[t]` for the model-defined occasion symbol) is the
occasion column for a balanced panel, and `Y_{it}` is a scalar entry for both
balanced and unbalanced panels. Python examples use 0-based indices; rendered
mathematical formulas use 1-based bounds. Index symbols belong to the
structural definitions, never to a separate index object:

```python
subject = sl.Group("subject", index=sp.Symbol("i", integer=True))
visit = sl.Covariate("time", index=sp.Symbol("t", integer=True))
tau2 = sp.Symbol("tau2", positive=True)
beta0, beta1 = sp.symbols("beta0 beta1", real=True)
sigma2 = sp.Symbol("sigma2", positive=True)
rho = sp.Symbol("rho", positive=True)

model = sl.LMM(
    fixed=sl.Fixed(intercept=beta0, slopes={visit: beta1}),
    random=[
        sl.RandomEffects(
            group=subject,
            intercept=True,
            cov=sl.Scalar(tau2),
        )
    ],
    residual=sl.AR1(group=subject, coordinate=visit, var=sigma2, rho=rho),
)

model.group_index is subject.index       # i
model.occasion_index is visit.index      # t
```

`Group`, `Nested`, `Crossed`, and `Covariate` derive an index symbol from the
name or factor label when `index=` is omitted. A model rejects equal group and
occasion index symbols with `ModelError`. The primary group and canonical
occasion follow the precedence in §3.0. An explicit `LMM(coordinate=...)`
selects the coordinate name; if it conflicts with a residual coordinate, the
model raises `ModelError`. A supplied `coordinate=` never yields "no occasion":
when no residual, fixed, or random-slope definition matches, the model
synthesizes `Covariate(coordinate_name)`. Nested and crossed factors remain
declared in the spec (`sl.Nested(...)`, `sl.Crossed(...)`). With multiple
grouping factors, the canonical `Y_i` block uses the primary group only;
`group_covariance`, `entry_covariance`, and block moments that would combine
factors, including `group_cross_covariance`, raise `NotDefined`.

**Canonical group/occasion methods.** `response`, `time_response`,
`group_mean`, `group_covariance`, `group_cross_covariance`, `entry_mean`,
`entry_covariance`, `entry_cross_covariance`, `time_mean`, `time_covariance`,
`fixed_design`, `random_design`, `residual_covariance`, `random_effect`,
`group_size`, `group_count`, `occasion_count`, and `response_matrix` return
symbolic expressions directly and take no `form` argument. Occasion-column
moments require a concrete group count or dense mode and otherwise raise
`NotDefined`.

**Stacked row engine (internal).** The matrix engine addresses the stacked
response with a 0-based observation-row cursor. The `form`-based methods
`mean(row)`, `covariance(row, row)`, `variance(row)`, `correlation(row, row)`,
`variogram(row, row)`, `covariogram(row, row)`, and
`blup_cross_covariance(row)` operate on that cursor. They remain available for
matrix assembly, dense evaluation, variograms, and BLUP, but they are not
canonical model notation. Structural accessors such as
`model.group(row, factor)`, `model.occasion(row, factor)`, and
`model.coordinate(row, ...)` are engine/diagnostic internals. Use the canonical
group/occasion methods for model work.

**Delta resolution.** The stacked row methods `covariance`, `variance`,
`correlation`, `variogram`, and `covariogram` accept `within=factor`. The query
places two stacked rows in one common level of that factor, resolving its
`KroneckerDelta` to `1`; the default query keeps the general cross-group
expression. Canonical entry queries are already within one group.

**Form and arity.** The required `form` contract applies to the stacked row
queries in §5.4, not to the canonical group/occasion methods. `form` selects
notation, not conditioning. Valid combinations are fixed:

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
mapping `(row, row) -> expression`. Invalid arity raises `FormArityError`; an
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

subject = sl.Group("subject", index=sp.Symbol("i", integer=True))
visit = sl.Covariate("visit_time", index=sp.Symbol("t", integer=True))

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

# Canonical group/occasion API
i = model.group_index
t = model.occasion_index
j = sp.Symbol("j", integer=True)
s = sp.Symbol("s", integer=True)

model.response_matrix()                          # Y, m x T
model.group_count()                              # m
model.occasion_count()                           # T
model.group_size(i)                              # n_i
model.response(i)                                # Y_i row vector (balanced)
model.time_response(t)                           # Y[:, t] column vector
model.response(i, t)                             # Y_it scalar
model.fixed_design(i)                            # X_i
model.random_design(i)                           # Z_i
model.residual_covariance(i)                     # R_i
model.random_effect(i)                           # b_i
model.group_mean(i)                              # E[Y_i] = X_i beta
model.group_covariance(i)                        # V_i = Z_i G Z_i' + R_i
model.group_cross_covariance(i, j)               # zero for i != j
model.entry_mean(i, t)
model.entry_covariance(i, t, s)                  # [V_i]_{ts}
model.entry_cross_covariance(i, t, j, s)         # zero for i != j
model.icc(at={"coordinate": 0}, form="both")     # conditional VPC
model.blup_mean(form="matrix")
model.blup_covariance(form="matrix")
model.summary(form="both")
model.to_latex(form="sum")

# Stacked row API (internal): rows are not Y_i
row = sp.Symbol("row", integer=True, nonnegative=True)
other = sp.Symbol("other", integer=True, nonnegative=True)

model.mean(form="matrix")                        # X beta
model.mean(row, form="sum")                      # x_row^T beta
model.mean(row, form="both")                     # DualForm
model.conditional_mean(form="matrix")            # X beta + Z b
model.conditional_mean(row, form="sum")          # x_row^T beta + z_row^T b
model.covariance(form="matrix")                  # V
model.covariance(row, other, form="sum")         # expanded V_row,other
model.covariance(row, other, form="matrix")      # V[row, other]
model.covariance(row, other, form="sum", within="subject")  # delta resolved
model.variance(form="matrix")                    # variance vector (DiagonalOf)
model.variance(row, form="sum")                  # V_row,row
model.correlation(form="matrix")                 # Corr(V)
model.correlation(row, other, form="sum")        # Corr(V)_row,other
model.variogram(row, other, kind="residual", form="both")
model.variogram(row, other, kind="residual", form="sum", within="subject")
model.covariogram(row, other, form="both")
model.covariogram(row, other, form="sum", within="subject")
model.blup_cross_covariance(row, form="matrix")
```

### 5.2 Marginal LMM

The canonical marginal model is `Y_i ~ N(X_i beta, V^m_i(alpha))`, with
independent groups. For the induced marginal model,
`V^m_i = Z_i G Z_i' + R_i`.

```python
marg = sl.MarginalLMM(
    fixed=sl.Fixed(intercept=beta0, slopes={visit: beta1}),
    covariance=sl.AR1(group=subject, coordinate=visit, var=sigma2, rho=rho),
)

marg.mean(form="matrix")
marg.covariance(row, other, form="both")       # stacked row API, internal
marg.covariance(row, other, form="sum", within="subject")
marg.correlation(row, other, form="sum")
marg.variogram(row, other, kind="residual", form="sum", within="subject")
marg.covariogram(row, other, form="sum", within="subject")
marg.icc(at={"coordinate": 0}, form="both")   # Gaussian identity only
marg.summary(form="both")
```

Induced marginal:

```python
induced = model.marginal(form="both")
induced.covariance(form="matrix")       # Z G Z^T + R
induced.covariance(row, other, form="sum")    # stacked entrywise sum
```

Marginal → conditional:

```python
marg.is_decomposable()                  # True | False; bounded template match
marg.decomposition_reason()             # reason when False
marg.conditional(form="both")           # LMM, or raises NotDecomposable
```

### 5.3 GEE

The canonical GEE moments are cluster-indexed: `Y_i` is the cluster response
vector with `E[Y_{it}] = mu_{it}` and working covariance `V_i`. The stacked row
queries below remain engine-internal.

```python
gee = sl.GEE(
    fixed=sl.Fixed(intercept=beta0, slopes={visit: beta1}),
    family=sl.Poisson(link="log"),
    correlation=sl.WorkingAR1(group=subject, coordinate=visit, rho=rho),
    dispersion=sp.Symbol("phi", positive=True),
)

gee.mean(form="matrix")
gee.mean(row, form="sum")
gee.variance(form="matrix")
gee.covariance(row, other, form="both")
gee.covariance(row, other, form="sum", within="subject")
gee.variogram(row, other, kind="raw", form="sum")
gee.covariogram(row, other, form="sum")
gee.occasion(row, "subject")             # within-cluster occasion, internal
gee.summary(form="both")
gee.conditional()                        # raises NoConditionalModel
gee.icc(at={"coordinate": 0}, form="both")  # raises NotDefined
```

### 5.4 Required `form`

Every stacked row query and every form-based public method requires `form`.
Missing `form` raises `MissingFormError`. Valid values:

- `"sum"` — indexed expression over the stacked row cursor, deltas, and `Sum`;
- `"matrix"` — a `MatrixExpr` for whole-matrix queries or a `MatrixElement` for
  entry queries;
- `"both"` — `DualForm(sum=..., matrix=...)`.

An invalid string raises `InvalidFormError`; an invalid arity/form combination
raises `FormArityError` (table in §5.0). `form="both"` does not mean conditional
+ marginal; conditional/marginal comparison uses `compare_specs`. Canonical
group/occasion methods return expressions directly and do not take `form`.

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
- `dense` requires concrete observation count, group labels, coordinates, and
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

### 5.7 Blackboard symbols and moments

`model.symbols()` returns `ModelSymbols`, a teaching facade over the canonical
group/occasion layer. The index names are not hardcoded: `syms.i` is the
model's group index and `syms.t` is its occasion index.

```python
syms = model.symbols()
s = sp.Symbol("s", integer=True)

assert syms.i == model.group_index
assert syms.t == model.occasion_index

syms.Y[syms.i]              # Y_i row vector (balanced)
syms.Y[syms.t]              # Y[:, t] occasion column
syms.Y[syms.i, syms.t]      # Y_it scalar
syms.Y[syms.i, :]           # row slice
syms.Y[:, syms.t]           # column slice

syms.b[syms.i]              # b_i vector
syms.beta                   # fixed-effect vector
syms.X(syms.i)              # X_i
syms.Z(syms.i)              # Z_i
syms.R(syms.i)              # R_i
syms.G                      # G
```

Canonical moments delegate to the group/occasion model methods:

```python
syms.E(syms.Y[syms.i])                    # X_i beta
syms.Var(syms.Y[syms.i])                  # V_i
syms.Cov(syms.Y[syms.i], syms.Y[syms.i])  # V_i
syms.E(syms.Y[syms.i, syms.t])            # E[Y_it]
syms.Var(syms.Y[syms.i, syms.t])          # [V_i]_{tt}
syms.Cov(syms.Y[syms.i, syms.t], syms.Y[syms.i, s])  # [V_i]_{ts}

syms.E(syms.Y[syms.i], given=syms.b[syms.i])  # X_i beta + Z_i b_i
syms.Var(syms.Y[syms.i], given=syms.b[syms.i])  # R_i
syms.total_variance(syms.Y[syms.i], given=syms.b[syms.i])  # V_i
```

The facade returns ordinary SymPy expressions, not `sympy.stats` random
variables. The moment engine supports linear expressions in model atoms;
nonlinear or unsupported combinations raise `UnsupportedMoment`. `Y[0]` is
ambiguous and raises `ModelError`; use `Y[0, :]` for a group row or `Y[:, 0]`
for an occasion column. Occasion-column moments require a concrete group count
or dense mode. See `docs/pedagogy/blackboard.md` for a worked example and the
law-of-total-variance derivation.

### 5.8 Errors

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

Random terms are specified in `specs.py`; covariance structure is separate.
Each grouping factor contributes one joint `RandomEffects` block:

- `RandomEffects(group, cov, intercept=False, slopes=())` pairs one grouping
  factor with one joint design and covariance. `intercept=True` prepends the
  constant column; `slopes` is a tuple of `RandomSlope` definitions. `cov` must
  be `dim × dim`, where `dim = int(intercept) + len(slopes)`, so
  intercept–slope and slope–slope covariances are modelled jointly.
- `RandomSlope(coordinate, within=True)` is a pure design definition: its
  `coordinate` argument takes a `Covariate` definition, not a bare string. It
  names the `Covariate` column and whether it is evaluated within the group
  (`within=True`, default: `X_{it}`) or globally (`within=False`: `X_i`). It
  carries no covariance. A bare string is accepted only as a compatibility
  shim and is normalized to `Covariate(name)`.
- Different grouping factors use separate blocks; a grouping factor may appear
  in at most one block. Blocks assemble as
  \(Z_i=[Z_{i,1}\ \cdots\ Z_{i,K}]\),
  \(G=\operatorname{BlockDiag}(G_1,\dots,G_K)\). Every block applies only to
  its own grouping factor; cross-group covariance is zero.

```python
subject = sl.Group("subject", index=sp.Symbol("i", integer=True))
item = sl.Group("item", index=sp.Symbol("k", integer=True))
visit = sl.Covariate("visit_time", index=sp.Symbol("t", integer=True))
dose = sl.Covariate("dose", index=sp.Symbol("d", integer=True))

# intercept-only: 1x1 covariance
sl.RandomEffects(group=subject, intercept=True, cov=sl.Scalar(tau2))

# joint intercept and slope: 2x2 covariance, including their covariance
sl.RandomEffects(
    group=subject,
    intercept=True,
    slopes=(sl.RandomSlope(visit),),
    cov=sl.Unstructured.from_sd_corr(sd=[tau0, tau1], corr=rho_b),
)

# slopes only, possibly in different coordinates
sl.RandomEffects(
    group=subject,
    slopes=(sl.RandomSlope(visit), sl.RandomSlope(dose, within=False)),
    cov=sl.Diagonal([tau_time, tau_dose]),
)

# a second grouping factor is a separate block
sl.RandomEffects(group=item, intercept=True, cov=sl.Scalar(omega2))
```

Covariance building blocks:

- `Scalar(tau2)`: 1×1 covariance, \(\tau^2>0\).
- `Diagonal(q)`: `IndexedBase("d", positive=True)`, \(d_k>0\).
- `Unstructured(q)`: default `from_cholesky(L)`; for symbolic `q`, an
  `IndexedBase("L", (q, q))` with lower-triangular structure and positive
  diagonal.
- `Kronecker`: product of positive-definite factors.
- `FactorAnalytic(k)`: future work; not in v0.1.

### 6.2 Residual / marginal covariance `R`

- `IID`: \(\sigma^2\delta(t,s)\) within a group and zero across groups,
  \(\sigma^2>0\).
- `CompoundSymmetry(group, var, rho, nugget=0)`:
  \(\rho>-1/(m_{\min}-1)\), where \(m_{\min}\) is the minimum cluster size.
- `AR1(group, coordinate, var, rho, nugget=0)`:
  \(\sigma^2\rho^{|X_{it}-X_{is}|}\) within group `i` and zero across groups,
  \(-1<\rho<1\).
- `Toeplitz(group, coordinate, variances, correlations, nugget=0)`:
  `from_cholesky` or checked dense construction.
- `Unstructured`: `from_cholesky`.
- `Heteroscedastic`: `IndexedBase("sigma2", positive=True)`.
- `Exponential(var, range, group=None, coords=("x", "y"), nugget=0)`:
  range \(>0\).
- `Gaussian(var, range, group=None, coords=("x", "y"), nugget=0)`:
  range \(>0\).
- `Matern(var, range, nu, group=None, coords=("x", "y"), nugget=0)`:
  range \(>0\), \(\nu\in\{\tfrac12,\tfrac32,\tfrac52\}\), Euclidean distance
  \(\lVert s_{it}-s_{is}\rVert\).
- `BlockDiag`: explicit blocks or group-index blocks.
- `KroneckerProduct`: separable group × coordinate structures.

`AR1`, `Toeplitz`, `CompoundSymmetry`, `Exponential`, `Gaussian`, and `Matern`
accept an additive `nugget >= 0`. The zero-lag covariance is `var + nugget`;
nonzero lags use `var * correlation(lag)`. `var` is the partial variance and
`nugget` is measurement error. Spatial structures with `group=` gate the
correlated part with the same-group delta and use generic `coords` resolved
within the group.

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
V^m_i = Z_i G Z_i^\top + R_i
\]

for each group `i`, with zero covariance across groups. Both sum and matrix
forms are available.

### 7.2 Marginal → conditional: bounded exact decomposition

`MarginalLMM.is_decomposable()` runs a bounded algorithm; it never attempts
general symbolic equivalence:

1. If the covariance carries `InducedCovariance` provenance, return the stored
   source model.
2. Canonicalize \(V^m_i\) into group-block form: diagonal blocks plus
   per-factor block structure.
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
  pedagogy/
    blackboard.md
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
- **Model specification:** fixed/random/residual/marginal/GEE specs, including
  `Group`/`Covariate` index definitions and canonical `Y_i`, `Y[:, t]`, and
  `Y_{it}` notation.
- **Covariance structures:** formulas and parameter constraints.
- **Moments:** expectation, variance, covariance, correlation, ICC, canonical
  group/occasion blocks, and `within=` row queries.
- **Variograms:** raw vs residual vs conditional; same-group vs cross-group;
  `within=` delta resolution.
- **Marginal vs conditional:** derivation, equivalence, decomposition limits.
- **GEE:** working covariance, families, equivalence with marginal LMMs.
- **Forms:** sum notation vs matrix notation, required `form`, conversion.
- **Blackboard:** canonical `ModelSymbols`, `E`/`Var`/`Cov`, conditional
  moments, and law of total variance.
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
- Canonical index: `Group`/`Covariate` index fields; `model.group_index` and
  `occasion_index`; `Y_i` row, `Y[:, t]` column, `Y_{it}` scalar; group sizes;
  `V_i` blocks and zero cross-group covariance; dense group/occasion moments;
  group/occasion symbols must differ.
- Stacked engine: observation-row cursor, structural accessors,
  within-group coordinates, `within=` delta resolution, nested/crossed labels,
  GEE cluster and true within-cluster occasion. Row queries are tested as
  internals, not as canonical notation.
- Blackboard: canonical `syms.Y[syms.i]`, `syms.Y[syms.t]`,
  `syms.Y[syms.i, syms.t]`, `syms.b[syms.i]`, group moments, conditional
  moments, and total variance/covariance.
- Constraints: PD tri-state (`True`, `False`, `None`), constraint collection,
  `ParameterConstraintError`.
- Covariance structures: `entry`/`dense` consistency for every G/R structure;
  PD-safe constructors; constraint checks; nugget at zero lag only; grouped
  spatial cross-group zeros.
- Shapes and assumptions: symbolic dimensions; `as_explicit()` is never called
  outside dense mode (monkeypatched guard).
- Sum/matrix consistency for every model and covariance structure.
- Closed forms: RI, RS, RI+RS, AR(1), CS, IID.
- Variograms: \(\gamma(Y_{it},Y_{it})=0\), symmetry, non-negativity for valid
  parameters, raw non-identity formula, and zero cross-group covariance.
- Correlation bounds and positive variance.
- ICC: RI scalar, RS covariate-dependent, GEE raises `NotDefined`.
- Covariogram equals covariance.
- BLUP: matrix vs dense equality; sum-form gating raises `UnsupportedForm`.
- Printers: LaTeX/Markdown snapshot tests for sum, matrix, and both.
- `Summary` schema and `compare_specs` tri-state result.
- Evaluation: `subs` immutability, `evaluate` unresolved-symbol errors,
  `lambdify` callable behavior.

### 10.2 Integration tests

- Dense explicit mode equals `Z_i G Z_i^T + R_i` and `V^m_i`; dense group
  blocks match corresponding submatrices of the stacked matrices.
- Canonical group rows, occasion columns, and scalar entries match the dense
  response matrix; cross-group blocks are zero.
- Conditional → marginal covariance equals `Z_i G Z_i^T + R_i`.
- Marginal → conditional succeeds for CS/RI provenance and registry templates.
- Marginal → conditional raises for AR(1), negative-\(\rho\) CS, general US,
  non-unique splits; `NotDecomposable` reason/suggestion asserted.
- GEE Gaussian/identity matches marginal LMM working covariance.
- GEE Poisson/log and Bernoulli/logit moment expressions.
- BLUP moments match explicit dense formulas for small instances.
- Nested and crossed random effects match hand-derived covariance entries.
- Spatial exponential, Gaussian, and Matérn entries match hand-derived values.
- Grouped spatial covariances are zero across groups; spatial nuggets contribute
  only at zero lag.
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
| Canonical group/occasion indexing | `tests/unit/test_canonical.py`, `tests/integration/test_canonical_dense.py` |
| Blackboard canonical moments | `tests/unit/test_blackboard.py` |
| Required `form` and error taxonomy | `tests/unit/test_forms.py`, `tests/unit/test_errors.py` |
| Sum/matrix consistency | `tests/unit/test_forms_bridge.py` |
| RI, RS, AR(1), CS, IID closed forms | `tests/unit/test_closed_forms.py` |
| Within-group coordinates and delta resolution | `tests/unit/test_indices.py`, `tests/unit/test_moments.py`, `tests/unit/test_variograms.py`, `tests/unit/test_dense.py` |
| Nuggets and grouped spatial covariance | `tests/unit/test_closed_forms.py`, `tests/unit/test_residual_covariances.py`, `tests/unit/test_spatial_covariances.py` |
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
- Blackboard facade: `docs/superpowers/plans/2026-11-10-symlmm-blackboard.md`
- Canonical indexing (0.5.0):
  `docs/superpowers/plans/2026-11-10-symlmm-canonical-index.md`

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
| M8 | Canonical group/occasion indexing (0.5.0): index-bearing definitions, `Y_i`/`Y[:, t]`/`Y_{it}`, group blocks, rebuilt blackboard, docs | `uv run pytest tests/unit/test_canonical.py tests/integration/test_canonical_dense.py tests/unit/test_blackboard.py` passes; docs use canonical notation; issue #1 closed | 3 d |

**Total: ~22 working days.** Estimates are provisional until the task plan is
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
- **Canonical row/column convention.** In a balanced panel, `Y_i = Y[i, :]`
  is a row vector while `V_i = Z_i G Z_i' + R_i` is a square group matrix.
  Document the row-vector convention explicitly and keep `Y[:, t]`/`Y[t]` for
  occasion columns; do not infer a column from a bare integer index.
- **Unbalanced panels.** `Y_i` and `Y_{it}` remain defined, but `Y_i` is not a
  row of a rectangular `Y` matrix in an unbalanced panel. The rectangular `Y`
  matrix and occasion columns require a balanced panel. Reject ragged occasion
  columns with `NotDefined` instead of inventing missing occasions.
- **Primary grouping factor.** The primary group follows a fixed precedence:
  first random block's `Group`, else the residual group definition, else the
  first registered group. The canonical `Y_i` block uses that group only. With
  multiple grouping factors, block covariance that would combine factors is
  `NotDefined`; crossed random effects remain additional terms.
- **Row API drift.** The stacked row API remains for matrix assembly, dense
  mode, variograms, and BLUP. Keep it documented as engine-internal so examples
  do not reintroduce row-index notation as canonical model notation.
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
- All stacked row/form-based public calls require `form`; form/arity errors are
  typed and tested. Canonical group/occasion methods return expressions
  directly and do not take `form`.
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

### Canonical indexing (0.5.0)

- `Group`, `Nested`, `Crossed`, and `Covariate` carry index symbols, and
  `model.group_index`/`occasion_index` come from those definitions; no index
  name is hardcoded.
- `Y_i` is the group response vector; for a balanced panel
  `Y_i = Y[i, :]` is a row, `Y[:, t]`/`Y[t]` is the occasion column, and
  `Y_{it}` is the scalar entry. For an unbalanced panel `Y_i` is not a row of a
  rectangular matrix.
- The primary group and canonical occasion follow the documented precedence;
  with multiple grouping factors, `Y_i` uses the primary group and group/entry
  covariance raises `NotDefined`.
- `V_i = Z_i G Z_i' + R_i`; `Cov(Y_i, Y_j) = 0` for `i != j`; group and entry
  moments are exposed and tested in symbolic and dense modes.
- The blackboard exposes canonical `syms.Y[syms.i]`, `syms.Y[syms.t]`, and
  `syms.Y[syms.i, syms.t]` with `E`/`Var`/`Cov`, conditional moments, and total
  variance/covariance.
- Public docs use canonical notation; the stacked row API is documented as
  engine-internal, not canonical model notation.

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
| No task breakdown | §11 points to executable subsystem and feature task plans with TDD steps and exit criteria. |
| Canonical group/occasion index | §2 decision 8, §3.0, and §5.0 define model B: `i` is the group, `t` the occasion, `Y_i` is the group vector (a row only for balanced panels), `Y[:, t]`/`Y[t]` an occasion column, and `Y_{it}` a scalar. §3.0 fixes primary-group and canonical-occasion precedence. Issue #1 is resolved by this layer. |
| Group/Covariate index fields | §3.0 and §6.1 document `Group`, `Nested`, `Crossed`, and `Covariate` index symbols, defaults, string normalization, and the distinct-index validation. |
| Row-index notation | §3.0, §3.5, and §5.0 mark `mean(row)`, `covariance(row, row)`, variograms, and BLUP as stacked-engine internals, not canonical model notation. |
| Blackboard facade | §5.7 documents `ModelSymbols`, canonical `Y_i`/`Y[:, t]`/`Y_{it}` atoms, `E`/`Var`/`Cov`, conditional moments, and total variance; `docs/pedagogy/blackboard.md` is the worked example. |
| Within-group coordinates and deltas | §3.4 and §5.0 document `within=` delta resolution; §6.1–6.2 document grouped spatial covariances and nuggets. |
| Matrix-form representation | §2 decision 9 defines an abstract matrix layer plus a structured entry layer; M0.5 validates SymPy capabilities. |
| PD/parameter constraints | §2 decision 10, §5.8, §6.1–6.2 define PD-safe constructors, `constraints()`, and tri-state PD checks. |
| Decomposition algorithm | §7.2 replaces "registry match" with a bounded canonicalize/match/solve/validate algorithm. |
| No evaluation API | §2 decision 12 and §5.5 define `subs`, `evaluate`, `lambdify`, and `dense`. |
| Incomplete public surface | §2 decision 13 and §5.1–5.3 add correlation, ICC, covariogram, BLUP, conditional moments, and `variogram(kind=)`. |
| Untested SymPy assumptions | §11 M0.5 adds a feasibility spike with acceptance tests and a decision record. |
| Raw variogram correctness | §3.6 uses the link-general mean difference. |
| Naming collisions | §2 decision 14 and §6.3 separate covariance and working-correlation classes. |
| Test traceability | §10.4 adds a DoD-to-test matrix; §13 expands the definition of done. |
