# SymPy capability spike — decision record

**Date:** 2026-11-09  
**Environment:** Python 3.14.5, SymPy 1.14.0  
**Test:** `tests/unit/test_sympy_capabilities.py`  
**Script:** `scripts/spike_sympy.py`

## Results

| Capability | Result | Implication |
|---|---|---|
| `MatrixSymbol` with symbolic dimensions | Supported | Keep symbolic shapes `(n, p, q)`. |
| `(Z * G * Z.T)[i, j]` expands to `Sum` | Supported | Matrix-to-sum bridge can read `MatMul` entries; dummy indices are `_i_1`, `_i_2` with 0-based bounds. |
| `BlockDiagMatrix` | Explicit varargs only | Use `BlockDiagMatrix(A, B)`; a list argument raises `SympifyError`. No symbolic block count. |
| `ElementwiseApplyFunction` | Supported | Use for `g^{-1}(X beta)` and elementwise variance functions; entry access works (`exp(x[0,0])`). |
| `DiagonalOf(V)` | Column vector `(n, 1)` | Variance in matrix form is a vector of variances, not a diagonal matrix. |
| `DiagonalMatrix` | Not a square matrix in SymPy 1.14 | Do not use it for covariance/correlation construction; it reports shape `(n, 1)` and `as_explicit()` is malformed. |
| `KroneckerDelta` sums | Not auto-collapsed | `.doit()` yields `Piecewise`; substituting a concrete index collapses it. Rewrite rules are required for closed forms. |
| `HadamardPower` | Supported | Available for elementwise powers if needed. |

## Decisions

**D-S1 — Abstract matrix layer.** Matrix form keeps `V`, `G`, `R`, `Z`, `X` as `MatrixSymbol` objects. Structured block-diagonal and concatenation are not represented symbolically; they appear only in entry/sum form or dense mode.

**D-S2 — Variance matrix form is a vector.** `variance(form="matrix")` returns `DiagonalOf(V)` with shape `(n, 1)`. Documentation and type hints must say "variance vector", not "diagonal matrix".

**D-S3 — Correlation matrix form is abstract.** Because `DiagonalMatrix` is unusable, `correlation(form="matrix")` returns an abstract `MatrixSymbol("Corr", n, n)`. The structured correlation is available through `correlation(i, j, form="sum")` and `EntryMap`.

**D-S4 — Dense block diagonal uses varargs.** Dense assembly calls `BlockDiagMatrix(*blocks)`, never `BlockDiagMatrix(blocks)`.

**D-S5 — Sum canonicalization is explicit.** `simplify.py` must implement Kronecker-delta and bound-normalization rewrite rules. Closed-form tests substitute concrete indices before comparing.

## Plan changes applied

- T0.5 tests and spike script use varargs `BlockDiagMatrix` and document the vector semantics of `DiagonalOf`.
- T1.5 variance/correlation implementations follow D-S2 and D-S3.
- T2.5 dense `matrix_g()` uses `BlockDiagMatrix(*blocks)`.
- `PLAN.md` §5.1 annotates `variance(form="matrix")` as a variance vector.
