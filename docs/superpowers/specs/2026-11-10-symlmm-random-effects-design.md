# RandomEffects and Deferred-Minor Cleanup Design

**Status:** proposed
**Scope:** replace covariance-bearing `RandomIntercept`/`RandomSlope` terms with
one joint `RandomEffects` block per group; fix the remaining M2.6 minors.

## Goal

Separate the random-effect design (intercept and slope columns) from the
random-effect covariance structure. A group's intercept and slopes must be able
to share one joint covariance matrix instead of being independent terms.

## RandomEffects API

`RandomSlope` becomes a pure design definition:

```python
@dataclass(frozen=True)
class RandomSlope:
    """A random-slope column in a named coordinate."""

    coordinate: str
    within: bool = True
```

`RandomEffects` owns the group, design, and covariance:

```python
RandomEffects(
    group="subject",
    intercept=True,
    slopes=(RandomSlope("visit_time"),),
    cov=Unstructured.from_sd_corr([s0, s1], rho),
)
```

- `dim = int(intercept) + len(slopes)`.
- `cov.dim` must equal `dim`; mismatch raises `CovarianceConstructionError`.
- At least one intercept or slope is required.
- `z(model, i)` returns the design column vector:
  `[1, visit_time[g, k]]` for the example above.
- Multiple slopes may use different coordinates and different `within` flags.
- `RandomIntercept` is removed. Intercept-only effects are
  `RandomEffects(group, intercept=True, cov=Scalar(tau2))`.

This is a clean break: `RandomIntercept` and the covariance-bearing
`RandomSlope(group, coordinate, cov)` signature are removed. Tests, examples,
and the M3 plan migrate to `RandomEffects`.

## LMM integration

`LMM.random: Iterable[RandomEffects]`. Each element is one joint block for one
grouping factor.

- Duplicate `RandomEffects` for the same group raises `ModelError`; combine the
  intercept/slopes and covariance instead.
- `dimensions()` sums `levels(term.group) * term.dim`.
- `matrix_g()` places one `term.cov.dense()` block per level.
- `matrix_z()` places `term.z(model, i)` columns per level.
- `_random_term_entry()` computes `z_i' G z_j * delta(group_i, group_j)`.
- `conditional_mean()` indexes `b_{group}_{term}[level, coefficient]` over
  `term.dim`.
- BLUP and evaluation reuse `matrix_g`/`matrix_z`; no separate change.

## Coordinates and ICC

Random slopes may declare several coordinates. `LMM(coordinate=...)` remains the
canonical default for `model.coordinate(i)` and single-coordinate ICC queries.
It no longer rejects multiple random-slope coordinates.

- If an explicit canonical coordinate is supplied, it is used.
- Otherwise the residual coordinate is canonical when present.
- Otherwise a single distinct random-slope coordinate is canonical.
- Otherwise `coordinate_name` is `None`; callers must name a coordinate.

`icc(at=...)` must provide a value for every coordinate-varying slope. A single
canonical coordinate accepts `at={"coordinate": value}`. Multiple coordinates
use `at={"visit_time": value, "baseline": value}`. Dense ICC rebuilds the design
from the supplied values rather than substituting numeric literals.

## Deferred minors

1. Base `IndexedModel.design` must not double-count a coordinate that is also a
   covariate.
2. Covariance docstrings must document `coordinate`, `nugget`, `group`, and
   `coords` as applicable.
3. Extract the repeated nugget-constraint append into one helper.
4. Use `_ZERO` consistently in `_nugget_entry`.
5. Add a dense grouped-spatial cross-group zero test.
6. Add per-factor `occasion` mapping coverage.
7. Cache the common within-group symbol per factor rather than minting a new
   one per entry.
8. Make derived occasion sorting robust to mixed value types; document numeric
   `1` vs `1.0` behavior.
9. Add a direct `IndexedModel.design` test.

## Testing

- `RandomEffects` dimension/validation and duplicate-group errors.
- Intercept-only, intercept+slope joint covariance, slopes-only, two-slope
  blocks, and multiple groups.
- Symbolic and dense `matrix_z`/`matrix_g` shapes and values.
- Conditional mean coefficient indexing and BLUP with joint blocks.
- ICC with one and multiple slope coordinates.
- Migration tests for all old `RandomIntercept`/`RandomSlope` usages.
- Minor-fix regression tests.
- Full gate: `uv run pytest -q && uv run ruff check &&
  uv run ruff format --check && uv run mypy src/symlmm`.

## Risks

- Breaking API removal of `RandomIntercept` and covariance-bearing
  `RandomSlope`; bump to 0.3.0 and document the migration table.
- Joint covariance changes numerical results for models that previously used
  independent intercept and slope blocks. Tests must make the intended
  independent case explicit with `BlockDiag`.
- Multiple slope coordinates require relaxing the single-canonical-coordinate
  rule; ICC must be explicit about which value applies to which slope.
