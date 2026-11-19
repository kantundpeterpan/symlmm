# BlockSet Workstream Report

STATUS: DONE

SUMMARY:
- Added `BlockSet` with symbolic sizes, response/mean/design, block-diagonal
  random design, block covariance, and block cross-covariance.
- Added `LMM.block_set(*blocks)` and exported `BlockSet` from `symlmm`.
- Added block-set shrinkage, BLUP mean, and BLUP covariance using
  block-diagonal effect design/covariance.
- Dense assembly returns explicit matrices so dense covariance equality holds.
- Symbolic paths use block matrices only; no `.dense()` model evaluation and no
  row access.
- Addressed all BlockSet review findings: multi-factor BLUP rejection test,
  dense `blup_covariance` equality, `cross_covariance` dense/global and
  cross-model tests, `ModelError` for unknown dense block levels, mixed-factor
  `random_design(term=None)` rejection, explicit `.equals(...) is True`
  assertions, and a `matrix_z` column pin for the dense per-level workaround.

COMMITS:
- b767581 feat: add symbolic block sets
- 8739f8d feat: add symbolic block-set BLUP
- bb9a4f5 fix(block-sets): address BlockSet review findings (Tasks 3-4)

TESTS:
- `uv run pytest tests/unit/test_blocksets.py -q` -> 13 passed
- `uv run pytest tests/unit/test_blocksets.py --cov=symlmm.blocksets -q` -> 100%
- `uv run pytest -q` -> 381 passed
- `uv run mypy src/symlmm` -> Success, 19 source files
- `uv run ruff check` -> All checks passed
- `uv run ruff format --check` -> 2 target files already formatted

REPORT:
- Files changed exactly as scoped: `src/symlmm/blocksets.py`,
  `tests/unit/test_blocksets.py`.
- `BlockSet._term` rejects blocks from more than one factor; the BLUP path is
  unchanged and the guard is covered by a two-grouping-factor test.
- `BlockSet.blup_covariance` is covered against `dense.blup_covariance(form="matrix")`.
- `BlockSet.cross_covariance` is covered against the dense global covariance
  slice and rejects block sets from different models.
- `_dense_term_design` uses a level-index dict and raises
  `ModelError("unknown block level")` instead of a bare `ValueError`.
- A dense test pins the per-level workaround to literal global `matrix_z`
  columns for a non-first level.
- `random_design(term=None)` now rejects mixed-factor block sets; a term is
  required in that case.

CONCERNS:
- `Block.random_design(term=...)` remains incorrect for non-first dense levels.
  BlockSet retains its local workaround; a core follow-up should fix
  `blocks.py` and delete the workaround.
