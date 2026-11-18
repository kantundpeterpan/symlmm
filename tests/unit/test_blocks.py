import pytest
import sympy as sp

from symlmm.blocks import (
    Block,
    block_coefficients,
    block_cross_residual,
    block_mean_symbol,
    block_random_design,
    block_residual,
    block_size_symbol,
    block_term_design,
)
from symlmm.covariances import IID, Scalar
from symlmm.decomposition import terms
from symlmm.errors import ModelError, NotDefined
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, Nested, RandomEffects


def _hierarchical():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="F", intercept=True, cov=Scalar(t2)),
            RandomEffects(group=Nested("H", "G"), intercept=True, cov=Scalar(t2)),
        ],
        residual=IID(var=s2),
        groups=[Group("F"), Nested("G", "F")],
        n=4,
    )


def test_block_dense_covariance_matches_submatrix():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    block = Block(dense, dense.factor("H"), 0)
    rows = block.rows()
    assert rows == [0, 1]
    assert block.covariance() == dense.matrix_covariance()[rows, :][:, rows]


def test_block_cross_covariance_matches_submatrix():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 0, 0], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    left = Block(dense, dense.factor("H"), 0)
    right = Block(dense, dense.factor("H"), 1)
    assert (
        left.cross_covariance(right) == dense.matrix_covariance()[left.rows(), :][:, right.rows()]
    )
    assert left.cross_covariance(right) != sp.zeros(2, 2)


def test_block_mean_and_design():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    block = Block(dense, dense.factor("H"), 0)
    assert block.mean() == dense.matrix_mean()[block.rows(), :]
    assert block.design() == dense.matrix_x()[block.rows(), :]


def test_block_dense_random_design_residual_and_effect():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    block = Block(dense, dense.factor("H"), 0)
    rows = block.rows()
    assert block.size() == 2
    assert block.random_design() == dense.matrix_z()[rows, :]
    assert block.residual() == dense.matrix_r()[rows, :][:, rows]
    assert block.random_effect() == block_coefficients(dense.factor("H"), 0, 1)


def test_symbolic_block_covariance_shape():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    assert block.covariance().shape == (block.size(), block.size())


def test_symbolic_block_size_mean_and_design():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    factor = model.factor("H")
    block = Block(model, factor, level)
    p = len(model.fixed.coefficients())
    assert block.size() == block_size_symbol(factor, level)
    assert block.mean() == block_mean_symbol(factor, level, p) * sp.Matrix(
        model.fixed.coefficients()
    )
    assert block.design() == block_mean_symbol(factor, level, p)


def test_symbolic_random_design_and_residual():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    factor = model.factor("H")
    block = Block(model, factor, level)
    q = model.dimensions()[1]
    assert block.random_design() == block_random_design(factor, level, q)
    assert block.residual() == block_residual(factor, level)


def test_symbolic_random_effect_and_missing_term():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    assert block.random_effect() == block_coefficients(model.factor("H"), level, 1)
    with pytest.raises(NotDefined, match="no random-effect"):
        Block(model, model.factor("G"), level).random_effect()


def test_symbolic_cross_covariance_uses_term_designs():
    model = _hierarchical()
    left_factor = model.factor("H")
    right_factor = model.factor("F")
    left_level = sp.Symbol("h", integer=True)
    right_level = sp.Symbol("f", integer=True)
    left = Block(model, left_factor, left_level)
    right = Block(model, right_factor, right_level)
    expected = block_cross_residual(left_factor, left_level, right_factor, right_level)
    for term in terms(model):
        expected += (
            block_term_design(left_factor, left_level, term)
            * term.random.cov.dense()
            * block_term_design(right_factor, right_level, term).T
        )
    assert left.cross_covariance(right) == expected


def test_cross_covariance_requires_the_same_model():
    columns = {"F": [0, 0, 1, 1], "G": [0, 0, 0, 0], "H": [0, 0, 1, 1]}
    left_model = _hierarchical().dense(**columns)
    right_model = _hierarchical().dense(**columns)
    left = Block(left_model, left_model.factor("H"), 0)
    right = Block(right_model, right_model.factor("H"), 1)
    with pytest.raises(ModelError, match="same model"):
        left.cross_covariance(right)


def test_block_rows_require_dense_mode():
    model = _hierarchical()
    block = Block(model, model.factor("H"), 0)
    with pytest.raises(NotDefined, match="dense"):
        block.rows()


def test_block_covariance_terms_sum_to_total():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    terms = block.covariance_terms()
    zero = sp.zeros(block.size(), block.size())
    assert sum(terms.values(), zero) == block.covariance()
    assert list(terms)[-1] == "residual"


def test_block_partial_covariance_excludes_term():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    terms = block.covariance_terms()
    excluded = next(iter(terms))
    assert block.partial_covariance(block, exclude=[excluded]) == (
        block.covariance() - terms[excluded]
    )


def test_block_partial_covariance_excludes_everything():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    exclude = tuple(block.covariance_terms())
    assert block.partial_covariance(block, exclude=exclude) == sp.zeros(block.size(), block.size())


def test_block_unknown_exclusion_raises():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    with pytest.raises(ModelError, match="unknown term"):
        block.partial_covariance(block, exclude=["missing"])


def test_block_partial_covariance_distinct_blocks_excludes_everything():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 0, 0], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    left = Block(dense, dense.factor("H"), 0)
    right = Block(dense, dense.factor("H"), 1)
    assert left.rows() != right.rows()
    exclude = tuple(left.covariance_terms())
    assert left.partial_covariance(right, exclude=exclude) == sp.zeros(2, 2)


def test_symbolic_covariance_terms_sum_to_total():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    values = block.covariance_terms()
    zero = sp.ZeroMatrix(block.size(), block.size())
    assert sum(values.values(), zero) == block.covariance()
    assert list(values)[-1] == "residual"


def test_symbolic_partial_covariance_excludes_everything():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    exclude = tuple(block.covariance_terms())
    assert block.partial_covariance(block, exclude=exclude) == sp.ZeroMatrix(
        block.size(), block.size()
    )


def test_random_design_rejects_unknown_terms():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    with pytest.raises(ModelError, match="unknown term"):
        block.random_design(term="missing")
    with pytest.raises(ModelError, match="unknown term"):
        block.random_design(term=terms(model)[0])
