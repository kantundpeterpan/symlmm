import sympy as sp

from symlmm.blocks import Block
from symlmm.covariances import IID, Scalar
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
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    left = Block(dense, dense.factor("H"), 0)
    right = Block(dense, dense.factor("H"), 1)
    assert (
        left.cross_covariance(right) == dense.matrix_covariance()[left.rows(), :][:, right.rows()]
    )


def test_block_mean_and_design():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    block = Block(dense, dense.factor("H"), 0)
    assert block.mean() == dense.matrix_mean()[block.rows(), :]
    assert block.design() == dense.matrix_x()[block.rows(), :]


def test_symbolic_block_covariance_shape():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    assert block.covariance().shape == (block.size(), block.size())
