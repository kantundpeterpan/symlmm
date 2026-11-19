import pytest
import sympy as sp

from symlmm.covariances import IID, Unstructured
from symlmm.errors import ModelError
from symlmm.model import LMM
from symlmm.specs import Covariate, Fixed, Group, RandomEffects, RandomSlope


def _ri_rs_symbolic_model():
    b0, b1 = sp.symbols("b0 b1", real=True)
    tau0, tau1, sigma2 = sp.symbols("tau0 tau1 sigma2", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    i = sp.Symbol("i", integer=True)

    subject = Group("subject", index=i)
    time = Covariate("time")

    model = LMM(
        fixed=Fixed(intercept=b0, slopes={time: b1}),
        random=[
            RandomEffects(
                group=subject,
                intercept=True,
                slopes=(RandomSlope(time),),
                cov=Unstructured.from_sd_corr([tau0, tau1], rho_b),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )
    return model, i


def test_blockset_symbolic_shapes():
    model, i = _ri_rs_symbolic_model()
    j = sp.Symbol("j", integer=True)
    left = model.block("subject", i)
    right = model.block("subject", j)
    blocks = model.block_set(left, right)

    assert blocks.size() == left.size() + right.size()
    assert blocks.response().shape == (blocks.size(), 1)
    assert blocks.mean().shape == (blocks.size(), 1)
    assert blocks.design().shape == (blocks.size(), 2)
    assert blocks.random_design(term="subject").shape == (blocks.size(), 4)
    assert blocks.covariance().shape == (blocks.size(), blocks.size())


def test_blockset_rejects_empty_and_duplicate_blocks():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    with pytest.raises(ModelError, match="at least one block"):
        model.block_set()

    with pytest.raises(ModelError, match="duplicate block"):
        model.block_set(block, block)


def test_blockset_rejects_blocks_from_different_models():
    model, i = _ri_rs_symbolic_model()
    other, _ = _ri_rs_symbolic_model()

    with pytest.raises(ModelError, match="same model"):
        model.block_set(model.block("subject", i), other.block("subject", i))


def test_blockset_dense_covariance_matches_global_matrix():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    blocks = dense.block_set(
        dense.block("subject", 0),
        dense.block("subject", 1),
    )
    rows = list(range(8))

    assert blocks.covariance() == dense.matrix_covariance()[rows, :][:, rows]


def test_blockset_shrinkage_shape():
    model, i = _ri_rs_symbolic_model()
    j = sp.Symbol("j", integer=True)
    blocks = model.block_set(
        model.block("subject", i),
        model.block("subject", j),
    )

    S = blocks.shrinkage(term="subject")

    assert S.shape == (4, blocks.size())


def test_blockset_blup_matches_global_dense():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    blocks = dense.block_set(
        dense.block("subject", 0),
        dense.block("subject", 1),
    )
    y = sp.MatrixSymbol("y", dense.n, 1)
    left, right = blocks.blocks
    actual = blocks.blup_mean(term="subject").subs(
        {
            left.response(): y[0:4, :],
            right.response(): y[4:8, :],
        }
    )
    expected = dense.blup_mean(form="matrix")

    assert (actual - expected).equals(sp.zeros(4, 1))


def test_blockset_blup_requires_one_factor():
    model, i = _ri_rs_symbolic_model()
    other_factor = model.factor("subject")
    blocks = model.block_set(model.block("subject", i))

    assert blocks.blocks[0].factor == other_factor
