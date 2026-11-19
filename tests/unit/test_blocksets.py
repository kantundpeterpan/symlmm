import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar, Unstructured
from symlmm.errors import ModelError, NotDefined
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


def _ri_symbolic_model():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)

    return (
        LMM(
            fixed=Fixed(),
            random=[
                RandomEffects(group=Group("subject"), intercept=True, cov=Scalar(tau2)),
            ],
            residual=IID(var=sigma2),
        ),
        tau2,
        sigma2,
    )


def _two_factor_symbolic_model():
    b0 = sp.Symbol("b0", real=True)
    tau2, omega2, sigma2 = sp.symbols("tau2 omega2 sigma2", positive=True)

    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group=Group("subject"), intercept=True, cov=Scalar(tau2)),
            RandomEffects(group=Group("item"), intercept=True, cov=Scalar(omega2)),
        ],
        residual=IID(var=sigma2),
    )


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


def test_blockset_ri_shrinkage_closed_form():
    model, _, _ = _ri_symbolic_model()
    i = sp.Symbol("i", integer=True)
    j = sp.Symbol("j", integer=True)
    left = model.block("subject", i)
    right = model.block("subject", j)
    blocks = model.block_set(left, right)

    S = blocks.shrinkage("subject")

    assert S.shape == (2, blocks.size())
    assert S.has(sp.OneMatrix(1, left.size()))
    assert S.has(sp.OneMatrix(1, right.size()))
    assert not any(
        isinstance(symbol, sp.Inverse) and symbol.shape == (blocks.size(), blocks.size())
        for symbol in S.atoms(sp.Inverse)
    )


def test_blockset_dense_ri_shrinkage_closed_form():
    model, tau2, sigma2 = _ri_symbolic_model()
    dense = model.dense(subject=[0, 0, 1, 1, 1])
    blocks = dense.block_set(
        dense.block("subject", 0),
        dense.block("subject", 1),
    )

    S = blocks.shrinkage("subject")
    expected = sp.BlockDiagMatrix(
        (tau2 / (sigma2 + 2 * tau2)) * sp.OneMatrix(1, 2),
        (tau2 / (sigma2 + 3 * tau2)) * sp.OneMatrix(1, 3),
    )

    assert S.shape == (2, 5)
    assert (S - expected).equals(sp.zeros(2, 5)) is True


def test_blockset_precision_matches_dense_inverse():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    blocks = dense.block_set(
        dense.block("subject", 0),
        dense.block("subject", 1),
    )
    expected = dense.matrix_covariance()[0:8, :][:, 0:8].inv()

    assert (blocks.precision() - expected).equals(sp.zeros(8, 8)) is True


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

    assert (actual - expected).equals(sp.zeros(4, 1)) is True


def test_blockset_blup_covariance_matches_global_dense():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    blocks = dense.block_set(
        dense.block("subject", 0),
        dense.block("subject", 1),
    )

    actual = blocks.blup_covariance(term="subject")
    expected = dense.blup_covariance(form="matrix")

    assert (actual - expected).equals(sp.zeros(4, 4)) is True


def test_blockset_cross_covariance_matches_global_dense_slice():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    left = dense.block_set(dense.block("subject", 0))
    right = dense.block_set(dense.block("subject", 1))

    actual = left.cross_covariance(right)
    expected = dense.matrix_covariance()[0:4, :][:, 4:8]

    assert actual == expected


def test_blockset_cross_covariance_rejects_different_models():
    model, i = _ri_rs_symbolic_model()
    other, _ = _ri_rs_symbolic_model()
    blocks = model.block_set(model.block("subject", i))
    other_blocks = other.block_set(other.block("subject", i))

    with pytest.raises(ModelError, match="same model"):
        blocks.cross_covariance(other_blocks)


def test_blockset_dense_term_design_rejects_unknown_level():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 1, 1],
        time=[0, 1, 0, 1],
    )
    blocks = dense.block_set(dense.block("subject", 99))

    with pytest.raises(ModelError, match="unknown block level"):
        blocks.random_design(term="subject")


def test_blockset_dense_term_design_pins_global_matrix_z_columns():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 1)
    blocks = dense.block_set(block)

    expected = dense.matrix_z()[block.rows(), 2:4]

    assert blocks.random_design(term="subject") == expected


def test_blockset_symbolic_precision_has_no_full_inverse():
    model, i = _ri_rs_symbolic_model()
    j = sp.Symbol("j", integer=True)
    blocks = model.block_set(
        model.block("subject", i),
        model.block("subject", j),
    )

    P = blocks.precision()

    assert P.shape == (blocks.size(), blocks.size())
    assert not any(
        isinstance(atom, sp.Inverse) and atom.shape == (blocks.size(), blocks.size())
        for atom in P.atoms(sp.Inverse)
    )


@pytest.mark.parametrize(
    "method",
    ["precision", "shrinkage", "blup_covariance", "blup_mean"],
)
def test_blockset_structured_queries_reject_multiple_random_terms(method):
    model = _two_factor_symbolic_model()
    blocks = model.block_set(model.block("subject", 0))
    args = () if method == "precision" else ("subject",)

    with pytest.raises(NotDefined, match="single random-effect term"):
        getattr(blocks, method)(*args)


def test_blockset_blup_requires_one_factor():
    model = _two_factor_symbolic_model()
    blocks = model.block_set(model.block("subject", 0), model.block("item", 0))

    with pytest.raises(ModelError, match="one factor"):
        blocks.blup_mean()


def test_blockset_random_design_requires_term_for_mixed_factors():
    model = _two_factor_symbolic_model()
    blocks = model.block_set(model.block("subject", 0), model.block("item", 0))

    with pytest.raises(ModelError, match="requires a term"):
        blocks.random_design()


def test_blockset_structured_random_design_rejects_term_from_other_factor():
    model = _two_factor_symbolic_model()
    blocks = model.block_set(model.block("subject", 0), model.block("item", 0))

    with pytest.raises(ModelError, match="structured_random_design"):
        blocks.structured_random_design("subject")
