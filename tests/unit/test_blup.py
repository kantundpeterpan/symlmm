"""BLUP moment query tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar, Unstructured
from symlmm.errors import FormArityError, MissingFormError, UnsupportedForm
from symlmm.forms import DualForm
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects, RandomSlope


def _model():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )


def _joint_model():
    b0 = sp.Symbol("b0")
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Unstructured.from_sd_corr([s0, s1], rho_b),
            )
        ],
        residual=IID(var=sigma2),
    )


def test_blup_matrix_form():
    model = _model()
    result = model.blup_mean(form="matrix")
    assert result is not None


def test_blup_mean_expression():
    model = _model()
    n, q = model.dimensions()
    G = sp.MatrixSymbol("G", q, q)
    Z = sp.MatrixSymbol("Z", n, q)
    y = sp.MatrixSymbol("y", n, 1)
    expected = G * Z.T * model.matrix_covariance().inverse() * (y - model.matrix_mean())
    result = model.blup_mean(form="matrix")
    assert result == expected
    assert result.shape == (q, 1)
    assert result.has(model.matrix_mean())


def test_blup_uses_model_matrix_symbols():
    model = _model()
    n, q = model.dimensions()
    G = sp.MatrixSymbol("G", q, q)
    Z = sp.MatrixSymbol("Z", n, q)
    R = sp.MatrixSymbol("R", n, n)
    covariance_symbols = model.matrix_covariance().free_symbols

    mean_symbols = model.blup_mean(form="matrix").free_symbols
    assert {G, Z} <= mean_symbols
    assert {G, Z} <= covariance_symbols

    covariance = model.blup_covariance(form="matrix").free_symbols
    assert {G, Z, R} <= covariance

    i, _ = model.indices()
    cross = model.blup_cross_covariance(i, form="matrix").free_symbols
    assert {G, Z} <= cross


def test_blup_both_form_returns_dual_form():
    model = _model()
    result = model.blup_mean(form="both")
    assert isinstance(result, DualForm)
    assert result.matrix is not None


def test_blup_sum_form_is_gated():
    model = _model()
    with pytest.raises(UnsupportedForm):
        model.blup_mean(form="sum")


def test_blup_covariance_forms():
    model = _model()
    _, q = model.dimensions()

    matrix = model.blup_covariance(form="matrix")
    assert matrix.shape == (q, q)

    dual = model.blup_covariance(form="both")
    assert isinstance(dual, DualForm)
    assert dual.matrix is not None

    with pytest.raises(UnsupportedForm):
        model.blup_covariance(form="sum")


def test_blup_cross_covariance_forms():
    model = _model()
    i, _ = model.indices()
    _, q = model.dimensions()

    matrix = model.blup_cross_covariance(i, form="matrix")
    assert matrix.shape == (1, q)

    dual = model.blup_cross_covariance(i, form="both")
    assert isinstance(dual, DualForm)
    assert dual.matrix is not None

    with pytest.raises(UnsupportedForm):
        model.blup_cross_covariance(i, form="sum")


def test_blup_joint_block_uses_full_random_dimension():
    model = _joint_model()
    n, q = model.dimensions()
    assert q == 2 * model.levels("subject")
    G = sp.MatrixSymbol("G", q, q)
    Z = sp.MatrixSymbol("Z", n, q)
    R = sp.MatrixSymbol("R", n, n)
    y = sp.MatrixSymbol("y", n, 1)

    mean = model.blup_mean(form="matrix")
    assert mean.shape == (q, 1)
    expected_mean = G * Z.T * model.matrix_covariance().inverse() * (y - model.matrix_mean())
    assert mean == expected_mean

    covariance = model.blup_covariance(form="matrix")
    assert covariance.shape == (q, q)
    expected_covariance = (G.inverse() + Z.T * R.inverse() * Z).inverse()
    assert covariance == expected_covariance

    i, _ = model.indices()
    cross = model.blup_cross_covariance(i, form="matrix")
    assert cross.shape == (1, q)
    assert cross == Z[i, :] * G


def test_blup_requires_form():
    model = _model()
    i, _ = model.indices()
    with pytest.raises(MissingFormError):
        model.blup_mean()
    with pytest.raises(MissingFormError):
        model.blup_covariance()
    with pytest.raises(MissingFormError):
        model.blup_cross_covariance(i)


def test_blup_cross_covariance_requires_index():
    model = _model()
    with pytest.raises(FormArityError):
        model.blup_cross_covariance(form="matrix")
