"""BLUP moment query tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import FormArityError, MissingFormError, UnsupportedForm
from symlmm.forms import DualForm
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def _model():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )


def test_blup_matrix_form():
    model = _model()
    result = model.blup_mean(form="matrix")
    assert result is not None


def test_blup_mean_expression():
    model = _model()
    n, q = model.dimensions()
    p = len(model.fixed.coefficients())
    G = sp.MatrixSymbol("G", q, q)
    Z = sp.MatrixSymbol("Z", n, q)
    X = sp.MatrixSymbol("X", n, p)
    beta = sp.MatrixSymbol("beta", p, 1)
    y = sp.MatrixSymbol("y", n, 1)
    expected = G * Z.T * model.matrix_covariance().inverse() * (y - X * beta)
    assert model.blup_mean(form="matrix") == expected


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
