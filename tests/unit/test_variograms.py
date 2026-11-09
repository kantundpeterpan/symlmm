"""Variogram and covariogram query tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import MissingFormError
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


def test_variogram_is_zero_on_diagonal():
    model = _model()
    i, _ = model.indices()
    assert sp.simplify(model.variogram(i, i, kind="residual", form="sum")) == 0


def test_kind_is_required():
    model = _model()
    i, j = model.indices()
    with pytest.raises(TypeError):
        model.variogram(i, j, form="sum")


def test_covariogram_equals_covariance():
    model = _model()
    i, j = model.indices()
    assert (
        sp.simplify(model.covariogram(i, j, form="sum") - model.covariance(i, j, form="sum")) == 0
    )


def test_form_is_required():
    model = _model()
    i, j = model.indices()
    with pytest.raises(MissingFormError):
        model.variogram(i, j, kind="residual")
    with pytest.raises(MissingFormError):
        model.covariogram(i, j)


def test_invalid_kind_raises():
    model = _model()
    i, j = model.indices()
    with pytest.raises(ValueError):
        model.variogram(i, j, kind="bogus", form="sum")


def test_residual_variogram_formula():
    model = _model()
    i, j = model.indices()
    expected = (
        model.covariance_entry(i, i)
        + model.covariance_entry(j, j)
        - 2 * model.covariance_entry(i, j)
    ) / 2
    assert sp.simplify(model.variogram(i, j, kind="residual", form="sum") - expected) == 0


def test_conditional_variogram_uses_residual_covariance():
    model = _model()
    i, j = model.indices()
    expected = (
        model.residual.entry(i, i, model)
        + model.residual.entry(j, j, model)
        - 2 * model.residual.entry(i, j, model)
    ) / 2
    assert sp.simplify(model.variogram(i, j, kind="conditional", form="sum") - expected) == 0


def test_raw_variogram_includes_mean_difference():
    b0 = sp.Symbol("b0")
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0, slopes={"x": b1}),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    mean_difference = b1 * (model.covariate(i, "x") - model.covariate(j, "x"))
    expected = (
        mean_difference**2 / 2
        + (
            model.covariance_entry(i, i)
            + model.covariance_entry(j, j)
            - 2 * model.covariance_entry(i, j)
        )
        / 2
    )
    assert sp.simplify(model.variogram(i, j, kind="raw", form="sum") - expected) == 0


def test_matrix_and_both_forms():
    model = _model()
    i, j = model.indices()
    entry = model.variogram(i, j, kind="residual", form="sum")
    matrix = model.variogram(i, j, kind="residual", form="matrix")
    both = model.variogram(i, j, kind="residual", form="both")
    assert matrix == sp.MatrixSymbol("Gamma", model.n, model.n)[i, j]
    assert isinstance(both, DualForm)
    assert both.matrix == matrix
    assert sp.simplify(both.sum - entry) == 0


def test_covariogram_matrix_form_matches_covariance():
    model = _model()
    i, j = model.indices()
    assert model.covariogram(i, j, form="matrix").dummy_eq(model.covariance(i, j, form="matrix"))
