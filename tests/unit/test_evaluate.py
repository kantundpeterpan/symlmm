"""Evaluation helper tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import UnresolvedSymbols
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
        n=4,
    )


def test_subs_is_non_mutating():
    model = _model()
    original = model.__dict__.copy()
    substituted = model.subs({sp.Symbol("b0"): 1})
    assert model.__dict__ == original
    assert substituted is not model


def test_subs_applies_parameter_mapping():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    dense = _model().dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    substituted = dense.subs({tau2: 2, sigma2: 3})
    matrix = substituted.matrix_covariance()
    assert matrix.free_symbols == set()
    assert matrix[0, 0] == 5
    assert matrix[0, 1] == 2


def test_evaluate_reports_unresolved_symbols():
    model = _model()
    with pytest.raises(UnresolvedSymbols):
        model.evaluate({sp.Symbol("b0"): 1})


def test_evaluate_rejects_missing_dense_symbol():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    dense = _model().dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    with pytest.raises(UnresolvedSymbols):
        dense.evaluate({sigma2: 1})
    with pytest.raises(UnresolvedSymbols):
        dense.evaluate({tau2: tau2, sigma2: 1})


def test_evaluate_returns_numeric_matrix():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    dense = _model().dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    matrix = dense.evaluate({sp.Symbol("b0"): 0, tau2: 2, sigma2: 3})
    assert matrix.free_symbols == set()
    assert matrix[0, 0] == 5


def test_lambdify_returns_callable_over_arguments():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    dense = _model().dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    function = dense.lambdify([tau2, sigma2])
    result = function(2.0, 3.0)
    assert result.shape == (4, 4)
    assert result[0, 0] == pytest.approx(5.0)
    assert result[0, 1] == pytest.approx(2.0)
