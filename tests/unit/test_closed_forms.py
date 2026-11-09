"""Closed-form covariance entries and ICC tests."""

import sympy as sp

from symlmm.covariances import AR1, IID, Scalar
from symlmm.model import LMM
from symlmm.simplify import canonicalize
from symlmm.specs import Fixed, RandomIntercept


def test_random_intercept_closed_form():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    expected = tau2 * sp.KroneckerDelta(
        model.group(i, "subject"), model.group(j, "subject")
    ) + sigma2 * sp.KroneckerDelta(i, j)
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def test_ar1_residual_closed_form():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(group="subject", time="time", var=sigma2, rho=rho),
    )
    i, j = model.indices()
    expected = (
        sigma2
        * rho ** sp.Abs(model.time(i) - model.time(j))
        * sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject"))
    )
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def test_icc_for_random_intercept():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    assert sp.simplify(model.icc(form="sum") - tau2 / (tau2 + sigma2)) == 0


def test_canonicalize_expands_and_simplifies():
    x = sp.Symbol("x")
    assert canonicalize((x + 1) ** 2 - x**2) == 2 * x + 1
