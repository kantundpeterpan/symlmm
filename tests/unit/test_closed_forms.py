"""Closed-form covariance entries and ICC tests."""

import pytest
import sympy as sp

from symlmm.covariances import AR1, IID, Diagonal, Scalar
from symlmm.errors import MissingFormError, ModelError
from symlmm.model import LMM
from symlmm.simplify import canonicalize
from symlmm.specs import Fixed, Group, RandomIntercept, RandomSlope


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


def test_ar1_uses_named_time_coordinate_as_canonical_symbol():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    ar1 = AR1(group="subject", time="time", var=sigma2, rho=rho)
    model = LMM(fixed=Fixed(intercept=b0), random=[], residual=ar1)
    i, j = model.indices()

    assert ar1.time == "time"
    entry = ar1.entry(i, j, model)
    names = {indexed.base.name for indexed in entry.atoms(sp.Indexed)}
    assert "time" in names
    assert "t" not in names


def test_residual_group_registration_deduplicates():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=AR1(group="subject", time="time", var=sigma2, rho=rho),
    )

    assert model.groups == (Group("subject"),)


def test_icc_requires_form():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )

    with pytest.raises(MissingFormError):
        model.icc()


def test_icc_matrix_and_both_forms_return():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )

    assert model.icc(form="matrix") is not None
    assert model.icc(form="both") is not None


def test_icc_random_slope_requires_time():
    b0 = sp.Symbol("b0")
    tau2, sigma2 = sp.symbols("tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomSlope(group="subject", time="time", cov=Diagonal([tau2, tau2]))],
        residual=IID(var=sigma2),
    )
    with pytest.raises(ModelError, match="icc requires at"):
        model.icc(form="sum")


def test_icc_random_slope_evaluates_at_given_time():
    b0 = sp.Symbol("b0")
    tau2, sigma2 = sp.symbols("tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomSlope(group="subject", time="time", cov=Diagonal([tau2, tau2]))],
        residual=IID(var=sigma2),
    )
    value = model.icc(at={"time": 2}, form="sum")
    assert sp.simplify(value - (tau2 + 4 * tau2) / (tau2 + 4 * tau2 + sigma2)) == 0
    assert not value.has(sp.Indexed)
