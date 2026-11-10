"""Closed-form covariance entries and ICC tests."""

import pytest
import sympy as sp

from symlmm.covariances import AR1, IID, Diagonal, Scalar
from symlmm.errors import MissingFormError, ModelError
from symlmm.model import LMM
from symlmm.simplify import canonicalize
from symlmm.specs import Fixed, Group, RandomEffects, RandomSlope


def test_random_intercept_closed_form():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
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
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
    )
    i, j = model.indices()
    inside_i = model.coordinate(i, "visit_time", within="subject")
    inside_j = model.coordinate(j, "visit_time", within="subject")
    expected = (
        sigma2
        * rho ** sp.Abs(inside_i - inside_j)
        * sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject"))
    )
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def test_ar1_uses_two_level_within_coordinate():
    b0 = sp.Symbol("b0")
    sigma2, rho = sp.symbols("sigma2 rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
    )
    i, j = model.indices()
    inside = model.coordinate(i, "visit_time", within="subject")
    entry = model.residual.entry(i, j, model)
    assert inside in entry.atoms(sp.Indexed)
    assert sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject")) in entry.atoms(
        sp.KroneckerDelta
    )


def test_ar1_within_resolves_group_delta():
    b0 = sp.Symbol("b0")
    sigma2, rho = sp.symbols("sigma2 rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
        coordinate="visit_time",
    )
    i, j = model.indices()

    inside = model.covariance_entry(i, j, within="subject")
    common = sp.Symbol("g_subject", integer=True)
    lag = sp.Abs(
        sp.IndexedBase("visit_time")[common, model.occasion(i, "subject")]
        - sp.IndexedBase("visit_time")[common, model.occasion(j, "subject")]
    )
    assert sp.simplify(inside - sigma2 * rho**lag) == 0
    assert not inside.has(sp.KroneckerDelta)


def test_ar1_nugget_at_zero_lag_only():
    b0 = sp.Symbol("b0")
    sigma2, rho, omega2 = sp.symbols("sigma2 rho omega2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(
            group="subject",
            coordinate="visit_time",
            var=sigma2,
            rho=rho,
            nugget=omega2,
        ),
        coordinate="visit_time",
        n=2,
    )
    dense = model.dense(subject=[0, 0], visit_time=[0, 1])
    assert sp.simplify(dense.covariance(0, 0, form="sum")) == sigma2 + omega2
    assert sp.simplify(dense.covariance(0, 1, form="sum")) == sigma2 * rho


def test_nugget_constraint():
    sigma2, rho, omega2 = sp.symbols("sigma2 rho omega2", positive=True)
    cov = AR1(group="subject", coordinate="x", var=sigma2, rho=rho, nugget=omega2)
    assert any(constraint.message == "omega2 >= 0" for constraint in cov.constraints())


def test_icc_for_random_intercept():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    assert sp.simplify(model.icc(form="sum") - tau2 / (tau2 + sigma2)) == 0


def test_canonicalize_expands_and_simplifies():
    x = sp.Symbol("x")
    assert canonicalize((x + 1) ** 2 - x**2) == 2 * x + 1


def test_ar1_uses_named_coordinate_as_canonical_symbol():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    ar1 = AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho)
    model = LMM(fixed=Fixed(intercept=b0), random=[], residual=ar1)
    i, j = model.indices()

    assert ar1.coordinate == "visit_time"
    entry = ar1.entry(i, j, model)
    inside = sp.IndexedBase("visit_time")[model.group(i, "subject"), model.occasion(i, "subject")]
    assert inside in entry.atoms(sp.Indexed)


def test_residual_group_registration_deduplicates():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
    )

    assert model.groups == (Group("subject"),)


def test_icc_requires_form():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
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
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )

    assert model.icc(form="matrix") is not None
    assert model.icc(form="both") is not None


def test_icc_random_slope_requires_coordinate():
    b0 = sp.Symbol("b0")
    tau2, sigma2 = sp.symbols("tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Diagonal([tau2, tau2]),
            )
        ],
        residual=IID(var=sigma2),
    )
    with pytest.raises(ModelError, match="icc requires at"):
        model.icc(form="sum")


def test_icc_random_slope_evaluates_at_given_coordinate():
    b0 = sp.Symbol("b0")
    tau2, sigma2 = sp.symbols("tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Diagonal([tau2, tau2]),
            )
        ],
        residual=IID(var=sigma2),
    )
    value = model.icc(at={"coordinate": 2}, form="sum")
    assert sp.simplify(value - (tau2 + 4 * tau2) / (tau2 + 4 * tau2 + sigma2)) == 0
    assert not value.has(sp.Indexed)


def test_icc_multiple_slopes_require_each_coordinate_value():
    b0 = sp.Symbol("b0")
    tau2, sigma2 = sp.symbols("tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"), RandomSlope("dose")),
                cov=Diagonal([tau2, tau2, tau2]),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )
    with pytest.raises(ModelError, match="coordinate 'dose'"):
        model.icc(at={"time": 2}, form="sum")


@pytest.mark.parametrize("at", [{"coordinate": None}, {"time": None}])
def test_icc_none_coordinate_value_raises_model_error(at):
    b0 = sp.Symbol("b0")
    tau2, sigma2 = sp.symbols("tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Diagonal([tau2, tau2]),
            )
        ],
        residual=IID(var=sigma2),
    )
    with pytest.raises(ModelError, match="coordinate 'time'"):
        model.icc(at=at, form="sum")
