"""Residual covariance structure tests."""

import pytest
import sympy as sp

from symlmm.covariances import AR1, IID, CompoundSymmetry, Heteroscedastic, Toeplitz
from symlmm.errors import CovarianceConstructionError, ModelError
from symlmm.model import IndexedModel
from symlmm.specs import Covariate, Group


def _model():
    n = sp.Symbol("n", positive=True, integer=True)
    return IndexedModel(n=n, groups=(Group("subject"),), coordinate="time")


def test_compound_symmetry_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)
    model = _model()
    i, j = model.indices()
    cov = CompoundSymmetry(group="subject", var=sigma2, rho=rho)

    assert cov.entry(i, i, model) == sigma2
    symbolic = cov.entry(i, j, model)
    assert symbolic.has(sp.Piecewise)
    assert symbolic.subs({i: 0, j: 0}) == sigma2

    concrete = symbolic.subs({i: 0, j: 1})
    subject_0 = model.group(0, "subject")
    subject_1 = model.group(1, "subject")
    within = concrete.subs({subject_0: 1, subject_1: 1})
    assert sp.simplify(within) == sigma2 * rho
    between = concrete.subs({subject_0: 1, subject_1: 2})
    assert between == 0


def test_iid_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = _model()
    i, _ = model.indices()
    cov = IID(var=sigma2)
    assert cov.entry(i, i, model) == sigma2
    assert cov.entry(0, 1, model) == 0


def test_heteroscedastic_entry():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    model = _model()
    cov = Heteroscedastic(variances=[s0, s1])
    assert cov.entry(0, 0, model) == s0
    assert cov.entry(1, 1, model) == s1
    assert cov.entry(0, 1, model) == 0


def test_heteroscedastic_symbolic_entry():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    model = _model()
    i, j = model.indices()
    cov = Heteroscedastic(variances=[s0, s1])

    diagonal = cov.entry(i, i, model)
    assert diagonal.subs({i: 0}) == s0
    assert diagonal.subs({i: 1}) == s1
    assert cov.entry(i, j, model).has(sp.KroneckerDelta)
    assert cov.entry(i, j, model).subs({i: 0, j: 1}) == 0


class _NumericModel:
    """Minimal model stub exposing numeric group and coordinate accessors."""

    def group(self, i, factor):
        """Return a constant group label."""
        return "subject"

    def coordinate(self, i, name, within=None):
        """Return the observation position as its coordinate value."""
        return i


def test_toeplitz_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho1, rho2 = sp.symbols("rho1 rho2", real=True)
    cov = Toeplitz(
        group="subject",
        coordinate="time",
        variances=[sigma2],
        correlations=[rho1, rho2],
    )
    model = _NumericModel()
    assert cov.entry(0, 0, model) == sigma2
    assert cov.entry(0, 1, model) == sigma2 * rho1
    assert cov.entry(0, 2, model) == sigma2 * rho2
    assert cov.entry(0, 3, model) == 0


def test_zero_nugget_adds_no_constraint():
    sigma2, rho = sp.symbols("sigma2 rho", positive=True)
    ar1 = AR1(group="subject", coordinate="time", var=sigma2, rho=rho)
    toeplitz = Toeplitz(group="subject", coordinate="time", variances=[sigma2], correlations=[rho])
    compound = CompoundSymmetry(group="subject", var=sigma2, rho=rho)

    assert [constraint.message for constraint in ar1.constraints()] == [
        "-1 < rho",
        "rho < 1",
        "sigma2 > 0",
    ]
    assert [constraint.message for constraint in toeplitz.constraints()] == ["sigma2 > 0"]
    assert len(compound.constraints()) == 3


def test_toeplitz_nugget_at_zero_lag_only():
    sigma2, rho1, omega2 = sp.symbols("sigma2 rho1 omega2", positive=True)
    cov = Toeplitz(
        group="subject",
        coordinate="time",
        variances=[sigma2],
        correlations=[rho1],
        nugget=omega2,
    )
    model = _NumericModel()
    assert cov.entry(0, 0, model) == sigma2 + omega2
    assert cov.entry(0, 1, model) == sigma2 * rho1


def test_compound_symmetry_nugget_at_zero_lag_only():
    sigma2, rho, omega2 = sp.symbols("sigma2 rho omega2", positive=True)
    cov = CompoundSymmetry(group="subject", var=sigma2, rho=rho, nugget=omega2)
    model = _model()
    i, j = model.indices()
    assert cov.entry(i, i, model) == sigma2 + omega2
    same_subject = cov.entry(i, j, model).subs({i: 0, j: 1})
    subject_0 = model.group(0, "subject")
    subject_1 = model.group(1, "subject")
    assert sp.simplify(same_subject.subs({subject_0: 1, subject_1: 1})) == sigma2 * rho


def test_compound_symmetry_constraints():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)
    cov = CompoundSymmetry(group="subject", var=sigma2, rho=rho)
    messages = [constraint.message for constraint in cov.constraints()]
    assert messages == ["rho > -1/(2-1)", "rho < 1", "sigma2 > 0"]


def test_compound_symmetry_constraints_at_boundary():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)
    cov = CompoundSymmetry(group="subject", var=sigma2, rho=rho, m_max=3)
    lower, upper, _ = cov.constraints()

    assert lower.message == "rho > -1/(3-1)"
    assert upper.message == "rho < 1"
    assert lower.holds({rho: sp.Rational(1, 2)}) is True
    assert upper.holds({rho: sp.Rational(1, 2)}) is True
    assert lower.holds({rho: sp.Rational(-1, 2)}) is False
    assert lower.holds({rho: sp.Rational(-3, 5)}) is False
    assert upper.holds({rho: 1}) is False
    assert upper.holds({rho: sp.Rational(3, 2)}) is False


def test_compound_symmetry_rejects_small_m_max():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)

    with pytest.raises(CovarianceConstructionError, match="at least 2"):
        CompoundSymmetry(group="subject", var=sigma2, rho=rho, m_max=1)


def test_toeplitz_rejects_multiple_variances():
    sigma_a, sigma_b = sp.symbols("sigma_a sigma_b", positive=True)
    rho = sp.Symbol("rho", real=True)

    with pytest.raises(CovarianceConstructionError, match="exactly one variance"):
        Toeplitz(
            group="subject",
            coordinate="time",
            variances=[sigma_a, sigma_b],
            correlations=[rho],
        )


def test_toeplitz_constraints():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)
    cov = Toeplitz(group="subject", coordinate="time", variances=[sigma2], correlations=[rho])
    constraint = cov.constraints()[0]
    assert constraint.message == "sigma2 > 0"
    assert constraint.holds({sigma2: 1}) is True
    assert constraint.holds({sigma2: 0}) is False


def test_heteroscedastic_constraints():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    cov = Heteroscedastic(variances=[s0, s1])
    messages = [constraint.message for constraint in cov.constraints()]
    assert messages == ["s0 > 0", "s1 > 0"]


def test_heteroscedastic_rejects_concrete_out_of_range():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    model = _model()
    cov = Heteroscedastic(variances=[s0, s1])
    with pytest.raises(ModelError, match="out of range"):
        cov.entry(2, 2, model)
    with pytest.raises(ModelError, match="out of range"):
        cov.entry(2, 0, model)


def test_ar1_and_toeplitz_accept_definitions():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    g = sp.Symbol("g", integer=True)
    t = sp.Symbol("t", integer=True)
    subject = Group("subject", index=g)
    visit = Covariate("visit", index=t)

    ar1 = AR1(group=subject, coordinate=visit, var=sigma2, rho=rho)
    toeplitz = Toeplitz(
        group=subject,
        coordinate=visit,
        variances=[sigma2],
        correlations=[rho],
    )

    assert ar1.group is subject
    assert ar1.coordinate is visit
    assert toeplitz.group is subject
    assert toeplitz.coordinate is visit

    model = _NumericModel()
    assert ar1.entry(0, 1, model) == sigma2 * rho
    assert toeplitz.entry(0, 1, model) == sigma2 * rho


def test_ar1_and_toeplitz_normalize_string_forms():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)

    ar1 = AR1(group="subject", coordinate="time", var=sigma2, rho=rho)
    toeplitz = Toeplitz(
        group="subject",
        coordinate="time",
        variances=[sigma2],
        correlations=[rho],
    )

    assert ar1.group == Group("subject")
    assert ar1.coordinate == Covariate("time")
    assert toeplitz.group == Group("subject")
    assert toeplitz.coordinate == Covariate("time")
