"""Residual covariance structure tests."""

import sympy as sp

from symlmm.covariances import IID, CompoundSymmetry, Heteroscedastic, Toeplitz
from symlmm.model import IndexedModel
from symlmm.specs import Group


def _model():
    n = sp.Symbol("n", positive=True, integer=True)
    return IndexedModel(n=n, groups=(Group("subject"),), time="time")


def test_compound_symmetry_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)
    model = _model()
    i, _ = model.indices()
    cov = CompoundSymmetry(group="subject", var=sigma2, rho=rho)
    assert cov.entry(i, i, model) == sigma2
    subject_0 = model.group(0, "subject")
    subject_1 = model.group(1, "subject")
    within = cov.entry(0, 1, model).subs({subject_0: 1, subject_1: 1})
    assert sp.simplify(within) == sigma2 * rho
    between = cov.entry(0, 1, model).subs({subject_0: 1, subject_1: 2})
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


class _NumericModel:
    """Minimal model stub exposing numeric group and time accessors."""

    def group(self, i, factor):
        """Return a constant group label."""
        return "subject"

    def time(self, i, coordinate):
        """Return the observation position as its time value."""
        return i


def test_toeplitz_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho1, rho2 = sp.symbols("rho1 rho2", real=True)
    cov = Toeplitz(
        group="subject",
        time="time",
        variances=[sigma2],
        correlations=[rho1, rho2],
    )
    model = _NumericModel()
    assert cov.entry(0, 0, model) == sigma2
    assert cov.entry(0, 1, model) == sigma2 * rho1
    assert cov.entry(0, 2, model) == sigma2 * rho2
    assert cov.entry(0, 3, model) == 0


def test_compound_symmetry_constraints():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)
    cov = CompoundSymmetry(group="subject", var=sigma2, rho=rho)
    messages = [constraint.message for constraint in cov.constraints()]
    assert messages == ["rho > -1/(m_min - 1)", "sigma2 > 0"]


def test_toeplitz_constraints():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)
    cov = Toeplitz(group="subject", time="time", variances=[sigma2], correlations=[rho])
    constraint = cov.constraints()[0]
    assert constraint.message == "sigma2 > 0"
    assert constraint.holds({sigma2: 1}) is True
    assert constraint.holds({sigma2: 0}) is False


def test_heteroscedastic_constraints():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    cov = Heteroscedastic(variances=[s0, s1])
    messages = [constraint.message for constraint in cov.constraints()]
    assert messages == ["sigma2[0] > 0", "sigma2[1] > 0"]
