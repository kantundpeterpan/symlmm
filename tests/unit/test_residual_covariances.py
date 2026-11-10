"""Residual covariance structure tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, CompoundSymmetry, Heteroscedastic, Toeplitz
from symlmm.errors import CovarianceConstructionError, ModelError
from symlmm.model import IndexedModel
from symlmm.specs import Group


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
            time="time",
            variances=[sigma_a, sigma_b],
            correlations=[rho],
        )


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
    assert messages == ["s0 > 0", "s1 > 0"]


def test_heteroscedastic_rejects_concrete_out_of_range():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    model = _model()
    cov = Heteroscedastic(variances=[s0, s1])
    with pytest.raises(ModelError, match="out of range"):
        cov.entry(2, 2, model)
    with pytest.raises(ModelError, match="out of range"):
        cov.entry(2, 0, model)
