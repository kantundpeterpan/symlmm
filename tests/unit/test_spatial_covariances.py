"""Spatial covariance structure tests."""

import pytest
import sympy as sp

from symlmm.covariances import Exponential, Gaussian, Matern
from symlmm.model import LMM, IndexedModel
from symlmm.specs import Fixed, Group


def test_exponential_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)
    cov = Exponential(var=sigma2, range=phi)
    x0, x1 = sp.symbols("x0 x1", real=True)
    expected = sigma2 * sp.exp(-sp.Abs(x0 - x1) / phi)
    assert sp.simplify(cov.entry_from_distance(sp.Abs(x0 - x1)) - expected) == 0


def test_matern_half_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)
    cov = Matern(var=sigma2, range=phi, nu=sp.Rational(1, 2))
    h = sp.Symbol("h", positive=True)
    assert sp.simplify(cov.entry_from_distance(h) - sigma2 * sp.exp(-h / phi)) == 0


def test_gaussian_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)
    cov = Gaussian(var=sigma2, range=phi)
    h = sp.Symbol("h", positive=True)
    expected = sigma2 * sp.exp(-(h**2) / (2 * phi**2))
    assert sp.simplify(cov.entry_from_distance(h) - expected) == 0


def test_matern_three_halves_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)
    cov = Matern(var=sigma2, range=phi, nu=sp.Rational(3, 2))
    h = sp.Symbol("h", positive=True)
    scaled = h / phi
    expected = sigma2 * (1 + sp.sqrt(3) * scaled) * sp.exp(-sp.sqrt(3) * scaled)
    assert sp.simplify(cov.entry_from_distance(h) - expected) == 0


def test_matern_five_halves_entry():
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)
    cov = Matern(var=sigma2, range=phi, nu=sp.Rational(5, 2))
    h = sp.Symbol("h", positive=True)
    scaled = h / phi
    expected = sigma2 * (1 + sp.sqrt(5) * scaled + 5 * scaled**2 / 3) * sp.exp(-sp.sqrt(5) * scaled)
    assert sp.simplify(cov.entry_from_distance(h) - expected) == 0


def test_matern_rejects_unsupported_nu():
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)
    with pytest.raises(ValueError, match="unsupported Matern nu"):
        Matern(var=sigma2, range=phi, nu=sp.Integer(1))


@pytest.mark.parametrize(
    "factory",
    [
        lambda var, phi: Exponential(var=var, range=phi),
        lambda var, phi: Gaussian(var=var, range=phi),
        lambda var, phi: Matern(var=var, range=phi, nu=sp.Rational(1, 2)),
        lambda var, phi: Matern(var=var, range=phi, nu=sp.Rational(3, 2)),
        lambda var, phi: Matern(var=var, range=phi, nu=sp.Rational(5, 2)),
    ],
)
def test_spatial_constraints(factory):
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)

    constraints = factory(sigma2, phi).constraints()

    assert [constraint.message for constraint in constraints] == ["sigma2 > 0", "phi > 0"]
    var_constraint, range_constraint = constraints
    assert var_constraint.holds({sigma2: 1, phi: 1}) is True
    assert var_constraint.holds({sigma2: -1, phi: 1}) is False
    assert range_constraint.holds({sigma2: 1, phi: 1}) is True
    assert range_constraint.holds({sigma2: 1, phi: -1}) is False


@pytest.mark.parametrize(
    ("cls", "distance"),
    [
        (Exponential, lambda h, phi: sp.exp(-h / phi)),
        (Gaussian, lambda h, phi: sp.exp(-(h**2) / (2 * phi**2))),
        (Matern, lambda h, phi: sp.exp(-h / phi)),
    ],
)
def test_entry_uses_euclidean_distance(cls, distance):
    sigma2 = sp.Symbol("sigma2", positive=True)
    phi = sp.Symbol("phi", positive=True)
    kwargs = {"nu": sp.Rational(1, 2)} if cls is Matern else {}
    cov = cls(var=sigma2, range=phi, **kwargs)
    model = IndexedModel(n=2)
    x0, x1 = model.coordinate(0, "x"), model.coordinate(1, "x")
    y0, y1 = model.coordinate(0, "y"), model.coordinate(1, "y")
    h = sp.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    expected = sigma2 * distance(h, phi)
    assert sp.simplify(cov.entry(0, 1, model) - expected) == 0


def test_spatial_group_gate_and_nugget():
    sigma2, phi, omega2 = sp.symbols("sigma2 phi omega2", positive=True)
    model = IndexedModel(n=2, groups=(Group("subject"),), coordinate="x")
    i, j = model.indices()
    cov = Exponential(var=sigma2, range=phi, group="subject", nugget=omega2)
    entry = cov.entry(i, j, model)
    assert sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject")) in entry.atoms(
        sp.KroneckerDelta
    )
    assert sp.simplify(entry.subs({i: 0, j: 0})) == sigma2 + omega2


def test_spatial_nugget_constraint():
    sigma2, phi, omega2 = sp.symbols("sigma2 phi omega2", positive=True)
    cov = Gaussian(var=sigma2, range=phi, nugget=omega2)
    assert any(constraint.message == "omega2 >= 0" for constraint in cov.constraints())


def test_dense_grouped_spatial_cross_group_entry_is_zero():
    b0 = sp.Symbol("b0")
    sigma2, phi = sp.symbols("sigma2 phi", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=Exponential(var=sigma2, range=phi, group="subject"),
        coordinate="x",
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], x=[0, 0, 0, 0], y=[0, 0, 0, 0])

    assert dense.covariance(0, 2, form="sum") == 0
    assert sp.simplify(dense.covariance(0, 1, form="sum") - sigma2) == 0


def test_spatial_residual_without_group_registers_no_factor():
    b0 = sp.Symbol("b0")
    sigma2, phi = sp.symbols("sigma2 phi", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=Exponential(var=sigma2, range=phi),
        coordinate="x",
        n=2,
    )
    assert model.groups == ()
