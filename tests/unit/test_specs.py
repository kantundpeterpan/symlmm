"""Tests for model specifications."""

import sympy as sp

from symlmm.model import IndexedModel
from symlmm.specs import Fixed, Group


def test_fixed_names_and_coefficients():
    b0, b1 = sp.symbols("b0 b1")
    fixed = Fixed(intercept=b0, slopes={"time": b1})

    assert fixed.names() == ("intercept", "time")
    assert fixed.coefficients() == (b0, b1)


def test_fixed_mean_at_observation():
    b0, b1 = sp.symbols("b0 b1")
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), time="time", covariates=("time",))
    i, _ = model.indices()
    fixed = Fixed(intercept=b0, slopes={"time": b1})

    expected = b0 + b1 * model.covariate(i, "time")
    assert sp.simplify(fixed.mean_at(model, i) - expected) == 0
