"""Tests for model specifications."""

import pytest
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


def test_fixed_slopes_are_immutable():
    b1 = sp.Symbol("b1")
    fixed = Fixed(slopes={"x": b1})

    with pytest.raises(TypeError):
        fixed.slopes["x"] = sp.Symbol("other")


def test_fixed_default_is_empty():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n)
    i, _ = model.indices()
    fixed = Fixed()

    assert fixed.names() == ()
    assert fixed.coefficients() == ()
    assert fixed.mean_at(model, i) == 0
