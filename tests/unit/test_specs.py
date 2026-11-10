"""Tests for model specifications."""

import pytest
import sympy as sp

from symlmm.covariances import Diagonal
from symlmm.errors import ModelError
from symlmm.model import IndexedModel
from symlmm.specs import Covariate, Crossed, Fixed, Group, Nested, RandomEffects, RandomSlope


def test_fixed_names_and_coefficients():
    b0, b1 = sp.symbols("b0 b1")
    fixed = Fixed(intercept=b0, slopes={Covariate("time"): b1})

    assert fixed.names() == ("intercept", "time")
    assert fixed.coefficients() == (b0, b1)


def test_group_explicit_and_default_indices():
    i = sp.Symbol("i", integer=True)

    assert Group("subject", index=i).index is i
    assert Group("subject").index == sp.Symbol("subject", integer=True)
    assert Nested("visit", within="subject").index == sp.Symbol("subject:visit", integer=True)
    assert Crossed("subject", "item").index == sp.Symbol("subject:item", integer=True)


def test_nested_and_crossed_accept_explicit_indices():
    i = sp.Symbol("i", integer=True)

    assert Nested("visit", within="subject", index=i).index is i
    assert Crossed("subject", "item", index=i).index is i


def test_covariate_explicit_and_default_indices():
    t = sp.Symbol("t", integer=True)

    assert Covariate("time", index=t).index is t
    assert Covariate("time").index == sp.Symbol("time", integer=True)


def test_fixed_slopes_keyed_by_covariate_definitions():
    b1 = sp.Symbol("b1")
    t = sp.Symbol("t", integer=True)
    time = Covariate("time", index=t)
    fixed = Fixed(slopes={time: b1})

    assert tuple(fixed.slopes) == (time,)
    assert fixed.covariates() == (time,)
    assert fixed.names() == ("time",)


def test_fixed_slopes_normalize_string_keys():
    b1 = sp.Symbol("b1")
    fixed = Fixed(slopes={"time": b1})
    (covariate,) = fixed.slopes

    assert isinstance(covariate, Covariate)
    assert covariate == Covariate("time")
    assert fixed.names() == ("time",)


def test_fixed_slopes_mix_strings_and_definitions():
    b1, b2 = sp.symbols("b1 b2")
    t = sp.Symbol("t", integer=True)
    time = Covariate("time", index=t)
    fixed = Fixed(slopes={"dose": b1, time: b2})

    assert fixed.covariates() == (Covariate("dose"), time)
    assert fixed.names() == ("dose", "time")


def test_fixed_slopes_dedupe_identical_definitions():
    b1 = sp.Symbol("b1")
    fixed = Fixed(slopes={"time": b1, Covariate("time"): b1})

    assert tuple(fixed.slopes) == (Covariate("time"),)


def test_fixed_slopes_reject_conflicting_covariate_indices():
    b1, b2 = sp.symbols("b1 b2")
    t = sp.Symbol("t", integer=True)

    with pytest.raises(ModelError, match="conflicting covariate definitions"):
        Fixed(slopes={"time": b1, Covariate("time", index=t): b2})


def test_random_effects_normalize_group_and_slope_definitions():
    tau2 = sp.Symbol("tau2", positive=True)
    i = sp.Symbol("i", integer=True)
    t = sp.Symbol("t", integer=True)
    subject = Group("subject", index=i)
    time = Covariate("time", index=t)

    term = RandomEffects(
        group=subject,
        intercept=True,
        slopes=(RandomSlope(time),),
        cov=Diagonal([tau2, tau2]),
    )

    assert term.group is subject
    assert term.slopes[0].coordinate is time
    assert term.coordinates() == ("time",)

    from_strings = RandomEffects(
        group="subject",
        intercept=True,
        slopes=(RandomSlope("time"),),
        cov=Diagonal([tau2, tau2]),
    )

    assert from_strings.group == Group("subject")
    assert from_strings.slopes[0].coordinate == Covariate("time")


def test_fixed_mean_at_observation():
    b0, b1 = sp.symbols("b0 b1")
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), coordinate="time", covariates=("time",))
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
