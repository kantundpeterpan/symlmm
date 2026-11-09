import pytest
import sympy as sp

from symlmm.errors import ModelError
from symlmm.model import IndexedModel
from symlmm.specs import Crossed, Group, Nested


def test_observation_accessors():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), time="time")
    i, _ = model.indices()

    assert model.group(i, "subject") == sp.IndexedBase("subject", shape=(n,))[i]
    assert model.time(i) == sp.IndexedBase("t", shape=(n,))[i]
    assert model.covariate(i, "time") == sp.IndexedBase("time", shape=(n,))[i]
    assert model.design(i) == sp.Matrix([1, sp.IndexedBase("t", shape=(n,))[i]])


def test_design_includes_time_symbol():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), time="time")
    i, _ = model.indices()

    assert model.design(i)[1] == model.time(i)


def test_unknown_grouping_factor_raises_model_error():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),))
    i, _ = model.indices()

    with pytest.raises(ModelError):
        model.group(i, "missing")


def test_missing_time_coordinate_raises_model_error():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),))
    i, _ = model.indices()

    with pytest.raises(ModelError):
        model.time(i, "time")


def test_nested_label_is_composite():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Nested("visit", within="subject"),))
    i, _ = model.indices()
    label = model.group(i, "visit")
    assert label.base == sp.IndexedBase("subject:visit", shape=(n,))


def test_crossed_label_is_composite():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Crossed("subject", "item"),))
    i, _ = model.indices()
    label = model.group(i, "item")
    assert label.base == sp.IndexedBase("subject:item", shape=(n,))
