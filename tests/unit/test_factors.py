import pytest
import sympy as sp

import symlmm as sl
from symlmm.covariances import IID, Scalar
from symlmm.errors import ModelError
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, Nested, RandomEffects


def _model(groups, random_group, n=4):
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group=random_group, intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=groups,
        n=n,
    )


def test_nested_string_resolves_registered_ancestry():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", "G")
    model = _model([F, G], H)
    factor = model.factor("H")
    assert factor.components == ("F", "G", "H")
    assert factor.label == "F:G:H"
    assert model.factor(("F", "G", "H")) is factor


def test_nested_explicit_spec_carries_ancestry():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", G)
    model = _model([F, G], H)
    assert model.factor("H").components == ("F", "G", "H")


def test_unregistered_string_is_legacy_shallow():
    H = Nested("H", "missing")
    model = _model([], H)
    assert model.factor("H").components == ("missing", "H")


def test_dense_codes_use_full_ancestry():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", "G")
    model = _model([F, G], H)
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    assert [dense.level_code("H", k) for k in range(4)] == [0, 1, 2, 3]


def test_repeated_child_labels_across_ancestors_do_not_collide():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", "G")
    model = _model([F, G], H)
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    assert dense.covariance_entry(0, 2) == 0


def test_duplicate_factor_names_with_different_paths_raise():
    F = Group("F")
    G = Nested("G", F)
    H = Nested("H", "G")
    with pytest.raises(ModelError, match="conflicting"):
        _model([F, G, Group("H")], H)


def test_cycle_raises():
    with pytest.raises(ModelError, match="cycle"):
        _model([Nested("F", "G"), Nested("G", "F")], Group("F"))


def test_crossed_accepts_multiple_components():
    model = _model([], sl.Crossed("F", "G", "H"))
    factor = model.factor("H")
    assert factor.components == ("F", "G", "H")
    assert factor.label == "F:G:H"


def test_random_effects_name_is_stored():
    term = RandomEffects(
        group="F",
        intercept=True,
        cov=Scalar(sp.Symbol("tau2", positive=True)),
        name="custom",
    )
    assert term.name == "custom"


def test_duplicate_factor_name_with_conflicting_index_raises():
    first = Group("F")
    second = Group("F", index=sp.Symbol("f2", integer=True))
    with pytest.raises(ModelError, match="conflicting"):
        _model([first, second], first)


def test_self_nesting_raises():
    with pytest.raises(ModelError, match="cycle"):
        _model([], Nested("F", "F"))


def test_self_nesting_with_explicit_spec_raises():
    with pytest.raises(ModelError, match="cycle"):
        _model([], Nested("F", Group("F")))
