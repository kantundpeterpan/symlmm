import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


def _model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
        n=2,
    ).dense(F=[0, 0], p=[0, 1])


def test_time_mean_warns():
    model = _model()
    with pytest.warns(DeprecationWarning, match="time_mean"):
        model.time_mean(0)


def test_response_matrix_warns():
    model = _model()
    with pytest.warns(DeprecationWarning, match="response_matrix"):
        model.response_matrix()


def _no_coordinate_model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
    )


def test_no_coordinate_block_wrappers_work():
    model = _no_coordinate_model()
    block = model.block("F", 0)
    with pytest.warns(DeprecationWarning, match="group_size"):
        assert model.group_size(0) == block.size()
    with pytest.warns(DeprecationWarning, match="group_mean"):
        assert model.group_mean(0) == block.mean()
    with pytest.warns(DeprecationWarning, match="group_covariance"):
        assert model.group_covariance(0) == block.covariance()
    with pytest.warns(DeprecationWarning, match="fixed_design"):
        assert model.fixed_design(0) == block.design()
    with pytest.warns(DeprecationWarning, match="random_effect"):
        assert model.random_effect(0) == block.random_effect()


def test_no_coordinate_symbols_work():
    model = _no_coordinate_model()
    with pytest.warns(DeprecationWarning, match="symbols"):
        syms = model.symbols()
    assert syms.t is None
    assert syms.i == model.group_index
    assert syms.E(syms.b[syms.i]) == sp.zeros(model.random[0].dim, 1)


def test_group_alias_warns():
    model = _model()
    with pytest.warns(DeprecationWarning, match="level_code"):
        assert model.group(0, "F") == model.level_code("F", 0)


def test_occasion_alias_warns():
    model = _model()
    with pytest.warns(DeprecationWarning, match="position_code"):
        assert model.occasion(0, "F") == 0
