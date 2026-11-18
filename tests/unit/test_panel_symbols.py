import warnings

import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import NotDefined
from symlmm.model import LMM
from symlmm.specs import Covariate, Fixed, RandomEffects


def _panel_model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
        n=4,
    )
    return model.dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])


def test_panel_symbols_bind_to_panel_view():
    panel = _panel_model().panel("F", "p")
    syms = panel.symbols()
    assert syms.Y[:, :].shape == panel.response_matrix().shape
    assert syms.t == panel.position.index


def test_model_panel_returns_panel_view():
    model = _panel_model()
    panel = model.panel("F", "p")
    assert panel.response_matrix().shape == (2, 2)


def test_model_wires_blocks_and_decomposition():
    model = _panel_model()
    block = model.block("F", 0)
    other = model.block("F", 1)
    assert block.size() == 2
    assert model.block_mean("F", 0) == block.mean()
    assert model.block_covariance("F", 0) == block.covariance()
    assert model.block_cross_covariance("F", 0, "F", 1) == block.cross_covariance(other)
    values = model.covariance_terms(0, 1)
    assert sum(values.values()) == model.covariance_entry(0, 1)
    assert model.partial_covariance(block, other, exclude=("F",)) == block.partial_covariance(
        other, exclude=("F",)
    )


def _indexed_panel_model():
    b0, b1 = sp.symbols("b0 b1")
    t = sp.Symbol("t", integer=True)
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0, slopes={Covariate("time", index=t): b1}),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="time",
        n=4,
    )
    return model.dense(F=[0, 0, 1, 1], time=[0, 1, 0, 1])


def _two_factor_panel_model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    o2 = sp.Symbol("o2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="F", intercept=True, cov=Scalar(t2)),
            RandomEffects(group="G", intercept=True, cov=Scalar(o2)),
        ],
        residual=IID(var=s2),
        coordinate="p",
        n=6,
    )
    return model.dense(
        F=[0, 0, 0, 1, 1, 1],
        G=[0, 1, 2, 0, 1, 2],
        p=[0, 1, 2, 0, 1, 2],
    )


def _no_coordinate_model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
    )


def test_panel_symbols_do_not_emit_deprecation_warnings():
    panel = _panel_model().panel("F", "p")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        panel.symbols()


def test_panel_preserves_canonical_position_index():
    model = _indexed_panel_model()
    panel = model.panel("F")
    assert panel.position is not None
    assert panel.position.index == model.occasion_index
    syms = panel.symbols()
    assert syms.t == model.occasion_index
    assert syms.Y[:, syms.t] == panel.response_column(syms.t)


def test_secondary_factor_panel_uses_panel_factor_for_sizes():
    model = _two_factor_panel_model()
    panel = model.panel("G", "p")
    assert panel.level_count() == 3
    syms = panel.symbols()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert syms.model.group_count() == 3
        assert syms.model.group_size(0) == 2


def test_panel_without_position_rejects_position_moments():
    panel = _no_coordinate_model().panel("F")
    assert panel.position is None
    calls = (
        panel.position_count,
        panel.response_matrix,
        lambda: panel.response_column(0),
        lambda: panel.position_code(0),
        lambda: panel.entry_mean(0, 0),
        lambda: panel.entry_covariance(0, 0, 1),
        lambda: panel.entry_cross_covariance(0, 0, 1, 0),
        lambda: panel.position_mean(0),
        lambda: panel.position_covariance(0, 1),
    )
    for call in calls:
        with pytest.raises(NotDefined, match="position axis"):
            call()
