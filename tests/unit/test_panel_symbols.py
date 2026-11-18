import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


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
