import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import NotDefined
from symlmm.model import LMM
from symlmm.panel import PanelView
from symlmm.specs import Fixed, Position, RandomEffects


def _model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
        n=4,
    )


def test_panel_uses_position_codes_not_raw_values():
    dense = _model().dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    assert panel.position_count() == 2
    assert panel.response_matrix().shape == (2, 2)


def test_panel_explicit_position_column():
    dense = _model().dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21], visit=["a", "b", "c", "d"])
    panel = PanelView(dense, dense.factor("F"), Position("p", column="visit"))
    assert panel.position_count() == 2


def test_panel_rejects_unbalanced_blocks():
    dense = _model().dense(F=[0, 0, 1], p=[10, 11, 20])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    with pytest.raises(NotDefined, match="balanced"):
        panel.position_count()


def test_panel_entry_moments():
    dense = _model().dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    assert panel.entry_mean(0, 0) == dense.matrix_mean()[0, 0]
    assert panel.entry_covariance(0, 0, 1) == dense.matrix_covariance()[0, 1]
