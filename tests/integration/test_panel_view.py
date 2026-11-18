import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


def test_panel_view_balanced_with_different_raw_coordinates():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
        n=4,
    ).dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])
    panel = model.panel("F", "p")
    assert panel.position_count() == 2
    assert panel.response_matrix().shape == (2, 2)
