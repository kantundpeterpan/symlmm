import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, Nested, RandomEffects


def test_three_level_hierarchy_covariance():
    b0 = sp.Symbol("b0")
    s2_e, s2_f, s2_g, s2 = sp.symbols("s2_e s2_f s2_g s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group=Group("E"), intercept=True, cov=Scalar(s2_e)),
            RandomEffects(group=Nested("F", "E"), intercept=True, cov=Scalar(s2_f)),
            RandomEffects(group=Nested("G", "F"), intercept=True, cov=Scalar(s2_g)),
        ],
        residual=IID(var=s2),
        groups=[Group("E"), Nested("F", "E")],
        n=4,
    )
    dense = model.dense(E=[0, 0, 1, 1], F=[0, 0, 0, 0], G=[0, 1, 0, 1])
    assert dense.covariance_entry(0, 0) == s2_e + s2_f + s2_g + s2
    assert dense.covariance_entry(0, 1) == s2_e + s2_f
    assert dense.covariance_entry(0, 2) == 0
    assert dense.covariance_entry(0, 3) == 0
