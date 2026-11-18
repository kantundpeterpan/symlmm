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
