"""Dense BLUP equality tests."""

import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def test_blup_dense_matches_explicit_formula():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    V = dense.matrix_covariance()
    G = dense.matrix_g()
    Z = dense.matrix_z()
    expected = G - G * Z.T * V.inv() * Z * G
    assert sp.simplify(dense.blup_covariance(form="matrix") - expected) == sp.zeros(2, 2)
