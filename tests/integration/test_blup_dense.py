"""Dense BLUP equality tests."""

import sympy as sp

from symlmm.covariances import IID, Scalar, Unstructured
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects, RandomSlope


def test_blup_dense_matches_explicit_formula():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    V = dense.matrix_covariance()
    G = dense.matrix_g()
    Z = dense.matrix_z()
    expected = G - G * Z.T * V.inv() * Z * G
    assert sp.simplify(dense.blup_covariance(form="matrix") - expected) == sp.zeros(2, 2)


def test_blup_dense_joint_block_matches_explicit_assembly():
    b0 = sp.Symbol("b0")
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Unstructured.from_sd_corr([s0, s1], rho_b),
            )
        ],
        residual=IID(var=sigma2),
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])

    Z = dense.matrix_z()
    G = dense.matrix_g()
    V = dense.matrix_covariance()
    assert Z == sp.Matrix(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ]
    )
    block = dense.random[0].cov.dense()
    assert G == sp.diag(block, block)

    y = sp.MatrixSymbol("y", 4, 1)
    y_symbols = sp.Matrix(sp.symbols("y0:4"))
    expected_mean = G * Z.T * V.inv() * (y_symbols - dense.matrix_mean())
    actual_mean = dense.blup_mean(form="matrix").subs(y, y_symbols)
    assert sp.simplify(actual_mean - expected_mean) == sp.zeros(4, 1)

    expected_covariance = G - G * Z.T * V.inv() * Z * G
    assert sp.simplify(dense.blup_covariance(form="matrix") - expected_covariance) == sp.zeros(4, 4)

    assert dense.blup_cross_covariance(1, form="matrix") == Z[1, :] * G
