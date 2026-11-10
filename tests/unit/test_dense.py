"""Dense-mode matrix assembly tests."""

import pytest
import sympy as sp

from symlmm.covariances import AR1, IID, Diagonal, Scalar
from symlmm.errors import ModelError
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept, RandomSlope


def _intercept_model():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=4,
    )


def test_dense_covariance_is_concrete():
    tau2 = sp.Symbol("tau2", positive=True)
    model = _intercept_model()
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    matrix = dense.matrix_covariance()
    assert matrix.shape == (4, 4)
    assert matrix[0, 1] == tau2
    assert matrix[0, 2] == 0


def test_dense_assembly_matches_z_g_z_transpose_plus_r():
    model = _intercept_model()
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    V = dense.matrix_covariance()
    Z = dense.matrix_z()
    G = dense.matrix_g()
    R = dense.matrix_r()
    assert Z.shape == (4, 2)
    assert G.shape == (2, 2)
    assert R.shape == (4, 4)
    assert sp.simplify(V - (Z * G * Z.T + R)) == sp.zeros(4, 4)


def test_matrix_x_returns_design_matrix():
    b0 = sp.Symbol("b0")
    beta = sp.Symbol("beta")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0, slopes={"time": beta}),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=4,
    )
    symbolic = model.matrix_x()
    assert symbolic.shape == (4, 2)
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 2, 1, 3])
    assert dense.matrix_x() == sp.Matrix(
        [
            [1, 0],
            [1, 2],
            [1, 1],
            [1, 3],
        ]
    )


def test_dense_random_slope_uses_observation_times():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=sp.Symbol("b0")),
        random=[RandomSlope(group="subject", time="time", cov=Diagonal([tau2, tau2]))],
        residual=IID(var=sigma2),
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 2, 1, 3])
    assert dense.matrix_z() == sp.Matrix(
        [
            [1, 0, 0, 0],
            [1, 2, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 3],
        ]
    )


def test_dense_ar1_matches_observation_lags():
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    model = LMM(
        fixed=Fixed(intercept=sp.Symbol("b0")),
        random=[],
        residual=AR1(group="subject", time="time", var=sigma2, rho=rho),
        n=3,
    )
    dense = model.dense(subject=[0, 0, 0], time=[0, 1, 2])
    R = dense.matrix_r()
    assert R[0, 1] == sigma2 * rho
    assert R[0, 2] == sigma2 * rho**2


def test_dense_requires_consistent_column_lengths():
    model = _intercept_model()
    with pytest.raises(ModelError):
        model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0])


def test_dense_reports_missing_group_column():
    model = _intercept_model()
    dense = model.dense(time=[0, 1, 0, 1])
    with pytest.raises(ModelError):
        dense.matrix_covariance()


def test_dense_nested_uses_composite_level_codes():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.specs import Nested

    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="visit", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=[Nested("visit", within="subject")],
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], visit=[0, 1, 0, 1])
    assert dense.dimensions()[1] == 4
    assert dense.levels("visit") == 4
    assert dense.group(0, "visit") == dense.group(1, "visit") - 1
    assert dense.matrix_z().shape == (4, 4)
    assert dense.matrix_g().shape == (4, 4)


def test_dense_crossed_uses_composite_level_codes():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.specs import Crossed

    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="item", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=[Crossed("subject", "item")],
        n=4,
    )
    dense = model.dense(subject=[0, 0, 1, 1], item=[0, 1, 0, 1])
    assert dense.dimensions()[1] == 4
    assert dense.matrix_z().shape == (4, 4)


def test_dense_design_matches_matrix_x_rows():
    b0 = sp.Symbol("b0")
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0, slopes={"time": b1}),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=3,
    )
    dense = model.dense(subject=[0, 0, 1], time=[0, 2, 1])
    for k in range(3):
        assert dense.design(k) == dense.matrix_x()[k, :].T


def test_dense_icc_random_slope_evaluates_at_given_time():
    b0 = sp.Symbol("b0")
    tau2, sigma2 = sp.symbols("tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomSlope(group="subject", time="time", cov=Diagonal([tau2, tau2]))],
        residual=IID(var=sigma2),
        n=2,
    )
    dense = model.dense(subject=[0, 0], time=[1, 2])
    value = dense.icc(at={"time": 5}, form="sum")
    assert sp.simplify(value - 26 * tau2 / (26 * tau2 + sigma2)) == 0
    assert not value.has(sp.Indexed)
