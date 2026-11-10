"""Dense canonical group and occasion moment consistency tests."""

import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


def _balanced_model() -> LMM:
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="time",
        n=4,
    )
    return model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])


def test_dense_group_moments_match_matrix_submatrices():
    model = _balanced_model()
    mean = model.matrix_mean()
    covariance = model.matrix_covariance()
    rows = [model._group_rows(group) for group in range(model.group_count())]

    for group, indices in enumerate(rows):
        assert model.group_mean(group) == mean[indices, :]
        assert model.group_covariance(group) == covariance[indices, :][:, indices]

    assert model.group_cross_covariance(0, 1) == covariance[rows[0], :][:, rows[1]]
    assert model.group_cross_covariance(1, 0) == covariance[rows[1], :][:, rows[0]]
    assert model.group_cross_covariance(0, 1) == sp.zeros(2, 2)
    assert model.group_cross_covariance(0, 0) == model.group_covariance(0)


def test_dense_occasion_moments_match_group_entries():
    model = _balanced_model()
    t, s = 0, 1

    expected_mean = sp.Matrix([model.entry_mean(group, t) for group in range(2)])
    expected_covariance = sp.diag(*(model.entry_covariance(group, t, s) for group in range(2)))

    assert model.time_mean(t) == expected_mean
    assert model.time_covariance(t, s) == expected_covariance
    assert model.time_covariance(t, s)[0, 1] == 0
