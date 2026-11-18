"""Dense block and panel moment consistency tests."""

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
    return model.dense(subject=[0, 0, 1, 1], time=[1, 0, 1, 0])


def test_dense_block_moments_match_matrix_submatrices():
    model = _balanced_model()
    mean = model.matrix_mean()
    covariance = model.matrix_covariance()
    rows = [[1, 0], [3, 2]]
    panel = model.panel("subject", "time")

    for group, indices in enumerate(rows):
        assert panel.grid.rows_for_level(group) == indices
        block = model.block("subject", group)
        assert block.mean() == mean[block.rows(), :]
        assert block.covariance() == covariance[block.rows(), :][:, block.rows()]

    left = model.block("subject", 0)
    right = model.block("subject", 1)
    assert left.cross_covariance(right) == covariance[left.rows(), :][:, right.rows()]
    assert right.cross_covariance(left) == covariance[right.rows(), :][:, left.rows()]
    assert left.cross_covariance(right) == sp.zeros(2, 2)
    assert left.cross_covariance(left) == left.covariance()


def test_dense_position_moments_match_panel_entries():
    model = _balanced_model()
    mean = model.matrix_mean()
    covariance = model.matrix_covariance()
    rows = [[1, 0], [3, 2]]
    panel = model.panel("subject", "time")
    t, s = 0, 1

    for group, indices in enumerate(rows):
        for position in range(2):
            assert panel.entry_mean(group, position) == mean[indices[position], 0]
            for other in range(2):
                assert (
                    panel.entry_covariance(group, position, other)
                    == covariance[indices[position], indices[other]]
                )

    expected_mean = sp.Matrix([panel.entry_mean(group, t) for group in range(2)])
    expected_covariance = sp.diag(*(panel.entry_covariance(group, t, s) for group in range(2)))

    assert panel.position_mean(t) == expected_mean
    assert panel.position_covariance(t, s) == expected_covariance
    assert panel.position_covariance(t, s)[0, 1] == 0
