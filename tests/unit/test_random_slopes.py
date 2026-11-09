"""Random slope and block-diagonal covariance tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, BlockDiag, Diagonal, Scalar
from symlmm.errors import CovarianceConstructionError
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept, RandomSlope


def test_random_slope_closed_form():
    b0 = sp.Symbol("b0")
    s0 = sp.Symbol("s0", positive=True)
    s1 = sp.Symbol("s1", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomSlope(
                group="subject",
                time="time",
                cov=_sd_corr(s0, s1, rho_b),
            )
        ],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    ti, tj = model.time(i), model.time(j)
    expected = (s0**2 + s0 * s1 * rho_b * (ti + tj) + s1**2 * ti * tj) * sp.KroneckerDelta(
        model.group(i, "subject"), model.group(j, "subject")
    ) + sigma2 * sp.KroneckerDelta(i, j)
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def test_random_intercept_and_slope_terms_sum():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    s0 = sp.Symbol("s0", positive=True)
    s1 = sp.Symbol("s1", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomIntercept(group="subject", cov=Scalar(tau2)),
            RandomSlope(group="subject", time="time", cov=_sd_corr(s0, s1, rho_b)),
        ],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    ti, tj = model.time(i), model.time(j)
    expected = (tau2 + s0**2 + s0 * s1 * rho_b * (ti + tj) + s1**2 * ti * tj) * sp.KroneckerDelta(
        model.group(i, "subject"), model.group(j, "subject")
    ) + sigma2 * sp.KroneckerDelta(i, j)
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def test_block_diag_entry_dense_and_constraints():
    tau_a, tau_b, tau_c = sp.symbols("tau_a tau_b tau_c", positive=True)
    covariance = BlockDiag([Scalar(tau_a), Diagonal([tau_b, tau_c])], group="subject")

    assert covariance.dim == 3
    assert covariance.group == "subject"
    assert covariance.entry(0, 0) == tau_a
    assert covariance.entry(1, 1) == tau_b
    assert covariance.entry(2, 2) == tau_c
    assert covariance.entry(0, 1) == 0
    assert covariance.entry(1, 2) == 0
    assert covariance.dense() == sp.diag(sp.Matrix([[tau_a]]), sp.diag(tau_b, tau_c))
    assert len(covariance.constraints()) == 3


def test_block_diag_accepts_explicit_matrix_blocks():
    a, b = sp.symbols("a b", positive=True)
    covariance = BlockDiag([sp.Matrix([[a]]), sp.Matrix([[b]])])

    assert covariance.dim == 2
    assert covariance.dense() == sp.diag(a, b)
    assert covariance.entry(0, 0) == a
    assert covariance.entry(1, 1) == b
    assert covariance.entry(0, 1) == 0


def test_block_diag_rejects_empty_blocks():
    with pytest.raises(CovarianceConstructionError, match="at least one block"):
        BlockDiag([])


def test_random_slope_rejects_non_2x2_covariance():
    tau2 = sp.Symbol("tau2", positive=True)

    with pytest.raises(CovarianceConstructionError, match="2x2"):
        RandomSlope(group="subject", time="time", cov=Scalar(tau2))


def _sd_corr(s0, s1, rho):
    from symlmm.covariances import Unstructured

    return Unstructured.from_sd_corr([s0, s1], rho)
