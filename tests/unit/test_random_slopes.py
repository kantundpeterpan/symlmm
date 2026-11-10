"""Random slope and block-diagonal covariance tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, BlockDiag, Diagonal, Scalar, Unstructured
from symlmm.errors import CovarianceConstructionError
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects, RandomSlope


def test_random_slope_closed_form():
    b0 = sp.Symbol("b0")
    s0 = sp.Symbol("s0", positive=True)
    s1 = sp.Symbol("s1", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=_sd_corr(s0, s1, rho_b),
            )
        ],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    ti, tj = (
        model.coordinate(i, "time", within="subject"),
        model.coordinate(j, "time", within="subject"),
    )
    expected = (s0**2 + s0 * s1 * rho_b * (ti + tj) + s1**2 * ti * tj) * sp.KroneckerDelta(
        model.group(i, "subject"), model.group(j, "subject")
    ) + sigma2 * sp.KroneckerDelta(i, j)
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def test_random_slope_within_false_uses_global_coordinate():
    b0 = sp.Symbol("b0")
    s0 = sp.Symbol("s0", positive=True)
    s1 = sp.Symbol("s1", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    term = RandomEffects(
        group="subject",
        intercept=True,
        slopes=(RandomSlope("time", within=False),),
        cov=_sd_corr(s0, s1, rho_b),
    )
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[term],
        residual=IID(var=sigma2),
    )
    i, _ = model.indices()
    assert term.z(model, i) == sp.Matrix([1, model.coordinate(i, "time")])


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
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Unstructured(
                    sp.Matrix(
                        [
                            [tau2 + s0**2, s0 * s1 * rho_b],
                            [s0 * s1 * rho_b, s1**2],
                        ]
                    )
                ),
            ),
        ],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    ti, tj = (
        model.coordinate(i, "time", within="subject"),
        model.coordinate(j, "time", within="subject"),
    )
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
        RandomEffects(
            group="subject",
            intercept=True,
            slopes=(RandomSlope("time"),),
            cov=Scalar(tau2),
        )


def test_random_intercept_rejects_non_scalar_covariance():
    a, b = sp.symbols("a b", positive=True)

    with pytest.raises(CovarianceConstructionError, match="1x1"):
        RandomEffects(group="subject", intercept=True, cov=Diagonal([a, b]))


def _sd_corr(s0, s1, rho):
    from symlmm.covariances import Unstructured

    return Unstructured.from_sd_corr([s0, s1], rho)
