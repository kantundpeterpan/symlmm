"""Joint random-effects block tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, Diagonal, Scalar, Unstructured
from symlmm.errors import CovarianceConstructionError, ModelError
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects, RandomSlope


def test_intercept_only_effects_have_dim_one():
    tau2 = sp.Symbol("tau2", positive=True)
    effects = RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))
    assert effects.dim == 1


def test_joint_intercept_slope_effects_keep_covariance():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    effects = RandomEffects(
        group="subject",
        intercept=True,
        slopes=(RandomSlope("time"),),
        cov=Unstructured.from_sd_corr([s0, s1], rho),
    )
    assert effects.dim == 2
    assert effects.cov.dim == 2


def test_slopes_only_effects_are_supported():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    effects = RandomEffects(
        group="subject",
        slopes=(RandomSlope("time"), RandomSlope("dose")),
        cov=Diagonal([s0, s1]),
    )
    assert effects.dim == 2
    assert effects.coordinates() == ("time", "dose")


def test_effects_require_design_and_matching_covariance():
    tau2 = sp.Symbol("tau2", positive=True)
    with pytest.raises(CovarianceConstructionError, match="intercept or slope"):
        RandomEffects(group="subject", cov=Scalar(tau2))
    with pytest.raises(CovarianceConstructionError, match="2x2"):
        RandomEffects(
            group="subject",
            intercept=True,
            slopes=(RandomSlope("time"),),
            cov=Scalar(tau2),
        )


def test_duplicate_group_blocks_raise():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    with pytest.raises(ModelError, match="multiple RandomEffects"):
        LMM(
            fixed=Fixed(intercept=sp.Symbol("b0")),
            random=[
                RandomEffects(group="subject", intercept=True, cov=Scalar(tau2)),
                RandomEffects(group="subject", intercept=True, cov=Scalar(tau2)),
            ],
            residual=IID(var=sigma2),
        )


def test_multiple_slope_coordinates_are_allowed():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=sp.Symbol("b0")),
        random=[
            RandomEffects(
                group="subject",
                slopes=(RandomSlope("time"), RandomSlope("dose", within=False)),
                cov=Diagonal([s0, s1]),
            )
        ],
        residual=IID(var=sigma2),
    )
    assert model.coordinate_name is None


def test_icc_accepts_coordinate_mapping_for_multiple_slopes():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=sp.Symbol("b0")),
        random=[
            RandomEffects(
                group="subject",
                slopes=(RandomSlope("time"), RandomSlope("dose")),
                cov=Diagonal([s0, s1]),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )
    value = model.icc(at={"time": 2, "dose": 3}, form="sum")
    assert sp.simplify(value - (4 * s0 + 9 * s1) / (4 * s0 + 9 * s1 + sigma2)) == 0
    assert not value.has(sp.Indexed)


def test_joint_covariance_entry_uses_one_block():
    b0 = sp.Symbol("b0")
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    effects = RandomEffects(
        group="subject",
        intercept=True,
        slopes=(RandomSlope("time"),),
        cov=Unstructured.from_sd_corr([s0, s1], rho),
    )
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[effects],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    ti = model.coordinate(i, "time", within="subject")
    tj = model.coordinate(j, "time", within="subject")
    expected = (s0**2 + s0 * s1 * rho * (ti + tj) + s1**2 * ti * tj) * sp.KroneckerDelta(
        model.group(i, "subject"), model.group(j, "subject")
    ) + sigma2 * sp.KroneckerDelta(i, j)
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0
