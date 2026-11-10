import pytest
import sympy as sp

from symlmm.errors import ModelError
from symlmm.model import IndexedModel
from symlmm.specs import Crossed, Group, Nested


def test_observation_accessors_are_distinct_index_spaces():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), coordinate="coordinate")
    i, _ = model.indices()

    assert model.group(i, "subject") == sp.IndexedBase("g_subject", shape=(n,))[i]
    assert model.occasion(i, "subject") == sp.IndexedBase("k_subject", shape=(n,))[i]
    assert model.coordinate(i) == sp.IndexedBase("coordinate", shape=(n,))[i]


def test_within_coordinate_is_two_level():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), coordinate="visit_time")
    i, _ = model.indices()

    inside = model.coordinate(i, "visit_time", within="subject")
    assert (
        inside
        == sp.IndexedBase("visit_time")[model.group(i, "subject"), model.occasion(i, "subject")]
    )


def test_public_time_alias_is_removed():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), coordinate="visit_time")
    assert not hasattr(model, "time")


def test_missing_canonical_coordinate_raises():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),))
    i, _ = model.indices()

    with pytest.raises(ModelError, match="no coordinate name"):
        model.coordinate(i)


def test_unknown_grouping_factor_raises_model_error():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),))
    i, _ = model.indices()

    with pytest.raises(ModelError):
        model.group(i, "missing")


def test_nested_label_is_composite():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Nested("visit", within="subject"),))
    i, _ = model.indices()
    label = model.group(i, "visit")
    assert label.base == sp.IndexedBase("g_subject_visit", shape=(n,))


def test_crossed_label_is_composite():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Crossed("subject", "item"),))
    i, _ = model.indices()
    label = model.group(i, "item")
    assert label.base == sp.IndexedBase("g_subject_item", shape=(n,))


def test_base_design_does_not_duplicate_coordinate_covariate():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, coordinate="time", covariates=("time", "x"))
    i, _ = model.indices()

    design = model.design(i)
    assert design.shape == (2, 1)
    assert design[0] == model.coordinate(i)
    assert design[1] == model.covariate(i, "x")


def test_within_resolution_reuses_cached_symbol():
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomEffects

    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(sigma2))],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()

    first = model._apply_within(model.group(i, "subject"), i, j, "subject")
    second = model._apply_within(model.group(i, "subject"), i, j, "subject")

    assert first is second


def test_lmm_design_uses_fixed_columns_without_intercept():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomEffects

    model = LMM(
        fixed=Fixed(intercept=None, slopes={"time": b1}),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    i, _ = model.indices()
    assert model.design(i) == sp.Matrix([model.covariate(i, "time")])
    assert model.design(i).shape == (len(model.fixed.coefficients()), 1)


def test_lmm_design_uses_declared_slope_name():
    b0 = sp.Symbol("b0")
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomEffects

    model = LMM(
        fixed=Fixed(intercept=b0, slopes={"visit_time": b1}),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="visit_time",
    )
    i, _ = model.indices()
    assert model.design(i) == sp.Matrix([1, model.covariate(i, "visit_time")])


def test_residual_coordinate_is_canonical_over_slope_coordinates():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomEffects, RandomSlope

    model = LMM(
        fixed=Fixed(intercept=b1),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Diagonal([tau2, tau2]),
            )
        ],
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
    )
    assert model.coordinate_name == "visit_time"


def test_single_declared_coordinate_is_inferred():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomEffects, RandomSlope

    model = LMM(
        fixed=Fixed(intercept=b1),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("visit_time"),),
                cov=Diagonal([tau2, tau2]),
            )
        ],
        residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
    )
    assert model.coordinate_name == "visit_time"


def test_explicit_coordinate_conflicts_with_residual_coordinate():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.errors import ModelError
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomEffects, RandomSlope

    with pytest.raises(ModelError, match="conflicts with residual coordinate"):
        LMM(
            fixed=Fixed(intercept=b1),
            random=[
                RandomEffects(
                    group="subject",
                    intercept=True,
                    slopes=(RandomSlope("time"),),
                    cov=Diagonal([tau2, tau2]),
                )
            ],
            residual=AR1(group="subject", coordinate="visit_time", var=sigma2, rho=rho),
            coordinate="time",
        )


def test_explicit_coordinate_may_differ_from_slope_coordinate():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.covariances import IID, Diagonal
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomEffects, RandomSlope

    model = LMM(
        fixed=Fixed(intercept=b1),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                slopes=(RandomSlope("time"),),
                cov=Diagonal([tau2, tau2]),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="visit_time",
    )
    assert model.coordinate_name == "visit_time"
