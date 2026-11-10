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


def test_conflicting_term_coordinates_raise():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.errors import ModelError
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomEffects, RandomSlope

    with pytest.raises(ModelError, match="conflicting coordinates"):
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
        )


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


def test_explicit_coordinate_validates_terms():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.errors import ModelError
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomEffects, RandomSlope

    with pytest.raises(ModelError, match="conflicts with term coordinates"):
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
            coordinate="visit_time",
        )
