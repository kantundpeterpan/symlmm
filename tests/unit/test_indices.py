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


def test_lmm_group_and_occasion_indices_from_definitions():
    b0, b1 = sp.symbols("b0 b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    i = sp.Symbol("i", integer=True)
    t = sp.Symbol("t", integer=True)
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Covariate, Fixed, Group, RandomEffects

    time = Covariate("time", index=t)
    model = LMM(
        fixed=Fixed(intercept=b0, slopes={time: b1}),
        random=[RandomEffects(group=Group("subject", index=i), intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="time",
    )

    assert model.group_index is i
    assert model.occasion_index is t
    assert model.occasion_definition == time


def test_lmm_default_indices_follow_definition_names():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Covariate, Fixed, Group, RandomEffects

    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="time",
    )

    assert model.group_index == Group("subject").index
    assert model.occasion_index == Covariate("time").index
    assert model.group_index != model.occasion_index


def test_lmm_without_coordinate_has_no_occasion_index():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Fixed, Group, RandomEffects

    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )

    assert model.group_index == Group("subject").index
    assert model.occasion_index is None
    assert model.occasion_definition is None


def test_lmm_without_group_does_not_fabricate_group_index():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.covariances import IID
    from symlmm.model import LMM
    from symlmm.specs import Fixed

    model = LMM(fixed=Fixed(intercept=b0), random=[], residual=IID(var=sigma2))

    assert model.group_index is None
    assert model.primary_group_definition is None


def test_lmm_uses_registered_definition_for_residual_group():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    i = sp.Symbol("i", integer=True)
    from symlmm.covariances import AR1
    from symlmm.model import LMM
    from symlmm.specs import Fixed, Group

    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(group="subject", coordinate="time", var=sigma2, rho=rho),
        groups=[Group("subject", index=i)],
    )

    assert model.group_index is i
    assert model.primary_group_definition == Group("subject", index=i)


def test_lmm_residual_coordinate_supplies_occasion_index():
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    t = sp.Symbol("t", integer=True)
    from symlmm.covariances import AR1
    from symlmm.model import LMM
    from symlmm.specs import Covariate, Fixed, Group

    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[],
        residual=AR1(
            group=Group("subject"),
            coordinate=Covariate("visit_time", index=t),
            var=sigma2,
            rho=rho,
        ),
    )

    assert model.group_index == Group("subject").index
    assert model.occasion_index is t


def test_lmm_equal_group_and_occasion_indices_raise():
    b0, b1 = sp.symbols("b0 b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    t = sp.Symbol("t", integer=True)
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Covariate, Fixed, Group, RandomEffects

    with pytest.raises(ModelError, match="distinct"):
        LMM(
            fixed=Fixed(intercept=b0, slopes={Covariate("time", index=t): b1}),
            random=[
                RandomEffects(group=Group("subject", index=t), intercept=True, cov=Scalar(tau2))
            ],
            residual=IID(var=sigma2),
            coordinate="time",
        )
