import pytest
import sympy as sp

from symlmm.errors import ModelError
from symlmm.model import IndexedModel
from symlmm.specs import Crossed, Group, Nested


def test_observation_accessors():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), time="time")
    i, _ = model.indices()

    assert model.group(i, "subject") == sp.IndexedBase("subject", shape=(n,))[i]
    assert model.time(i) == sp.IndexedBase("time", shape=(n,))[i]
    assert model.covariate(i, "time") == sp.IndexedBase("time", shape=(n,))[i]
    assert model.design(i) == sp.Matrix([1, sp.IndexedBase("time", shape=(n,))[i]])


def test_time_symbols_are_canonical():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), time="time")
    i, _ = model.indices()

    assert model.time(i) == model.time(i, "time")


def test_design_includes_time_symbol():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),), time="time")
    i, _ = model.indices()

    assert model.design(i)[1] == model.time(i)


def test_unknown_grouping_factor_raises_model_error():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),))
    i, _ = model.indices()

    with pytest.raises(ModelError):
        model.group(i, "missing")


def test_missing_time_coordinate_falls_back_to_t():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Group("subject"),))
    i, _ = model.indices()

    assert model.time(i) == sp.IndexedBase("t", shape=(n,))[i]


def test_nested_label_is_composite():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Nested("visit", within="subject"),))
    i, _ = model.indices()
    label = model.group(i, "visit")
    assert label.base == sp.IndexedBase("subject:visit", shape=(n,))


def test_crossed_label_is_composite():
    n = sp.Symbol("n", positive=True, integer=True)
    model = IndexedModel(n=n, groups=(Crossed("subject", "item"),))
    i, _ = model.indices()
    label = model.group(i, "item")
    assert label.base == sp.IndexedBase("subject:item", shape=(n,))


def test_lmm_design_uses_fixed_columns_without_intercept():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    from symlmm.covariances import IID, Scalar
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomIntercept

    model = LMM(
        fixed=Fixed(intercept=None, slopes={"time": b1}),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
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
    from symlmm.specs import Fixed, RandomIntercept

    model = LMM(
        fixed=Fixed(intercept=b0, slopes={"visit_time": b1}),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        time="visit_time",
    )
    i, _ = model.indices()
    assert model.design(i) == sp.Matrix([1, model.covariate(i, "visit_time")])


def test_conflicting_term_time_coordinates_raise():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.errors import ModelError
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomSlope

    with pytest.raises(ModelError, match="conflicting time coordinates"):
        LMM(
            fixed=Fixed(intercept=b1),
            random=[RandomSlope(group="subject", time="time", cov=Diagonal([tau2, tau2]))],
            residual=AR1(group="subject", time="visit_time", var=sigma2, rho=rho),
        )


def test_single_declared_time_coordinate_is_inferred():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomSlope

    model = LMM(
        fixed=Fixed(intercept=b1),
        random=[RandomSlope(group="subject", time="visit_time", cov=Diagonal([tau2, tau2]))],
        residual=AR1(group="subject", time="visit_time", var=sigma2, rho=rho),
    )
    assert model.time_name == "visit_time"


def test_explicit_time_coordinate_validates_terms():
    b1 = sp.Symbol("b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    from symlmm.covariances import AR1, Diagonal
    from symlmm.errors import ModelError
    from symlmm.model import LMM
    from symlmm.specs import Fixed, RandomSlope

    with pytest.raises(ModelError, match="conflicts with term coordinates"):
        LMM(
            fixed=Fixed(intercept=b1),
            random=[RandomSlope(group="subject", time="time", cov=Diagonal([tau2, tau2]))],
            residual=AR1(group="subject", time="visit_time", var=sigma2, rho=rho),
            time="visit_time",
        )
