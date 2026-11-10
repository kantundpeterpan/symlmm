"""Canonical group-index and response-symbol tests."""

import pytest
import sympy as sp

from symlmm.canonical import (
    group_coefficients,
    group_mean,
    group_random_design,
    group_residual,
    group_size_symbol,
)
from symlmm.covariances import AR1, IID, Scalar
from symlmm.errors import NotDefined
from symlmm.model import LMM
from symlmm.specs import Covariate, Fixed, Group, RandomEffects


def _random_intercept_model(coordinate: str | None = "time") -> LMM:
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate=coordinate,
    )


def _two_factor_model() -> LMM:
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    omega2 = sp.Symbol("omega2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="subject", intercept=True, cov=Scalar(tau2)),
            RandomEffects(group="item", intercept=True, cov=Scalar(omega2)),
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )


def _slope_ar1_model() -> LMM:
    b0, b1 = sp.symbols("b0 b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0, slopes={"time": b1}),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=AR1(group="subject", coordinate="time", var=sigma2, rho=rho),
        coordinate="time",
    )


def _group_only_model() -> LMM:
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )


def _group_free_model() -> LMM:
    b0 = sp.Symbol("b0")
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(fixed=Fixed(intercept=b0), random=[], residual=IID(var=sigma2))


def test_group_size_symbolic_and_dense():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    assert model.group_size(g) == sp.Symbol("n_g", positive=True, integer=True)

    dense = model.dense(subject=[0, 0, 1], time=[0, 1, 0])
    assert dense.group_size(0) == 2
    assert dense.group_size(1) == 1


def test_response_uses_canonical_matrix_slices():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    t = sp.Symbol("t", integer=True)
    matrix = model.response_matrix()

    assert model.response(g) == matrix[g, :]
    assert model.response(g).shape == (1, model.occasion_count())
    assert model.time_response(t) == matrix[:, t]
    assert model.time_response(t).shape == (model.group_count(), 1)
    assert model.response(g, t) == matrix[g, t]
    assert not hasattr(model.response(g, t), "shape")


def test_design_and_covariance_blocks():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    p = len(model.fixed.coefficients())
    q = model.random[0].dim

    assert model.fixed_design(g).shape == (model.group_size(g), p)
    assert model.random_design(g).shape == (model.group_size(g), q)
    assert model.residual_covariance(g).shape == (
        model.group_size(g),
        model.group_size(g),
    )
    assert model.random_effect(g).shape == (q, 1)


def test_response_matrix_and_occasion_column():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    t = sp.Symbol("t", integer=True)

    assert model.response_matrix().shape == (model.group_count(), model.occasion_count())
    assert model.response(g).shape == (1, model.occasion_count())
    assert model.time_response(t).shape == (model.group_count(), 1)


def test_primary_group_and_canonical_indices():
    model = _random_intercept_model()

    assert model.primary_group == "subject"
    assert model.group_index == Group("subject").index
    assert model.occasion_index == Covariate("time").index


def test_group_and_occasion_counts():
    model = _random_intercept_model()
    assert model.group_count() == sp.Symbol("m", positive=True, integer=True)
    assert model.occasion_count() == sp.Symbol("T", positive=True, integer=True)

    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    assert dense.group_count() == 2
    assert dense.occasion_count() == 2


def test_dense_occasion_count_uses_distinct_occasion_values():
    dense = _random_intercept_model().dense(
        subject=[0, 0, 0, 1, 1, 1],
        time=[0, 1, 2, 2, 0, 1],
    )

    assert dense.occasion_count() == 3


def test_dense_occasion_count_rejects_duplicate_occasions():
    dense = _random_intercept_model().dense(
        subject=[0, 0, 1, 1],
        time=[0, 0, 0, 1],
    )

    with pytest.raises(NotDefined, match="balanced"):
        dense.occasion_count()


def test_dense_occasion_count_rejects_different_occasion_sets():
    dense = _random_intercept_model().dense(
        subject=[0, 0, 1, 1],
        time=[0, 1, 2, 3],
    )

    with pytest.raises(NotDefined, match="balanced"):
        dense.occasion_count()


def test_dense_occasion_count_requires_balanced_panel():
    dense = _random_intercept_model().dense(subject=[0, 0, 1], time=[0, 1, 0])

    with pytest.raises(NotDefined, match="balanced"):
        dense.occasion_count()


def test_dense_blocks_select_primary_group_rows():
    model = _random_intercept_model()
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    first = [0, 1]
    second = [2, 3]

    assert dense.fixed_design(0) == dense.matrix_x()[first, :]
    assert dense.random_design(0) == dense.matrix_z()[first, 0:1]
    assert dense.random_design(1) == dense.matrix_z()[second, 1:2]
    assert dense.residual_covariance(0) == dense.matrix_r()[first, :][:, first]
    assert dense.random_effect(0).shape == (1, 1)
    assert dense.response(0) == dense.response_matrix()[0, :]
    assert dense.response(0, 0) == dense.response_matrix()[0, 0]
    assert not hasattr(dense.response(0, 0), "shape")


def test_random_design_and_effect_use_primary_group_term_dimension():
    model = _two_factor_model()
    g = sp.Symbol("g", integer=True)

    assert model.random[0].dim == 1
    assert model.random_design(g).shape == (model.group_size(g), 1)
    assert model.random_effect(g).shape == (1, 1)

    dense = model.dense(
        subject=[0, 0, 1, 1],
        item=[0, 1, 0, 1],
        time=[0, 1, 0, 1],
    )
    first = [0, 1]
    second = [2, 3]

    assert dense.random_design(0).shape == (2, 1)
    assert dense.random_design(0) == dense.matrix_z()[first, 0:1]
    assert dense.random_design(1) == dense.matrix_z()[second, 1:2]


def test_group_queries_require_group_index():
    model = _group_free_model()
    g = sp.Symbol("g", integer=True)

    with pytest.raises(NotDefined):
        model.group_count()
    with pytest.raises(NotDefined):
        model.group_size(g)
    with pytest.raises(NotDefined):
        model.response(g)
    with pytest.raises(NotDefined):
        model.fixed_design(g)
    with pytest.raises(NotDefined):
        model.random_design(g)
    with pytest.raises(NotDefined):
        model.residual_covariance(g)
    with pytest.raises(NotDefined):
        model.random_effect(g)


def test_occasion_queries_require_occasion_index():
    model = _group_only_model()
    g = sp.Symbol("g", integer=True)
    t = sp.Symbol("t", integer=True)

    with pytest.raises(NotDefined):
        model.occasion_count()
    with pytest.raises(NotDefined):
        model.response_matrix()
    with pytest.raises(NotDefined):
        model.time_response(t)
    with pytest.raises(NotDefined):
        model.response(g)
    with pytest.raises(NotDefined):
        model.response(g, 0)


def test_canonical_helpers_build_named_symbols():
    g = sp.Symbol("g", integer=True)
    size = group_size_symbol(g)

    assert size == sp.Symbol("n_g", positive=True, integer=True)
    assert group_mean(g, size, 2) == sp.MatrixSymbol("X_g", size, 2)
    assert group_random_design(g, size, 2) == sp.MatrixSymbol("Z_g", size, 2)
    assert group_residual(g, size) == sp.MatrixSymbol("R_g", size, size)
    assert group_coefficients(g, 2) == sp.MatrixSymbol("b_g", 2, 1)


def test_group_mean_and_covariance_blocks():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    beta = sp.Matrix(model.fixed.coefficients())
    X = model.fixed_design(g)
    Z = model.random_design(g)
    G = model._primary_random_term().cov.dense()

    assert model.group_mean(g) == X * beta
    assert model.group_covariance(g) == Z * G * Z.T + model.residual_covariance(g)


def test_group_mean_with_no_fixed_effects_has_group_rows():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="time",
    )
    g = sp.Symbol("g", integer=True)

    assert model.group_mean(g).shape == (model.group_size(g), 1)

    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    assert dense.group_mean(0) == sp.zeros(2, 1)


def test_dense_moments_use_canonical_occasion_codes_for_unsorted_panels():
    model = _slope_ar1_model().dense(
        subject=[0, 0, 0, 1, 1, 1],
        time=[2, 0, 1, 2, 0, 1],
    )
    b0, b1 = model.fixed.coefficients()
    tau2 = model.random[0].cov.var
    sigma2 = model.residual.var
    rho = model.residual.rho

    assert model.entry_mean(0, 0) == b0
    assert model.entry_mean(0, 1) == b0 + b1
    assert model.entry_mean(0, 2) == b0 + 2 * b1
    assert model.group_mean(0)[:, 0] == sp.Matrix([b0, b0 + b1, b0 + 2 * b1])
    assert model.time_mean(0) == sp.Matrix([b0, b0])
    assert model.time_mean(2) == sp.Matrix([b0 + 2 * b1, b0 + 2 * b1])

    for group in range(2):
        for t in range(3):
            assert model.group_mean(group)[t, 0] == model.entry_mean(group, t)
            for s in range(3):
                assert model.group_covariance(group)[t, s] == model.entry_covariance(group, t, s)

    lag_two_covariance = tau2 + sigma2 * rho**2
    assert model.entry_covariance(0, 0, 2) == lag_two_covariance
    assert model.entry_cross_covariance(0, 0, 0, 2) == lag_two_covariance
    assert model.time_covariance(0, 2) == sp.diag(lag_two_covariance, lag_two_covariance)
    assert model.time_covariance(0, 2)[0, 1] == 0


def test_cross_group_covariance_is_zero():
    model = _random_intercept_model()
    g, h = sp.symbols("g h", integer=True)

    assert model.group_cross_covariance(g, h) == sp.ZeroMatrix(
        model.group_size(g), model.group_size(h)
    )
    assert model.group_cross_covariance(g, g) == model.group_covariance(g)


def test_entry_moments_are_group_components():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    mean = model.group_mean(g)
    cov = model.group_covariance(g)

    assert model.entry_mean(g, 0) == mean[0, 0]
    assert model.entry_covariance(g, 0, 1) == cov[0, 1]
    assert model.entry_cross_covariance(g, 0, g, 1) == cov[0, 1]


def test_occasion_column_moments_dense():
    model = _random_intercept_model().dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    t, s = 0, 1

    assert model.time_mean(t).shape == (2, 1)
    assert model.time_covariance(t, s).shape == (2, 2)
    assert model.time_covariance(t, s)[0, 0] == model.entry_covariance(0, t, s)
    assert model.time_covariance(t, s)[0, 1] == 0


def test_occasion_column_moments_require_concrete_group_count():
    model = _random_intercept_model()
    t = sp.Symbol("t", integer=True)

    with pytest.raises(NotDefined, match="concrete group count"):
        model.time_mean(t)
    with pytest.raises(NotDefined, match="concrete group count"):
        model.time_covariance(t, t)


def test_occasion_column_moments_require_balanced_dense_panel():
    model = _random_intercept_model().dense(subject=[0, 0, 1], time=[0, 1, 0])

    with pytest.raises(NotDefined, match="balanced"):
        model.time_mean(0)
    with pytest.raises(NotDefined, match="balanced"):
        model.time_covariance(0, 1)


def test_group_covariance_uses_primary_random_term_for_multiple_factors():
    model = _two_factor_model()
    g = sp.Symbol("g", integer=True)
    Z = model.random_design(g)
    G = model.random[0].cov.dense()

    assert model.group_covariance(g) == Z * G * Z.T + model.residual_covariance(g)


def test_group_cross_covariance_rejects_multiple_grouping_factors():
    model = _two_factor_model()
    g, h = sp.symbols("g h", integer=True)

    with pytest.raises(NotDefined, match="single grouping factor"):
        model.group_cross_covariance(g, h)
    with pytest.raises(NotDefined, match="single grouping factor"):
        model.group_cross_covariance(g, g)


def test_occasion_column_covariance_rejects_multiple_grouping_factors():
    model = _two_factor_model().dense(
        subject=[0, 0, 1, 1],
        item=[0, 1, 0, 1],
        time=[0, 1, 0, 1],
    )

    with pytest.raises(NotDefined, match="single grouping factor"):
        model.time_covariance(0, 1)


def test_occasion_column_mean_rejects_multiple_grouping_factors():
    model = _two_factor_model().dense(
        subject=[0, 0, 1, 1],
        item=[0, 1, 0, 1],
        time=[0, 1, 0, 1],
    )

    with pytest.raises(NotDefined, match="single grouping factor"):
        model.time_mean(0)
