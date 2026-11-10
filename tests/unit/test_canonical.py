"""Canonical group-index and response-symbol tests."""

import pytest
import sympy as sp

from symlmm.canonical import (
    group_coefficients,
    group_mean,
    group_random_design,
    group_residual,
    group_response,
    group_size_symbol,
)
from symlmm.covariances import IID, Scalar
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


def test_response_is_group_vector_and_entry_is_scalar():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    assert model.response(g).shape == (model.group_size(g), 1)
    assert model.response(g, 0).shape == (1, 1)
    assert model.response(g, 0) == model.response(g)[0, 0]


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
    assert model.response(g).shape == (model.group_size(g), 1)
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


def test_dense_occasion_count_requires_balanced_panel():
    dense = _random_intercept_model().dense(subject=[0, 0, 1], time=[0, 1, 2])

    with pytest.raises(NotDefined, match="balanced"):
        dense.occasion_count()


def test_dense_blocks_select_primary_group_rows():
    model = _random_intercept_model()
    dense = model.dense(subject=[0, 0, 1], time=[0, 1, 0])
    rows = [0, 1]

    assert dense.fixed_design(0) == dense.matrix_x()[rows, :]
    assert dense.random_design(0) == dense.matrix_z()[rows, :]
    assert dense.residual_covariance(0) == dense.matrix_r()[rows, :][:, rows]
    assert dense.random_effect(0).shape == (1, 1)
    assert dense.response(0).shape == (2, 1)
    assert dense.response(0, 0) == dense.response(0)[0, 0]


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
        model.response(g, 0)

    assert model.response(g).shape == (model.group_size(g), 1)


def test_canonical_helpers_build_named_symbols():
    g = sp.Symbol("g", integer=True)
    size = group_size_symbol(g)

    assert size == sp.Symbol("n_g", positive=True, integer=True)
    assert group_response(g, size) == sp.MatrixSymbol("Y_g", size, 1)
    assert group_mean(g, size, 2) == sp.MatrixSymbol("X_g", size, 2)
    assert group_random_design(g, size, 2) == sp.MatrixSymbol("Z_g", size, 2)
    assert group_residual(g, size) == sp.MatrixSymbol("R_g", size, size)
    assert group_coefficients(g, 2) == sp.MatrixSymbol("b_g", 2, 1)
