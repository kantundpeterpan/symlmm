"""Blackboard-style symbolic moment tests on canonical group indexing."""

import pytest
import sympy as sp

from symlmm.blackboard import GroupEffectSymbols, ModelSymbols, ResponseSymbols
from symlmm.covariances import IID, Scalar, Unstructured
from symlmm.errors import ModelError, NotDefined, UnsupportedMoment
from symlmm.model import LMM
from symlmm.specs import Covariate, Fixed, Group, RandomEffects, RandomSlope


def _random_intercept_model() -> LMM:
    b0, b1 = sp.symbols("b0 b1")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    g = sp.Symbol("g", integer=True)
    t = sp.Symbol("t", integer=True)
    return LMM(
        fixed=Fixed(intercept=b0, slopes={Covariate("time", index=t): b1}),
        random=[
            RandomEffects(
                group=Group("subject", index=g),
                intercept=True,
                cov=Scalar(tau2),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )


def _random_slope_model() -> LMM:
    b0, b1 = sp.symbols("b0 b1")
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    g = sp.Symbol("g", integer=True)
    t = sp.Symbol("t", integer=True)
    time = Covariate("time", index=t)
    return LMM(
        fixed=Fixed(intercept=b0, slopes={time: b1}),
        random=[
            RandomEffects(
                group=Group("subject", index=g),
                intercept=True,
                slopes=(RandomSlope(time),),
                cov=Unstructured.from_sd_corr([s0, s1], rho),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )


def _two_factor_model() -> LMM:
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    omega2 = sp.Symbol("omega2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group=Group("subject"), intercept=True, cov=Scalar(tau2)),
            RandomEffects(group=Group("item"), intercept=True, cov=Scalar(omega2)),
        ],
        residual=IID(var=sigma2),
    )


def test_model_symbols_exposes_canonical_indices_and_accessors():
    model = _random_intercept_model()
    syms = model.symbols()
    g = syms.i
    p = len(model.fixed.coefficients())
    q = model.random[0].dim

    assert isinstance(syms, ModelSymbols)
    assert isinstance(syms.Y, ResponseSymbols)
    assert isinstance(syms.b, GroupEffectSymbols)
    assert syms.i == model.group_index
    assert syms.t == model.occasion_index
    assert syms.beta.shape == (p, 1)
    assert syms.X(g) == model.fixed_design(g)
    assert syms.Z(g) == model.random_design(g)
    assert syms.R(g) == model.residual_covariance(g)
    assert syms.G == model.random[0].cov.dense()
    assert syms.b[g] == model.random_effect(g)
    assert syms.b[g].shape == (q, 1)


def test_response_symbols_dispatch_on_canonical_indices():
    model = _random_intercept_model()
    syms = model.symbols()
    g, t = syms.i, syms.t
    s = sp.Symbol("s", integer=True)

    assert syms.Y[g] == model.response(g)
    assert syms.Y[g].shape == (1, model.occasion_count())
    assert syms.Y[:, t] == model.time_response(t)
    assert syms.Y[:, t].shape == (model.group_count(), 1)
    assert syms.Y[t] == model.time_response(t)
    assert syms.Y[syms.t] == model.time_response(syms.t)
    assert syms.Y[g, 0] == syms.Y[g][0, 0]
    assert syms.Y[g, t] == model.response(g, t)
    assert syms.Y[g, s] == model.response(g, s)
    assert syms.Y[0, :] == model.response(0)
    assert syms.Y[:, 0] == model.time_response(0)


def test_response_symbols_reject_ambiguous_integer_keys():
    model = _random_intercept_model()
    syms = model.symbols()
    h = sp.Symbol("h", integer=True)

    with pytest.raises(ModelError, match="ambiguous"):
        syms.Y[0]
    with pytest.raises(ModelError, match="ambiguous"):
        syms.Y[h]


def test_response_symbols_use_model_indices_without_name_fallback():
    model = _random_intercept_model()
    syms = model.symbols()

    assert syms.i is not sp.Symbol("i", integer=True)
    with pytest.raises(ModelError, match="ambiguous"):
        syms.Y[sp.Symbol("i", integer=True)]


def test_group_response_moments_delegate_to_canonical_blocks():
    model = _random_intercept_model()
    syms = model.symbols()
    g = syms.i
    h = sp.Symbol("h", integer=True)

    assert syms.E(syms.Y[g]) == model.group_mean(g).T
    assert syms.Var(syms.Y[g]) == model.group_covariance(g)
    assert syms.Cov(syms.Y[g], syms.Y[g]) == model.group_covariance(g)
    assert syms.Cov(syms.Y[g], syms.Y[h, :]) == model.group_cross_covariance(g, h)


def test_entry_moments_delegate_to_canonical_entries():
    model = _random_intercept_model()
    syms = model.symbols()
    g = syms.i
    h = sp.Symbol("h", integer=True)
    t, s = 0, 1

    assert syms.E(syms.Y[g, t]) == model.entry_mean(g, t)
    assert syms.Var(syms.Y[g, t]) == model.entry_covariance(g, t, t)
    assert syms.Cov(syms.Y[g, t], syms.Y[g, s]) == model.entry_covariance(g, t, s)
    assert syms.Cov(syms.Y[g, t], syms.Y[h, s]) == model.entry_cross_covariance(g, t, h, s)


def test_scalar_linear_expressions_are_supported():
    model = _random_intercept_model()
    syms = model.symbols()
    g = syms.i
    t, s = 0, 1
    expr = 2 * syms.Y[g, t] + 3 * syms.Y[g, s]

    assert syms.E(expr) == 2 * model.entry_mean(g, t) + 3 * model.entry_mean(g, s)
    assert syms.Var(expr) == (
        4 * model.entry_covariance(g, t, t)
        + 9 * model.entry_covariance(g, s, s)
        + 6 * model.entry_covariance(g, t, s)
        + 6 * model.entry_covariance(g, s, t)
    )


def test_occasion_column_moments_use_dense_canonical_columns():
    model = _random_intercept_model().dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    syms = model.symbols()
    t, s = 0, 1

    assert syms.Y[:, t].shape == (model.group_count(), 1)
    assert syms.E(syms.Y[:, t]) == model.time_mean(t)
    assert syms.E(syms.Y[:, t]).shape == (model.group_count(), 1)
    assert syms.Var(syms.Y[:, t]) == model.time_covariance(t, t)
    assert syms.Var(syms.Y[:, t]).shape == (model.group_count(), model.group_count())
    assert syms.Cov(syms.Y[:, t], syms.Y[:, s]) == model.time_covariance(t, s)


def test_dense_group_response_requires_concrete_group_labels():
    model = _random_intercept_model().dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    syms = model.symbols()

    assert syms.Y[0, :] == model.response(0)
    with pytest.raises(NotDefined, match="concrete group label"):
        syms.Y[syms.i]
    with pytest.raises(NotDefined, match="concrete group label"):
        syms.Y[syms.i, :]
    with pytest.raises(NotDefined, match="concrete group label"):
        syms.Y[syms.i, 0]


def test_random_effect_moments_use_group_covariance():
    model = _random_slope_model()
    syms = model.symbols()
    g = syms.i
    h = sp.Symbol("h", integer=True)
    q = model.random[0].dim
    G = model.random[0].cov.dense()

    assert syms.E(syms.b[g]) == sp.zeros(q, 1)
    assert syms.Var(syms.b[g]) == G
    assert syms.Cov(syms.b[g], syms.b[g]) == G
    assert syms.Cov(syms.b[g], syms.b[h]) == sp.zeros(q, q)
    assert syms.Cov(syms.b[g][0, 0], syms.b[g][1, 0]) == G[0, 1]


def test_unknown_random_effect_atoms_are_rejected():
    model = _random_intercept_model()
    syms = model.symbols()
    h = sp.Symbol("h", integer=True)
    unknown = sp.MatrixSymbol("b_unknown", 1, 1)

    with pytest.raises(ModelError, match="registered"):
        syms.Var(unknown)
    with pytest.raises(ModelError, match="registered"):
        syms.Var(model.random_effect(h))


def test_multi_factor_effect_symbols_reject_non_primary_groups():
    model = _two_factor_model()
    syms = model.symbols()
    secondary = sp.Symbol("item", integer=True)

    assert syms.b[syms.i] == model.random_effect(syms.i)
    with pytest.raises(NotDefined, match="primary"):
        syms.b[secondary]
    with pytest.raises(NotDefined, match="primary"):
        syms.b[Group("item")]


def test_response_random_effect_cross_covariance():
    model = _random_intercept_model()
    syms = model.symbols()
    g = syms.i
    h = sp.Symbol("h", integer=True)

    assert syms.Cov(syms.Y[g], syms.b[g]) == model.random_design(g) * syms.G
    assert syms.Cov(syms.Y[g], syms.b[h]) == sp.ZeroMatrix(model.group_size(g), model.random[0].dim)
    assert syms.Cov(syms.Y[g, 0], syms.b[g]) == (model.random_design(g) * syms.G)[0, :]


def test_conditional_group_response_moments():
    model = _random_intercept_model()
    syms = model.symbols()
    g = syms.i
    b = syms.b[g]
    conditional_mean = model.group_mean(g) + model.random_design(g) * b

    assert syms.E(syms.Y[g], given=b) == conditional_mean.T
    assert syms.Var(syms.Y[g], given=b) == model.residual_covariance(g)
    assert syms.Cov(syms.Y[g], syms.Y[g], given=b) == model.residual_covariance(g)
    assert syms.E(syms.Y[g, 0], given=b) == conditional_mean[0, 0]
    assert syms.Var(syms.Y[g, 0], given=b) == model.residual_covariance(g)[0, 0]
    assert syms.Cov(syms.Y[g, 0], syms.Y[g, 1], given=b) == model.residual_covariance(g)[0, 1]
    assert syms.E(b, given=b) == b
    assert syms.Var(b, given=b) == sp.zeros(model.random[0].dim, model.random[0].dim)


def test_conditional_mean_variance_and_law_of_total_variance():
    model = _random_intercept_model()
    syms = model.symbols()
    g = syms.i
    b = syms.b[g]
    Z = model.random_design(g)
    G = model.random[0].cov.dense()
    R = model.residual_covariance(g)

    conditional_mean = syms.E(syms.Y[g], given=b)
    assert syms.Var(conditional_mean) == Z * G * Z.T
    assert syms.E(syms.Var(syms.Y[g], given=b)) == R
    assert syms.total_variance(syms.Y[g], given=b) == model.group_covariance(g)


def test_total_variance_for_scalar_entries_and_random_effects():
    model = _random_intercept_model()
    syms = model.symbols()
    g = syms.i
    b = syms.b[g]
    G = model.random[0].cov.dense()

    assert syms.total_variance(syms.Y[g, 0], given=b) == model.entry_covariance(g, 0, 0)
    assert syms.total_variance(syms.b[g], given=b) == G
    assert syms.total_variance(syms.Y[g]) == model.group_covariance(g)


def test_nonlinear_and_vector_combinations_are_rejected():
    model = _random_intercept_model()
    syms = model.symbols()
    g = syms.i
    h = sp.Symbol("h", integer=True)
    coefficient = sp.MatrixSymbol("A", 1, 2)

    with pytest.raises(UnsupportedMoment, match="linear"):
        syms.Var(syms.Y[g, 0] ** 2)
    with pytest.raises(UnsupportedMoment, match="linear|vector"):
        syms.Var(syms.Y[g, 0] * syms.Y[g, 1])
    with pytest.raises(UnsupportedMoment, match="vector"):
        syms.Var(syms.Y[g] + syms.Y[h, :])
    with pytest.raises(UnsupportedMoment, match="matrix coefficients on scalar response atoms"):
        syms.E(coefficient * syms.Y[g, 0])
    with pytest.raises(UnsupportedMoment, match="given"):
        syms.E(syms.Y[g], given=sp.Symbol("x"))


def test_random_effect_and_parameter_moments_need_no_occasion_index():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    g = sp.Symbol("g", integer=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group=Group("subject", index=g), intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    syms = model.symbols()

    assert syms.t is None
    assert syms.Var(syms.b[g]) == model.random[0].cov.dense()
    assert syms.Var(syms.beta) == sp.zeros(1, 1)
    assert syms.E(syms.beta) == syms.beta
    with pytest.raises(NotDefined):
        syms.Y[g]


def test_blackboard_queries_do_not_fabricate_indices():
    model = _random_intercept_model()
    syms = model.symbols()

    with pytest.raises(ModelError, match="ambiguous"):
        syms.Y[sp.Symbol("subject", integer=True)]
    with pytest.raises(ModelError, match="ambiguous"):
        syms.Y[sp.Symbol("time", integer=True)]
