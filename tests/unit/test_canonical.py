"""Factor-first block and panel regression tests."""

import pytest
import sympy as sp

from symlmm.blocks import (
    block_coefficients,
    block_mean_symbol,
    block_random_design,
    block_residual,
    block_size_symbol,
)
from symlmm.covariances import AR1, IID, Scalar
from symlmm.errors import NotDefined
from symlmm.model import LMM
from symlmm.specs import Covariate, Fixed, RandomEffects


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


def test_block_size_symbolic_and_dense():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    assert model.block("subject", g).size() == sp.Symbol("n_subject:g", positive=True, integer=True)

    dense = model.dense(subject=[0, 0, 1], time=[0, 1, 0])
    assert dense.block("subject", 0).size() == 2
    assert dense.block("subject", 1).size() == 1


def test_response_uses_panel_matrix_slices():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    t = sp.Symbol("t", integer=True)
    panel = model.panel()
    matrix = panel.response_matrix()

    assert panel.response(g) == matrix[g, :]
    assert panel.response(g).shape == (1, panel.position_count())
    assert panel.response_column(t) == matrix[:, t]
    assert panel.response_column(t).shape == (panel.level_count(), 1)
    assert not hasattr(matrix[g, t], "shape")


def test_design_and_covariance_blocks():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    p = len(model.fixed.coefficients())
    q = model.random[0].dim
    block = model.block("subject", g)

    assert block.design().shape == (block.size(), p)
    assert block.random_design(term="subject").shape == (block.size(), q)
    assert block.residual().shape == (block.size(), block.size())
    assert block.random_effect().shape == (q, 1)


def test_response_matrix_and_position_column():
    model = _random_intercept_model()
    g = sp.Symbol("g", integer=True)
    t = sp.Symbol("t", integer=True)
    panel = model.panel()

    assert panel.response_matrix().shape == (panel.level_count(), panel.position_count())
    assert panel.response(g).shape == (1, panel.position_count())
    assert panel.response_column(t).shape == (panel.level_count(), 1)


def test_primary_group_and_canonical_indices():
    model = _random_intercept_model()

    assert model.primary_group == "subject"
    assert model.group_index == sp.Symbol("subject", integer=True)
    assert model.occasion_index == Covariate("time").index


def test_level_and_position_counts():
    model = _random_intercept_model()
    panel = model.panel()
    assert panel.level_count() == sp.Symbol("m_subject", positive=True, integer=True)
    assert panel.position_count() == sp.Symbol("T", positive=True, integer=True)

    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    dense_panel = dense.panel()
    assert dense_panel.level_count() == 2
    assert dense_panel.position_count() == 2


def test_dense_position_count_uses_distinct_position_values():
    dense = _random_intercept_model().dense(
        subject=[0, 0, 0, 1, 1, 1],
        time=[0, 1, 2, 2, 0, 1],
    )

    assert dense.panel().position_count() == 3


def test_dense_position_count_rejects_duplicate_positions():
    dense = _random_intercept_model().dense(
        subject=[0, 0, 1, 1],
        time=[0, 0, 0, 1],
    )

    with pytest.raises(NotDefined, match="balanced"):
        dense.panel().position_count()


def test_dense_position_count_uses_position_codes_for_different_raw_values():
    dense = _random_intercept_model().dense(
        subject=[0, 0, 1, 1],
        time=[0, 1, 2, 3],
    )
    panel = dense.panel()

    assert panel.grid.is_balanced()
    assert panel.position_count() == 2
    assert panel.response_matrix().shape == (2, 2)


def test_dense_position_count_requires_balanced_panel():
    dense = _random_intercept_model().dense(subject=[0, 0, 1], time=[0, 1, 0])

    with pytest.raises(NotDefined, match="balanced"):
        dense.panel().position_count()


def test_dense_blocks_select_primary_group_rows():
    model = _random_intercept_model()
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    first = [0, 1]
    second = [2, 3]
    block0 = dense.block("subject", 0)
    block1 = dense.block("subject", 1)

    assert block0.design() == dense.matrix_x()[first, :]
    assert block0.random_design(term="subject") == dense.matrix_z()[first, 0:1]
    assert block1.random_design() == dense.matrix_z()[second, :]
    assert block0.residual() == dense.matrix_r()[first, :][:, first]
    assert block0.random_effect().shape == (1, 1)
    panel = dense.panel("subject", "time")
    assert panel.response(0) == panel.response_matrix()[0, :]
    assert not hasattr(panel.response_matrix()[0, 0], "shape")


def test_random_design_and_effect_use_term_dimension():
    model = _two_factor_model()
    g = sp.Symbol("g", integer=True)

    assert model.random[0].dim == 1
    block = model.block("subject", g)
    assert block.random_design(term="subject").shape == (block.size(), 1)
    assert block.random_effect().shape == (1, 1)

    dense = model.dense(
        subject=[0, 0, 1, 1],
        item=[0, 1, 0, 1],
        time=[0, 1, 0, 1],
    )
    first = [0, 1]
    second = [2, 3]

    assert dense.block("subject", 0).random_design(term="subject").shape == (2, 1)
    assert dense.block("subject", 0).random_design(term="subject") == dense.matrix_z()[first, 0:1]
    assert dense.block("subject", 1).random_design() == dense.matrix_z()[second, :]


def test_block_queries_require_group_index():
    model = _group_free_model()

    assert model.primary_group is None
    with pytest.raises(NotDefined):
        model.panel()


def test_position_queries_require_position_axis():
    model = _group_only_model()
    panel = model.panel()
    assert panel.position is None

    calls = (
        panel.position_count,
        panel.response_matrix,
        lambda: panel.response_column(0),
        lambda: panel.position_code(0),
        lambda: panel.entry_mean(0, 0),
        lambda: panel.entry_covariance(0, 0, 1),
        lambda: panel.entry_cross_covariance(0, 0, 1, 0),
        lambda: panel.position_mean(0),
        lambda: panel.position_covariance(0, 1),
    )
    for call in calls:
        with pytest.raises(NotDefined, match="position axis"):
            call()


def test_block_helpers_build_named_symbols():
    model = _random_intercept_model()
    factor = model.factor("subject")
    g = sp.Symbol("g", integer=True)
    size = block_size_symbol(factor, g)

    assert size == sp.Symbol("n_subject:g", positive=True, integer=True)
    assert block_mean_symbol(factor, g, 2) == sp.MatrixSymbol("X_subject:g", size, 2)
    assert block_random_design(factor, g, 2) == sp.MatrixSymbol("Z_subject:g", size, 2)
    assert block_residual(factor, g) == sp.MatrixSymbol("R_subject:g", size, size)
    assert block_coefficients(factor, g, 2) == sp.MatrixSymbol("b_subject:g", 2, 1)


def test_block_mean_and_covariance_terms():
    model = _random_intercept_model()
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    block = dense.block("subject", 0)
    beta = sp.Matrix(model.fixed.coefficients())
    Z = block.random_design(term="subject")
    G = model.random[0].cov.dense()
    R = block.residual()

    assert block.mean() == block.design() * beta
    assert block.covariance() == Z * G * Z.T + R
    values = block.covariance_terms()
    assert sum(values.values(), sp.zeros(block.size(), block.size())) == block.covariance()


def test_block_mean_with_no_fixed_effects_has_group_rows():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="time",
    )
    g = sp.Symbol("g", integer=True)
    block = model.block("subject", g)

    assert block.mean().shape == (block.size(), 1)

    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 1])
    assert dense.block("subject", 0).mean() == sp.zeros(2, 1)


def test_panel_entry_moments_track_positions_for_unsorted_panels():
    model = _slope_ar1_model().dense(
        subject=[0, 0, 0, 1, 1, 1],
        time=[2, 0, 1, 2, 0, 1],
    )
    b0, b1 = model.fixed.coefficients()
    tau2 = model.random[0].cov.var
    sigma2 = model.residual.var
    rho = model.residual.rho
    panel = model.panel("subject", "time")

    assert [panel.entry_mean(0, p) for p in range(3)] == [b0, b0 + b1, b0 + 2 * b1]
    assert panel.position_mean(0) == sp.Matrix([b0, b0])
    assert panel.position_mean(2) == sp.Matrix([b0 + 2 * b1, b0 + 2 * b1])

    for level in range(2):
        block = model.block("subject", level)
        for p in range(3):
            row = panel.grid.row(level, p)
            assert panel.entry_mean(level, p) == block.mean()[row, 0]
            for q in range(3):
                column = panel.grid.row(level, q)
                assert panel.entry_covariance(level, p, q) == block.covariance()[row, column]

    lag_two_covariance = tau2 + sigma2 * rho**2
    assert panel.entry_covariance(0, 0, 2) == lag_two_covariance
    assert panel.entry_cross_covariance(0, 0, 0, 2) == lag_two_covariance
    assert panel.position_covariance(0, 2) == sp.diag(lag_two_covariance, lag_two_covariance)
    assert panel.position_covariance(0, 2)[0, 1] == 0


def test_block_cross_covariance_is_zero_for_distinct_levels():
    dense = _random_intercept_model().dense(
        subject=[0, 0, 1, 1],
        time=[0, 1, 0, 1],
    )
    left = dense.block("subject", 0)
    right = dense.block("subject", 1)

    assert left.cross_covariance(right) == sp.zeros(2, 2)
    assert left.cross_covariance(left) == left.covariance()


def test_entry_moments_are_block_components():
    dense = _random_intercept_model().dense(
        subject=[0, 0, 1, 1],
        time=[0, 1, 0, 1],
    )
    panel = dense.panel("subject", "time")
    block = dense.block("subject", 0)

    assert panel.entry_mean(0, 0) == block.mean()[0, 0]
    assert panel.entry_covariance(0, 0, 1) == block.covariance()[0, 1]
    assert panel.entry_cross_covariance(0, 0, 0, 1) == block.covariance()[0, 1]


def test_panel_position_moments_dense():
    dense = _random_intercept_model().dense(
        subject=[0, 0, 1, 1],
        time=[0, 1, 0, 1],
    )
    panel = dense.panel("subject", "time")
    t, s = 0, 1

    assert panel.position_mean(t).shape == (2, 1)
    assert panel.position_covariance(t, s).shape == (2, 2)
    assert panel.position_covariance(t, s)[0, 0] == panel.entry_covariance(0, t, s)
    assert panel.position_covariance(t, s)[0, 1] == 0


def test_panel_position_moments_require_concrete_level_count():
    panel = _random_intercept_model().panel()
    t = sp.Symbol("t", integer=True)

    with pytest.raises(NotDefined, match="concrete level count"):
        panel.position_mean(t)
    with pytest.raises(NotDefined, match="concrete level count"):
        panel.position_covariance(t, t)


def test_panel_position_count_requires_balanced_dense_panel():
    panel = (
        _random_intercept_model()
        .dense(
            subject=[0, 0, 1],
            time=[0, 1, 0],
        )
        .panel("subject", "time")
    )

    with pytest.raises(NotDefined, match="balanced"):
        panel.position_count()
    with pytest.raises(NotDefined, match="balanced"):
        panel.response_matrix()


def test_block_covariance_supports_multiple_grouping_factors():
    model = _two_factor_model()
    g = sp.Symbol("g", integer=True)
    block = model.block("subject", g)

    assert list(block.covariance_terms()) == ["subject", "item", "residual"]
    assert block.covariance().shape == (block.size(), block.size())


def test_dense_block_covariance_supports_multiple_grouping_factors():
    dense = _two_factor_model().dense(
        subject=[0, 0, 1, 1],
        item=[0, 1, 0, 1],
        time=[0, 1, 0, 1],
    )
    block = dense.block("subject", 0)

    assert block.covariance() == dense.matrix_covariance()[block.rows(), :][:, block.rows()]
    panel = dense.panel("subject", "time")
    assert panel.entry_covariance(0, 0, 1) == block.covariance()[0, 1]


def test_block_cross_covariance_supports_multiple_grouping_factors():
    dense = _two_factor_model().dense(
        subject=[0, 0, 1, 1],
        item=[0, 1, 0, 1],
        time=[0, 1, 0, 1],
    )
    left = dense.block("subject", 0)
    right = dense.block("subject", 1)

    assert (
        left.cross_covariance(right) == dense.matrix_covariance()[left.rows(), :][:, right.rows()]
    )
    assert left.cross_covariance(right) != sp.zeros(2, 2)


def test_panel_position_covariance_requires_single_grouping_factor():
    dense = _two_factor_model().dense(
        subject=[0, 0, 1, 1],
        item=[0, 1, 0, 1],
        time=[0, 1, 0, 1],
    )
    panel = dense.panel("subject", "time")

    with pytest.raises(NotDefined, match="single grouping factor"):
        panel.position_covariance(0, 1)


def test_panel_position_mean_supports_multiple_grouping_factors():
    b0 = sp.Symbol("b0")
    dense = _two_factor_model().dense(
        subject=[0, 0, 1, 1],
        item=[0, 1, 0, 1],
        time=[0, 1, 0, 1],
    )
    panel = dense.panel("subject", "time")

    assert panel.position_mean(0) == sp.Matrix([b0, b0])
    assert panel.position_mean(0).shape == (2, 1)
