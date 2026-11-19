import pytest
import sympy as sp

from symlmm.covariances import IID, Diagonal, Scalar, Unstructured
from symlmm.errors import ModelError, NotDefined
from symlmm.model import LMM
from symlmm.panel import PanelView
from symlmm.specs import (
    Covariate,
    Fixed,
    Group,
    Position,
    RandomEffects,
    RandomSlope,
)


def _model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2))],
        residual=IID(var=s2),
        coordinate="p",
        n=4,
    )


def test_panel_uses_position_codes_not_raw_values():
    dense = _model().dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    assert panel.position_count() == 2
    assert panel.response_matrix().shape == (2, 2)


def test_panel_explicit_position_column():
    dense = _model().dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21], visit=["a", "b", "c", "d"])
    panel = PanelView(dense, dense.factor("F"), Position("p", column="visit"))
    assert panel.position_count() == 2


def test_panel_rejects_unbalanced_blocks():
    dense = _model().dense(F=[0, 0, 1], p=[10, 11, 20])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    with pytest.raises(NotDefined, match="balanced"):
        panel.position_count()


def test_panel_rejects_free_symbolic_derived_positions():
    t1 = sp.Symbol("t1", positive=True)
    dense = _model().dense(F=[0, 0, 1, 1], p=[0, 1, t1, 2])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    with pytest.raises(ModelError, match="free symbols"):
        panel.position_count()


def test_panel_entry_moments():
    dense = _model().dense(F=[0, 0, 1, 1], p=[10, 11, 20, 21])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    assert panel.entry_mean(0, 0) == dense.matrix_mean()[0, 0]
    assert panel.entry_covariance(0, 0, 1) == dense.matrix_covariance()[0, 1]


def test_panel_entry_mean_uses_position_order():
    b0 = sp.Symbol("b0")
    b1 = sp.Symbol("b1")
    t2 = sp.Symbol("t2", positive=True)
    t3 = sp.Symbol("t3", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0, slopes={"p": b1}),
        random=[
            RandomEffects(
                group="F",
                intercept=True,
                slopes=(RandomSlope("p", within=False),),
                cov=Diagonal([t2, t3]),
            )
        ],
        residual=IID(var=s2),
        coordinate="p",
        n=4,
    )
    dense = model.dense(F=[0, 0, 1, 1], p=[11, 10, 20, 21])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    assert panel.entry_mean(0, 0) == dense.matrix_mean()[1, 0]
    assert panel.entry_mean(0, 1) == dense.matrix_mean()[0, 0]


def test_position_covariance_requires_single_grouping_factor():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="F", intercept=True, cov=Scalar(t2)),
            RandomEffects(group="G", intercept=True, cov=Scalar(t2)),
        ],
        residual=IID(var=s2),
        coordinate="p",
        n=4,
    )
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 1, 0, 1], p=[10, 11, 20, 21])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    assert panel.entry_cross_covariance(0, 0, 1, 0) == t2
    with pytest.raises(NotDefined, match="single grouping factor"):
        panel.position_covariance(0, 0)


def test_symbolic_position_code_uses_factor_label():
    model = _model()
    panel = PanelView(
        model,
        model.factor("F"),
        Position("p", index=sp.Symbol("s", integer=True)),
    )
    i = sp.Symbol("i", integer=True)
    expected = sp.IndexedBase("k_F", shape=(4,))[i]
    assert panel.position_code(i) == expected
    assert panel.position_code(i) == model._occasion_code(i, "F")


def test_unbalanced_panel_rejects_position_mean_and_covariance():
    dense = _model().dense(F=[0, 0, 1], p=[10, 11, 20])
    panel = PanelView(dense, dense.factor("F"), Position("p"))
    with pytest.raises(NotDefined, match="balanced"):
        panel.position_mean(0)
    with pytest.raises(NotDefined, match="balanced"):
        panel.position_covariance(0, 1)


def _symbolic_ri_rs_model():
    b0, b1 = sp.symbols("b0 b1", real=True)
    tau0, tau1, sigma2 = sp.symbols("tau0 tau1 sigma2", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    i = sp.Symbol("i", integer=True)

    subject = Group("subject", index=i)
    time = Covariate("time")

    return LMM(
        fixed=Fixed(intercept=b0, slopes={time: b1}),
        random=[
            RandomEffects(
                group=subject,
                intercept=True,
                slopes=(RandomSlope(time),),
                cov=Unstructured.from_sd_corr([tau0, tau1], rho_b),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )


def test_symbolic_panel_entry_moments_use_block_entries():
    model = _symbolic_ri_rs_model()
    panel = model.panel("subject", "time")
    i = sp.Symbol("i", integer=True)
    t = sp.Symbol("t", integer=True)
    s = sp.Symbol("s", integer=True)
    block = model.block("subject", i)

    assert panel.entry_mean(i, t) == block.mean()[t, 0]
    assert panel.entry_covariance(i, t, s) == block.covariance()[t, s]
    assert panel.entry_cross_covariance(i, t, i, s) == block.covariance()[t, s]
