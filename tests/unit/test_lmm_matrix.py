"""LMM assembly and matrix-form tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.errors import ModelError
from symlmm.model import LMM
from symlmm.specs import Crossed, Fixed, Group, Nested, RandomIntercept


def test_matrix_shapes():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    n, q = model.dimensions()
    m_subject = sp.Symbol("m_subject", positive=True, integer=True)

    assert model.matrix_mean().shape == (n, 1)
    assert model.matrix_covariance().shape == (n, n)
    assert q == m_subject


def test_explicit_n_and_group_registration():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    n = sp.Symbol("n", positive=True, integer=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        n=n,
    )

    assert model.dimensions() == (n, sp.Symbol("m_subject", positive=True, integer=True))
    assert model.groups == (Group("subject"),)
    assert model.matrix_mean().shape == (n, 1)
    assert model.matrix_covariance().shape == (n, n)


def test_levels_use_composite_labels_for_nested_and_crossed():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)

    nested = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="visit", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=[Nested("visit", within="subject")],
    )
    assert nested.dimensions()[1] == sp.Symbol("m_subject_visit", positive=True, integer=True)

    crossed = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="item", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=[Crossed("subject", "item")],
    )
    assert crossed.dimensions()[1] == sp.Symbol("m_subject_item", positive=True, integer=True)

    plain = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    assert plain.dimensions()[1] == sp.Symbol("m_subject", positive=True, integer=True)


def test_conflicting_duplicate_grouping_specs_raise():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)

    with pytest.raises(ModelError, match="conflicting grouping-factor specs"):
        LMM(
            fixed=Fixed(intercept=b0),
            random=[RandomIntercept(group="visit", cov=Scalar(tau2))],
            residual=IID(var=sigma2),
            groups=[Nested("visit", within="subject"), Crossed("subject", "visit")],
        )


def test_identical_duplicate_grouping_specs_are_deduplicated():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        groups=[Group("subject"), Group("subject")],
    )

    assert model.groups == (Group("subject"),)


def test_multiple_random_terms_sum_dimensions():
    b0, tau2, sigma2 = sp.symbols("b0 tau2 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomIntercept(group="subject", cov=Scalar(tau2)),
            RandomIntercept(group="item", cov=Scalar(tau2)),
        ],
        residual=IID(var=sigma2),
    )
    _, q = model.dimensions()

    assert q == sp.Symbol("m_subject", positive=True, integer=True) + sp.Symbol(
        "m_item", positive=True, integer=True
    )
