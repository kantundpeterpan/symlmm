"""LMM assembly and matrix-form tests."""

import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, RandomIntercept


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
