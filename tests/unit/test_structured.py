import sympy as sp

from symlmm.covariances import AR1, IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, RandomEffects
from symlmm.structured import (
    as_scalar,
    inverse_matrix,
    woodbury_covariance,
    woodbury_precision,
    woodbury_reduced_inverse,
    woodbury_shrinkage,
)


def _ri_model():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    subject = Group("subject")
    return (
        LMM(
            fixed=Fixed(),
            random=[
                RandomEffects(
                    group=subject,
                    intercept=True,
                    cov=Scalar(tau2),
                )
            ],
            residual=IID(var=sigma2),
        ),
        tau2,
        sigma2,
    )


def test_iid_precision_is_scaled_identity():
    model, _, sigma2 = _ri_model()
    i = sp.Symbol("i", integer=True)
    block = model.block("subject", i)

    R_inv = block.model.residual.precision(block)

    assert R_inv == sp.Identity(block.size()) / sigma2


def test_default_residual_precision_is_formal_inverse():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    rho = sp.Symbol("rho", real=True)
    i = sp.Symbol("i", integer=True)
    model = LMM(
        fixed=Fixed(),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                cov=Scalar(tau2),
            )
        ],
        residual=AR1(group="subject", coordinate="time", var=sigma2, rho=rho),
        coordinate="time",
    )
    block = model.block("subject", i)

    assert block.residual_precision() == block.residual().inverse()


def test_woodbury_ri_shrinkage_closed_form():
    model, tau2, sigma2 = _ri_model()
    i = sp.Symbol("i", integer=True)
    block = model.block("subject", i)
    term = block.term("subject")

    Z = sp.OneMatrix(block.size(), 1)
    G = term.random.cov.dense()
    R_inv = model.residual.precision(block)
    S = woodbury_shrinkage(G, Z, R_inv)

    expected = (tau2 / (sigma2 + block.size() * tau2)) * sp.OneMatrix(1, block.size())

    assert S.shape == (1, block.size())
    assert sp.simplify((S - expected)[0, 0]) == 0


def test_woodbury_ri_covariance_closed_form():
    model, tau2, sigma2 = _ri_model()
    i = sp.Symbol("i", integer=True)
    block = model.block("subject", i)
    term = block.term("subject")

    reduced = woodbury_covariance(
        term.random.cov.dense(),
        sp.OneMatrix(block.size(), 1),
        model.residual.precision(block),
    )

    expected = tau2 * sigma2 / (sigma2 + block.size() * tau2)

    assert sp.simplify(reduced - expected) == 0


def test_woodbury_precision_matches_explicit_inverse():
    G = sp.Matrix([[4, 1], [1, 3]])
    Z = sp.Matrix([[1, 0], [1, 1], [1, 2]])
    R_inv = sp.eye(3) / 2
    V = sp.eye(3) * 2 + Z * G * Z.T

    precision = woodbury_precision(G, Z, R_inv)
    reduced = woodbury_reduced_inverse(G, Z, R_inv)

    assert sp.simplify(precision - V.inv()) == sp.zeros(3, 3)
    assert reduced.shape == (2, 2)
    assert sp.simplify(reduced - (G.inv() + Z.T * R_inv * Z).inv()) == sp.zeros(2, 2)


def test_inverse_matrix_supports_explicit_and_symbolic_matrices():
    explicit = sp.Matrix([[2]])
    symbol = sp.MatrixSymbol("A", 2, 2)

    assert inverse_matrix(explicit) == sp.Matrix([[sp.Rational(1, 2)]])
    assert inverse_matrix(symbol) == symbol.inverse()
    assert inverse_matrix(4) == sp.Rational(1, 4)


def test_as_scalar_unwraps_only_one_by_one_matrices():
    value = sp.Symbol("x")

    assert as_scalar(sp.Matrix([[value]])) == value
    assert as_scalar(sp.MatrixSymbol("A", 1, 1)) == sp.MatrixSymbol("A", 1, 1)[0, 0]
    assert as_scalar(sp.MatrixSymbol("B", 2, 2)) == sp.MatrixSymbol("B", 2, 2)
    assert as_scalar(value) == value
