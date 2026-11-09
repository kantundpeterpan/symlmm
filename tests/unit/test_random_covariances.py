"""Random-effect and residual covariance structure tests."""

import pytest
import sympy as sp

from symlmm.covariances import (
    IID,
    Diagonal,
    RandomCovariance,
    Scalar,
    Unstructured,
)
from symlmm.errors import CovarianceConstructionError, SymlmmError


def test_scalar_entry():
    tau2 = sp.Symbol("tau2", positive=True)
    covariance = Scalar(tau2)

    assert covariance.dim == 1
    assert covariance.entry(0, 0) == tau2
    assert covariance.entry(0, 1) == 0


def test_scalar_dense_and_constraints():
    tau2 = sp.Symbol("tau2", positive=True)
    covariance = Scalar(tau2)

    assert covariance.dense() == sp.Matrix([[tau2]])
    assert covariance.is_positive_definite() is True
    constraints = covariance.constraints()
    assert len(constraints) == 1
    assert constraints[0].holds({tau2: 1}) is True
    assert constraints[0].holds({tau2: -1}) is False


def test_diagonal_entry():
    d0, d1 = sp.symbols("d0 d1", positive=True)
    covariance = Diagonal([d0, d1])

    assert covariance.dim == 2
    assert covariance.entry(0, 0) == d0
    assert covariance.entry(0, 1) == 0
    assert covariance.entry(1, 1) == d1


def test_diagonal_dense_and_constraints():
    d0, d1 = sp.symbols("d0 d1", positive=True)
    covariance = Diagonal([d0, d1])

    assert covariance.dense() == sp.Matrix([[d0, 0], [0, d1]])
    messages = [constraint.message for constraint in covariance.constraints()]
    assert messages == ["d0 > 0", "d1 > 0"]


def test_unstructured_from_sd_corr():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    covariance = Unstructured.from_sd_corr([s0, s1], rho)

    assert covariance.entry(0, 0) == s0**2
    assert covariance.entry(0, 1) == s0 * s1 * rho
    assert covariance.entry(1, 1) == s1**2
    assert any("rho" in constraint.message for constraint in covariance.constraints())


def test_unstructured_from_sd_corr_constraints_bound_correlation():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    covariance = Unstructured.from_sd_corr([s0, s1], rho)

    constraints = covariance.constraints()
    assert len(constraints) == 2
    assert all(constraint.holds({rho: 0.5}) is True for constraint in constraints)
    assert any(constraint.holds({rho: 1.5}) is False for constraint in constraints)
    assert any(constraint.holds({rho: -1.5}) is False for constraint in constraints)


def test_unstructured_from_sd_corr_is_keyword_friendly():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    covariance = Unstructured.from_sd_corr(sd=[s0, s1], corr=rho)

    assert covariance.dim == 2
    assert covariance.dense() == sp.Matrix([[s0**2, s0 * s1 * rho], [s0 * s1 * rho, s1**2]])


def test_unstructured_from_sd_corr_rejects_non_2x2():
    s0 = sp.Symbol("s0", positive=True)
    rho = sp.Symbol("rho", real=True)

    with pytest.raises(CovarianceConstructionError, match="2x2"):
        Unstructured.from_sd_corr([s0], rho)


def test_covariance_construction_error_is_value_and_symlmm_error():
    s0 = sp.Symbol("s0", positive=True)
    rho = sp.Symbol("rho", real=True)

    with pytest.raises(CovarianceConstructionError) as excinfo:
        Unstructured.from_sd_corr([s0], rho)

    assert isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value, SymlmmError)


def test_unstructured_from_cholesky():
    a, b, c = sp.symbols("a b c", positive=True)
    lower = sp.Matrix([[a, 0], [b, c]])
    covariance = Unstructured.from_cholesky(lower)

    assert covariance.dim == 2
    assert covariance.dense() == lower * lower.T
    assert covariance.entry(0, 0) == a**2
    assert covariance.entry(0, 1) == a * b
    assert covariance.entry(1, 1) == b**2 + c**2
    assert covariance.constraints() == ()


def test_unstructured_rejects_asymmetric_matrix():
    with pytest.raises(CovarianceConstructionError, match="symmetric"):
        Unstructured(sp.Matrix([[1, 2], [3, 4]]))


def test_unstructured_rejects_non_square_matrix():
    with pytest.raises(CovarianceConstructionError, match="square"):
        Unstructured(sp.Matrix([[1, 2, 3], [4, 5, 6]]))


def test_unstructured_rejects_non_positive_definite_matrix():
    with pytest.raises(CovarianceConstructionError, match="positive definite"):
        Unstructured(sp.Matrix([[-1]]))


def test_unstructured_from_sd_corr_and_cholesky_construct_valid():
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho = sp.Symbol("rho", real=True)
    assert Unstructured.from_sd_corr([s0, s1], rho).dim == 2

    a, b, c = sp.symbols("a b c", positive=True)
    assert Unstructured.from_cholesky(sp.Matrix([[a, 0], [b, c]])).dim == 2


def test_iid_entry_and_constraints():
    sigma2 = sp.Symbol("sigma2", positive=True)
    covariance = IID(var=sigma2)
    model = object()

    assert covariance.entry(0, 0, model) == sigma2
    assert covariance.entry(0, 1, model) == 0
    constraint = covariance.constraints()[0]
    assert constraint.message == "sigma2 > 0"
    assert constraint.holds({sigma2: 1}) is True
    assert constraint.holds({sigma2: 0}) is False


def test_concrete_covariances_share_random_covariance_interface():
    d0 = sp.Symbol("d0", positive=True)

    for covariance in (Scalar(d0), Diagonal([d0]), Unstructured(sp.Matrix([[d0]]))):
        assert isinstance(covariance, RandomCovariance)
        assert isinstance(covariance.dim, int)
        assert covariance.dense().shape == (covariance.dim, covariance.dim)
