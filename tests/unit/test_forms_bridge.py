import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.forms import EntryMap, matrix_to_sum, sum_to_matrix
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept


def test_random_intercept_sum_entry():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()

    expected = tau2 * sp.KroneckerDelta(
        model.group(i, "subject"), model.group(j, "subject")
    ) + sigma2 * sp.KroneckerDelta(i, j)
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0


def test_matrix_to_sum_builds_lazy_entry_map():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    entry_map = matrix_to_sum(model.matrix_covariance(), model)

    assert isinstance(entry_map, EntryMap)
    assert sp.simplify(entry_map[i, j] - model.covariance_entry(i, j)) == 0


def test_sum_to_matrix_returns_abstract_covariance():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )

    assert sum_to_matrix(sp.Symbol("entry"), model) == model.matrix_covariance()
