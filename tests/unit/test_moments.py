"""Moment queries and form dispatch tests."""

import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar, Unstructured
from symlmm.errors import FormArityError, MissingFormError
from symlmm.forms import DualForm, EntryMap
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomIntercept, RandomSlope


def _model():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomIntercept(group="subject", cov=Scalar(tau2))],
        residual=IID(var=sigma2),
    )


def test_form_is_required():
    model = _model()
    with pytest.raises(MissingFormError):
        model.mean()


def test_invalid_arity_raises():
    model = _model()
    i, j = model.indices()
    with pytest.raises(FormArityError):
        model.mean(i, j, form="sum")


def test_both_returns_dual_form():
    model = _model()
    i, _ = model.indices()
    result = model.mean(i, form="both")
    assert isinstance(result, DualForm)
    assert result.sum is not None
    assert result.matrix is not None


def test_correlation_entry():
    model = _model()
    i, j = model.indices()
    assert model.correlation(i, j, form="sum") is not None


def test_conditional_mean_entry():
    model = _model()
    i, _ = model.indices()
    assert model.conditional_mean(i, form="sum") is not None


def test_conditional_mean_intercept_uses_group_level():
    model = _model()
    i, _ = model.indices()
    entry = model.conditional_mean(i, form="sum")

    indexed = {
        atom.base.name: atom for atom in entry.atoms(sp.Indexed) if atom.base.name.startswith("b_")
    }
    assert set(indexed) == {"b_subject_0"}
    assert indexed["b_subject_0"].indices[0] == model.group(i, "subject")

    level_0 = entry.subs({i: 0})
    level_1 = entry.subs({i: 1})
    assert level_0 != level_1


def test_conditional_mean_intercept_and_slope_do_not_collide():
    b0 = sp.Symbol("b0")
    tau2 = sp.Symbol("tau2", positive=True)
    s0, s1 = sp.symbols("s0 s1", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomIntercept(group="subject", cov=Scalar(tau2)),
            RandomSlope(
                group="subject",
                time="time",
                cov=Unstructured.from_sd_corr([s0, s1], rho_b),
            ),
        ],
        residual=IID(var=sigma2),
    )
    i, _ = model.indices()
    entry = sp.expand(model.conditional_mean(i, form="sum"))
    bases = {atom.base.name for atom in entry.atoms(sp.Indexed) if atom.base.name.startswith("b_")}
    assert bases == {"b_subject_0", "b_subject_1"}

    level = model.group(i, "subject")
    assert entry.coeff(sp.IndexedBase("b_subject_0")[level, 0]) == 1
    assert entry.coeff(sp.IndexedBase("b_subject_1")[level, 0]) == 1


def test_mean_sum_and_matrix_entries():
    model = _model()
    b0 = sp.Symbol("b0")
    i, _ = model.indices()

    assert model.mean(i, form="sum") == b0
    assert model.mean(i, form="matrix") is not None
    whole = model.mean(form="matrix")
    assert whole.shape == (model.n, 1)
    dual = model.mean(form="both")
    assert isinstance(dual, DualForm)
    assert isinstance(dual.sum, EntryMap)


def test_covariance_forms():
    model = _model()
    i, j = model.indices()

    sum_entry = model.covariance(i, j, form="sum")
    assert sum_entry == model.covariance_entry(i, j)
    assert model.covariance(i, j, form="matrix") is not None
    dual = model.covariance(i, j, form="both")
    assert isinstance(dual, DualForm)
    assert dual.sum == sum_entry

    whole = model.covariance(form="matrix")
    assert whole.shape == (model.n, model.n)
    whole_dual = model.covariance(form="both")
    assert isinstance(whole_dual.sum, EntryMap)


def test_covariance_requires_both_indices():
    model = _model()
    i, _ = model.indices()
    with pytest.raises(FormArityError):
        model.covariance(i, form="sum")


def test_variance_forms():
    model = _model()
    i, _ = model.indices()

    assert model.variance(i, form="sum") == model.covariance_entry(i, i)
    whole = model.variance(form="matrix")
    assert whole.shape == (model.n, 1)
    assert whole[0] is not None
    dual = model.variance(i, form="both")
    assert isinstance(dual, DualForm)
    assert dual.sum == model.covariance_entry(i, i)


def test_correlation_whole_and_matrix_entry():
    model = _model()
    i, j = model.indices()

    whole = model.correlation(form="matrix")
    assert whole.shape == (model.n, model.n)
    assert model.correlation(i, j, form="matrix") is not None
    dual = model.correlation(i, j, form="both")
    assert isinstance(dual, DualForm)
    assert dual.sum is not None


def test_conditional_mean_whole_and_matrix_entry():
    model = _model()
    i, _ = model.indices()

    whole = model.conditional_mean(form="matrix")
    assert whole.shape == (model.n, 1)
    dual = model.conditional_mean(form="both")
    assert isinstance(dual, DualForm)
    assert isinstance(dual.sum, EntryMap)
    entry = model.conditional_mean(i, form="matrix")
    assert entry.has(sp.Sum)
    n, q = model.dimensions()
    Z = sp.MatrixSymbol("Z", n, q)
    b = sp.MatrixSymbol("b", q, 1)
    assert entry.dummy_eq((model.matrix_mean() + Z * b)[i, 0])


def test_sum_whole_vector_queries_raise():
    model = _model()
    with pytest.raises(FormArityError):
        model.mean(form="sum")
    with pytest.raises(FormArityError):
        model.variance(form="sum")
    with pytest.raises(FormArityError):
        model.covariance(form="sum")
    with pytest.raises(FormArityError):
        model.conditional_mean(form="sum")
    with pytest.raises(FormArityError):
        model.correlation(form="sum")


def test_conditional_covariance_entry_sum():
    model = _model()
    i, j = model.indices()
    assert model.conditional_covariance(i, j, form="sum") == model.residual.entry(i, j, model)


def test_conditional_covariance_whole_matrix():
    model = _model()
    i, j = model.indices()

    whole = model.conditional_covariance(form="matrix")
    assert whole.shape == (model.n, model.n)
    assert whole == sp.MatrixSymbol("R", model.n, model.n)

    dual = model.conditional_covariance(form="both")
    assert isinstance(dual, DualForm)
    assert isinstance(dual.sum, EntryMap)
    assert model.conditional_covariance(i, j, form="matrix") is not None


def test_conditional_covariance_arity_and_sum_form_errors():
    model = _model()
    i, _ = model.indices()
    with pytest.raises(FormArityError):
        model.conditional_covariance(i, form="sum")
    with pytest.raises(FormArityError):
        model.conditional_covariance(form="sum")


def test_conditional_variance_entry_sum():
    model = _model()
    i, _ = model.indices()
    assert model.conditional_variance(i, form="sum") == model.residual.entry(i, i, model)


def test_conditional_variance_whole_and_form_errors():
    model = _model()
    i, _ = model.indices()

    whole = model.conditional_variance(form="matrix")
    assert whole.shape == (model.n, 1)
    assert model.conditional_variance(i, form="matrix") is not None
    dual = model.conditional_variance(form="both")
    assert isinstance(dual, DualForm)
    assert isinstance(dual.sum, EntryMap)

    with pytest.raises(FormArityError):
        model.conditional_variance(form="sum")
