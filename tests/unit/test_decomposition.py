import pytest
import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.decomposition import (
    covariance_terms,
    partial_correlation,
    partial_covariance,
    terms,
)
from symlmm.errors import ModelError
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


def _model():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="F", intercept=True, cov=Scalar(t2), name="f"),
            RandomEffects(group="G", intercept=True, cov=Scalar(t2), name="g"),
        ],
        residual=IID(var=s2),
        n=2,
    )


def test_terms_are_named_and_ordered():
    model = _model()
    assert [term.name for term in terms(model)] == ["f", "g"]


def test_covariance_terms_sum_to_total():
    model = _model()
    total = model.covariance_entry(0, 1)
    values = covariance_terms(model, 0, 1)
    assert sum(values.values()) == total
    assert list(values)[-1] == "residual"


def test_partial_covariance_excludes_term():
    model = _model()
    total = model.covariance_entry(0, 1)
    values = covariance_terms(model, 0, 1)
    assert partial_covariance(model, 0, 1, exclude=("f",)) == total - values["f"]


def test_partial_covariance_excludes_everything():
    model = _model()
    assert partial_covariance(model, 0, 1, exclude=("f", "g", "residual")) == 0


def test_partial_correlation_diagonal_is_one():
    model = _model()
    assert partial_correlation(model, 0, 0, exclude=("f",)) == 1


def test_unknown_exclusion_raises():
    model = _model()
    with pytest.raises(ModelError, match="unknown term"):
        partial_covariance(model, 0, 1, exclude=("missing",))


def test_residual_name_is_reserved():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[RandomEffects(group="F", intercept=True, cov=Scalar(t2), name="residual")],
        residual=IID(var=s2),
        n=2,
    )
    with pytest.raises(ModelError, match="reserved"):
        terms(model)


def test_duplicate_exclusion_names_raise():
    model = _model()
    with pytest.raises(ModelError, match="duplicate"):
        partial_covariance(model, 0, 1, exclude=("f", "f"))
