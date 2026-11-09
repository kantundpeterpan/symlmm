import pytest
import sympy as sp

from symlmm.errors import FormArityError, FormError, InvalidFormError, MissingFormError
from symlmm.forms import DualForm, EntryMap, Form, require_form


def test_missing_form_raises():
    with pytest.raises(MissingFormError):
        require_form(None)


def test_invalid_form_raises():
    with pytest.raises(InvalidFormError):
        require_form("latex")


def test_form_arity_error_is_a_form_error():
    assert issubclass(FormArityError, FormError)


def test_valid_forms_return_enum():
    assert require_form("sum") is Form.SUM
    assert require_form("matrix") is Form.MATRIX
    assert require_form("both") is Form.BOTH


def test_entry_map_lookup():
    i, j = sp.symbols("i j", integer=True)
    entry_map = EntryMap(lambda a, b: a + b)
    assert entry_map[i, j] == i + j


def test_dual_form_holds_both_notations():
    dual = DualForm(sum=sp.Symbol("s"), matrix=sp.Symbol("m"))
    assert dual.sum == sp.Symbol("s")
    assert dual.matrix == sp.Symbol("m")
