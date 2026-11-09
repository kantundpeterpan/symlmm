"""Symbolic moments for linear mixed, marginal, and GEE models."""

from importlib.metadata import version

from .constraints import (
    Constraint,
    Relation,
    check_constraints,
    is_positive_definite,
)
from .errors import (
    FormArityError,
    FormError,
    InvalidFormError,
    MissingFormError,
    ModelError,
    NoConditionalModel,
    NotDecomposable,
    NotDefined,
    ParameterConstraintError,
    SymlmmError,
    UnresolvedSymbols,
    UnsupportedForm,
)
from .forms import DualForm, EntryMap, Form, require_form

__version__ = version("symlmm")

__all__ = [
    "Constraint",
    "DualForm",
    "EntryMap",
    "Form",
    "FormArityError",
    "FormError",
    "InvalidFormError",
    "MissingFormError",
    "ModelError",
    "NoConditionalModel",
    "NotDecomposable",
    "NotDefined",
    "ParameterConstraintError",
    "Relation",
    "SymlmmError",
    "UnresolvedSymbols",
    "UnsupportedForm",
    "__version__",
    "check_constraints",
    "is_positive_definite",
    "require_form",
]
