"""Symbolic moments for linear mixed, marginal, and GEE models."""

from importlib.metadata import version

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
    "SymlmmError",
    "UnresolvedSymbols",
    "UnsupportedForm",
    "__version__",
    "require_form",
]
