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
from .model import IndexedModel
from .specs import Crossed, Group, Nested
from .symbols import observation_index, parameter

__version__ = version("symlmm")

__all__ = [
    "Crossed",
    "DualForm",
    "EntryMap",
    "Form",
    "FormArityError",
    "FormError",
    "Group",
    "IndexedModel",
    "InvalidFormError",
    "MissingFormError",
    "ModelError",
    "Nested",
    "NoConditionalModel",
    "NotDecomposable",
    "NotDefined",
    "ParameterConstraintError",
    "SymlmmError",
    "UnresolvedSymbols",
    "UnsupportedForm",
    "__version__",
    "observation_index",
    "parameter",
    "require_form",
]
