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
from .model import IndexedModel
from .specs import Crossed, Fixed, Group, Nested
from .symbols import observation_index, parameter

__version__ = version("symlmm")

__all__ = [
    "Constraint",
    "Crossed",
    "DualForm",
    "EntryMap",
    "Fixed",
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
    "Relation",
    "SymlmmError",
    "UnresolvedSymbols",
    "UnsupportedForm",
    "__version__",
    "check_constraints",
    "is_positive_definite",
    "observation_index",
    "parameter",
    "require_form",
]
