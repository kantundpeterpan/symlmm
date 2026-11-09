"""Symbolic moments for linear mixed, marginal, and GEE models."""

from importlib.metadata import version

from .constraints import (
    Constraint,
    Relation,
    check_constraints,
    is_positive_definite,
)
from .covariances import (
    IID,
    CovarianceSpec,
    Diagonal,
    RandomCovariance,
    ResidualCovariance,
    Scalar,
    Unstructured,
)
from .errors import (
    CovarianceConstructionError,
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
    "IID",
    "Constraint",
    "CovarianceConstructionError",
    "CovarianceSpec",
    "Crossed",
    "Diagonal",
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
    "RandomCovariance",
    "Relation",
    "ResidualCovariance",
    "Scalar",
    "SymlmmError",
    "UnresolvedSymbols",
    "Unstructured",
    "UnsupportedForm",
    "__version__",
    "check_constraints",
    "is_positive_definite",
    "observation_index",
    "parameter",
    "require_form",
]
