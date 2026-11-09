"""Symbolic moments for linear mixed, marginal, and GEE models."""

from importlib.metadata import version

from .constraints import (
    Constraint,
    Relation,
    check_constraints,
    is_positive_definite,
)
from .covariances import (
    AR1,
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
from .forms import (
    DualForm,
    EntryMap,
    Form,
    matrix_to_sum,
    require_form,
    sum_to_matrix,
)
from .model import LMM, IndexedModel
from .simplify import canonicalize
from .specs import Crossed, Fixed, Group, Nested, RandomIntercept
from .symbols import observation_index, parameter

__version__ = version("symlmm")

__all__ = [
    "AR1",
    "IID",
    "LMM",
    "Constraint",
    "CovarianceConstructionError",
    "CovarianceSpec",
    "Crossed",
    "Diagonal",
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
    "RandomCovariance",
    "RandomIntercept",
    "Relation",
    "ResidualCovariance",
    "Scalar",
    "SymlmmError",
    "UnresolvedSymbols",
    "Unstructured",
    "UnsupportedForm",
    "__version__",
    "canonicalize",
    "check_constraints",
    "is_positive_definite",
    "matrix_to_sum",
    "observation_index",
    "parameter",
    "require_form",
    "sum_to_matrix",
]
