"""Symbolic moments for linear mixed, marginal, and GEE models."""

from importlib.metadata import version

from .blup import blup_covariance, blup_cross_covariance, blup_mean
from .constraints import (
    Constraint,
    Relation,
    check_constraints,
    is_positive_definite,
)
from .covariances import (
    AR1,
    IID,
    BlockDiag,
    CompoundSymmetry,
    CovarianceSpec,
    Diagonal,
    Exponential,
    Gaussian,
    Heteroscedastic,
    Matern,
    RandomCovariance,
    ResidualCovariance,
    Scalar,
    Toeplitz,
    Unstructured,
)
from .errors import (
    CovarianceConstructionError,
    FormArityError,
    FormError,
    InvalidFormError,
    InvalidKindError,
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
from .specs import Crossed, Fixed, Group, Nested, RandomEffects, RandomSlope
from .symbols import observation_index, parameter
from .variogram import covariogram, variogram

__version__ = version("symlmm")

__all__ = [
    "AR1",
    "IID",
    "LMM",
    "BlockDiag",
    "CompoundSymmetry",
    "Constraint",
    "CovarianceConstructionError",
    "CovarianceSpec",
    "Crossed",
    "Diagonal",
    "DualForm",
    "EntryMap",
    "Exponential",
    "Fixed",
    "Form",
    "FormArityError",
    "FormError",
    "Gaussian",
    "Group",
    "Heteroscedastic",
    "IndexedModel",
    "InvalidFormError",
    "InvalidKindError",
    "Matern",
    "MissingFormError",
    "ModelError",
    "Nested",
    "NoConditionalModel",
    "NotDecomposable",
    "NotDefined",
    "ParameterConstraintError",
    "RandomCovariance",
    "RandomEffects",
    "RandomSlope",
    "Relation",
    "ResidualCovariance",
    "Scalar",
    "SymlmmError",
    "Toeplitz",
    "UnresolvedSymbols",
    "Unstructured",
    "UnsupportedForm",
    "__version__",
    "blup_covariance",
    "blup_cross_covariance",
    "blup_mean",
    "canonicalize",
    "check_constraints",
    "covariogram",
    "is_positive_definite",
    "matrix_to_sum",
    "observation_index",
    "parameter",
    "require_form",
    "sum_to_matrix",
    "variogram",
]
