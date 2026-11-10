"""Public exception hierarchy for symlmm."""


class SymlmmError(Exception):
    """Base class for all symlmm errors."""


class FormError(SymlmmError):
    """Base class for notation-selection errors."""


class MissingFormError(FormError, TypeError):
    """Raised when a public call omits the required `form` argument."""


class InvalidFormError(FormError, ValueError):
    """Raised when `form` is not one of 'sum', 'matrix', or 'both'."""


class FormArityError(FormError, ValueError):
    """Raised when an index arity is not valid for the requested form."""


class UnsupportedForm(FormError):
    """Raised when a quantity has no representation in the requested form."""


class ModelError(SymlmmError):
    """Base class for model-level errors."""


class InvalidKindError(ModelError, ValueError):
    """Raised when a variogram `kind` is not 'raw', 'residual', or 'conditional'."""


class NotDecomposable(ModelError):
    """Raised when a marginal model has no exact conditional representation."""

    def __init__(self, reason: str, suggestion: str) -> None:
        super().__init__(f"{reason}; suggestion: {suggestion}")
        self.reason = reason
        self.suggestion = suggestion


class NoConditionalModel(ModelError):
    """Raised when a model has no conditional counterpart."""


class NotDefined(ModelError):
    """Raised when a quantity is undefined for a model class."""


class UnsupportedMoment(ModelError, ValueError):
    """Raised when the blackboard moment engine cannot evaluate an expression."""


class UnresolvedSymbols(ModelError):
    """Raised when numeric evaluation still has free symbols."""


class CovarianceConstructionError(SymlmmError, ValueError):
    """Raised for invalid covariance-construction arguments."""


class ParameterConstraintError(ModelError):
    """Raised when covariance parameter constraints are violated."""

    def __init__(self, violations: list[str]) -> None:
        super().__init__("; ".join(violations))
        self.violations = tuple(violations)
