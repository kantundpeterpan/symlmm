"""Symbolic moments for linear mixed, marginal, and GEE models."""

from importlib.metadata import version

from symlmm.constraints import (
    Constraint,
    Relation,
    check_constraints,
    is_positive_definite,
)

__version__ = version("symlmm")

__all__ = [
    "Constraint",
    "Relation",
    "__version__",
    "check_constraints",
    "is_positive_definite",
]
