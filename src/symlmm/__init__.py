"""Symbolic moments for linear mixed, marginal, and GEE models."""

from importlib.metadata import version

from .model import IndexedModel
from .specs import Crossed, Group, Nested
from .symbols import observation_index, parameter

__version__ = version("symlmm")

__all__ = [
    "Crossed",
    "Group",
    "IndexedModel",
    "Nested",
    "__version__",
    "observation_index",
    "parameter",
]
