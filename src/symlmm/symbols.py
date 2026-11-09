"""Observation and parameter symbol helpers."""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]


def observation_index(name: str = "i") -> sp.Symbol:
    """Return a nonnegative integer observation index."""
    return sp.Symbol(name, integer=True, nonnegative=True)


def parameter(name: str, *, positive: bool = False, real: bool = False) -> sp.Symbol:
    """Return a model parameter symbol with explicit assumptions."""
    return sp.Symbol(name, positive=positive, real=real)
