"""Canonicalization and rewrite rules."""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]


def canonicalize(expr: sp.Expr) -> sp.Expr:
    """Return a canonical form for comparison and printing."""
    return sp.simplify(sp.expand(expr))
