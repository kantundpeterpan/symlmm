"""Term-level covariance decomposition and partial covariance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]

from .errors import ModelError, NotDefined
from .specs import Factor, RandomEffects

RESIDUAL = "residual"


@dataclass(frozen=True)
class Term:
    """A named random-effect component."""

    name: str
    factor: Factor
    random: RandomEffects
    index: int


def terms(model: Any) -> tuple[Term, ...]:
    """Return named random-effect terms in model order."""
    resolved = []
    for index, random in enumerate(model.random):
        factor = model.factor(random.group.name)
        name = random.name or factor.label
        resolved.append(Term(name=name, factor=factor, random=random, index=index))
    names = [term.name for term in resolved]
    if len(names) != len(set(names)):
        raise ModelError("random-effect term names must be unique")
    if RESIDUAL in names:
        raise ModelError("'residual' is reserved for the residual term")
    return tuple(resolved)


def validate_exclusions(terms: tuple[Term, ...], exclude: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and de-duplicate public exclusion names."""
    names = {term.name for term in terms}
    names.add(RESIDUAL)
    unknown = sorted(set(exclude) - names)
    if unknown:
        raise ModelError(f"unknown term names: {unknown!r}")
    return tuple(dict.fromkeys(exclude))


def term_value(model: Any, term: Term, a: Any, b: Any) -> Any:
    """Return one term's contribution to Cov(Y_a, Y_b)."""
    return model._random_term_entry(term.random, a, b)


def covariance_terms(model: Any, a: Any, b: Any) -> dict[str, Any]:
    """Return per-term contributions to Cov(Y_a, Y_b)."""
    resolved = terms(model)
    values = {term.name: term_value(model, term, a, b) for term in resolved}
    values[RESIDUAL] = model.residual.entry(a, b, model)
    return values


def partial_covariance(
    model: Any,
    a: Any,
    b: Any,
    exclude: tuple[str, ...] = (),
) -> Any:
    """Return Cov(Y_a, Y_b) with named term contributions removed."""
    resolved = terms(model)
    validate_exclusions(resolved, exclude)
    total = model.covariance_entry(a, b)
    for term in resolved:
        if term.name in exclude:
            total -= term_value(model, term, a, b)
    if RESIDUAL in exclude:
        total -= model.residual.entry(a, b, model)
    return sp.simplify(total)


def partial_correlation(
    model: Any,
    a: Any,
    b: Any,
    exclude: tuple[str, ...] = (),
) -> Any:
    """Return the partial correlation between two scalar entries."""
    covariance = partial_covariance(model, a, b, exclude=exclude)
    variance_a = partial_covariance(model, a, a, exclude=exclude)
    variance_b = partial_covariance(model, b, b, exclude=exclude)
    denominator = sp.sqrt(variance_a * variance_b)
    if denominator == 0:
        raise NotDefined("partial correlation requires nonzero partial variances")
    return sp.simplify(covariance / denominator)


__all__ = [
    "RESIDUAL",
    "Term",
    "covariance_terms",
    "partial_correlation",
    "partial_covariance",
    "terms",
    "validate_exclusions",
]
