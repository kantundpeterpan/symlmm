"""Parameter constraints and positive-definiteness checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

import sympy as sp  # type: ignore[import-untyped]


class Relation(str, Enum):
    """Constraint relation."""

    EQ = "eq"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


@dataclass(frozen=True)
class Constraint:
    """A named parameter constraint."""

    expr: sp.Expr
    relation: Relation
    message: str

    def holds(self, subs: Mapping[sp.Symbol, object]) -> bool | None:
        """Evaluate the constraint under substitutions; None means unknown."""
        value = sp.simplify(self.expr.subs(subs))
        if value.is_positive:
            return self.relation in (Relation.GT, Relation.GE)
        if value.is_zero:
            return self.relation in (Relation.EQ, Relation.LE, Relation.GE)
        if value.is_negative:
            return self.relation in (Relation.LT, Relation.LE)
        return None


def is_positive_definite(matrix: sp.MatrixBase, positive: Iterable[sp.Expr] = ()) -> bool | None:
    """Tri-state PD check using Sylvester minors.

    `positive` lists expressions known to be positive, such as
    `a*c - b**2` for a 2x2 covariance.
    """
    if not isinstance(matrix, sp.MatrixBase):
        return None
    size = matrix.rows
    if size == 0:
        return True
    known_positive = tuple(positive)
    for k in range(1, size + 1):
        minor = sp.simplify(matrix[:k, :k].det())
        if minor.is_positive:
            continue
        if any(sp.simplify(minor - expr) == 0 for expr in known_positive):
            continue
        if minor.is_nonpositive:
            return False
        return None
    return True


def check_constraints(
    constraints: Iterable[Constraint], subs: Mapping[sp.Symbol, object]
) -> tuple[str, ...]:
    """Return messages for constraints that evaluate to False."""
    violations = []
    for constraint in constraints:
        if constraint.holds(subs) is False:
            violations.append(constraint.message)
    return tuple(violations)
