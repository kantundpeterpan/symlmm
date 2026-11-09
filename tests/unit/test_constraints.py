"""Constraint evaluation and positive-definiteness tests."""

import sympy as sp

from symlmm.constraints import (
    Constraint,
    Relation,
    check_constraints,
    is_positive_definite,
)


def test_pd_tri_state():
    a, b, c = sp.symbols("a b c", positive=True)
    matrix = sp.Matrix([[a, b], [b, c]])

    assert is_positive_definite(matrix) is None
    assert is_positive_definite(matrix, positive=[a * c - b**2]) is True


def test_constraint_evaluation():
    x = sp.Symbol("x", positive=True)
    constraint = Constraint(x - 1, Relation.GT, "x > 1")

    assert constraint.holds({x: 2}) is True
    assert constraint.holds({x: 1}) is False
    assert constraint.holds({}) is None


def test_check_constraints_reports_violations():
    x = sp.Symbol("x", positive=True)
    constraints = [
        Constraint(x - 1, Relation.GT, "x > 1"),
        Constraint(x - 2, Relation.GT, "x > 2"),
    ]

    assert check_constraints(constraints, {x: 2}) == ("x > 2",)
    assert check_constraints(constraints, {}) == ()
