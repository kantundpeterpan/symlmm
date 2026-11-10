"""Blackboard-style symbolic moments on canonical group/occasion indexing.

Importing this module registers :meth:`LMM.symbols` so the facade can be used
without editing the controller-owned model module.

The facade delegates moment definitions to the canonical model layer, including
the private ``_entry_row`` and ``_primary_random_term`` helpers, so covariance
math stays in one place.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sympy as sp  # type: ignore[import-untyped]
from sympy.matrices.expressions.matexpr import (  # type: ignore[import-untyped]
    MatrixElement,
    MatrixSymbol,
)
from sympy.matrices.expressions.slice import MatrixSlice  # type: ignore[import-untyped]
from sympy.matrices.expressions.transpose import Transpose  # type: ignore[import-untyped]

from .errors import ModelError, NotDefined, UnsupportedMoment
from .model import LMM

__all__ = ["GroupEffectSymbols", "ModelSymbols", "ResponseSymbols", "symbols"]

_MATRIX_KINDS = ("response_matrix", "unsupported")


def _is_matrix(value: Any) -> bool:
    """Return whether `value` is a SymPy matrix expression."""
    return isinstance(value, (sp.MatrixExpr, sp.MatrixBase))


def _full_slice(value: Any) -> bool:
    """Return whether `value` is a full ``:`` slice."""
    return (
        isinstance(value, slice)
        and value.start is None
        and value.stop is None
        and value.step is None
    )


def _zero_matrix(rows: Any, cols: Any) -> Any:
    """Return a zero matrix with symbolic or concrete dimensions."""
    if isinstance(rows, int) and isinstance(cols, int):
        return sp.zeros(rows, cols)
    return sp.ZeroMatrix(rows, cols)


def _zero_for_expr(expr: Any) -> Any:
    """Return the additive identity matching `expr`."""
    shape = getattr(expr, "shape", None)
    if shape is None:
        return sp.Integer(0)
    return _zero_matrix(shape[0], shape[1])


def _component_length(expr: Any) -> Any | None:
    """Return the component count of a vector expression, if it is one."""
    shape = getattr(expr, "shape", None)
    if shape is None:
        return None
    if shape[0] == 1:
        return shape[1]
    if shape[1] == 1:
        return shape[0]
    return None


def _zero_covariance(expr1: Any, expr2: Any) -> Any:
    """Return a zero covariance block matching two deterministic expressions."""
    left = _component_length(expr1)
    right = _component_length(expr2)
    if left is None and right is None:
        return sp.Integer(0)
    if left is None:
        return _zero_matrix(1, right)
    if right is None:
        return _zero_matrix(left, 1)
    return _zero_matrix(left, right)


def _transpose(value: Any) -> Any:
    """Return a transpose only for matrix-valued coefficients."""
    return value.T if _is_matrix(value) else value


def _chain(*values: Any) -> Any:
    """Multiply factors left-to-right while dropping scalar identity factors."""
    result: Any = sp.Integer(1)
    for value in values:
        if value == 1:
            continue
        result = value if result == 1 else result * value
    return result


def _scalarize(value: Any) -> Any:
    """Return the scalar entry of a 1 x 1 matrix result."""
    if _is_matrix(value) and getattr(value, "shape", None) == (1, 1):
        return value[0, 0]
    return value


@dataclass(frozen=True)
class _Atom:
    """One canonical random atom."""

    kind: str
    group: Any = None
    occasion: Any = None
    component: Any = None
    orientation: str = "scalar"


@dataclass(frozen=True)
class _Term:
    """A scalar or matrix coefficient multiplying one random atom."""

    coeff: Any
    atom: _Atom
    side: str


class ResponseSymbols:
    """Canonical response symbols ``Y[group]``, ``Y[:, occasion]``, and ``Y[g, t]``.

    Dense models require concrete group labels for group-indexed responses;
    symbolic canonical group indices raise ``NotDefined`` instead of producing
    empty blocks. Use ``Y[:, t]`` or a concrete row for dense panels.
    """

    def __init__(self, model: LMM) -> None:
        self.model = model

    def _require_group_label(self, group: Any) -> Any:
        """Return a concrete dense group code, or validate a symbolic model.

        This intentionally reads the private dense flag because group-indexed
        responses are a dense-only ambiguity, not a public model query.
        """
        if not self.model._dense:
            return group
        count = self.model.group_count()
        if (
            not isinstance(count, int)
            or not isinstance(group, (int, sp.Integer))
            or not 0 <= int(group) < count
        ):
            raise NotDefined(
                "dense group response requires a concrete group label; "
                "use Y[:, t] or a concrete row"
            )
        return int(group)

    def __getitem__(self, key: Any) -> Any:
        """Dispatch on the model's canonical group and occasion index symbols."""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ModelError("response indexing requires one or two indices")
            row, column = key
            if _full_slice(row) and _full_slice(column):
                return self.model.response_matrix()
            if _full_slice(row):
                if isinstance(column, slice) and not _full_slice(column):
                    raise ModelError("response slices must be full slices")
                return self.model.time_response(column)
            if isinstance(row, slice):
                raise ModelError("response slices must be full slices")
            group = self._require_group_label(row)
            if _full_slice(column):
                return self.model.response(group)
            if isinstance(column, slice):
                raise ModelError("response slices must be full slices")
            return self.model.response(group, column)
        if self.model.occasion_index is not None and key == self.model.occasion_index:
            return self.model.time_response(key)
        if self.model.group_index is not None and key == self.model.group_index:
            return self.model.response(self._require_group_label(key))
        raise ModelError(
            "ambiguous response index; use Y[group_index], Y[occasion_index], Y[i, :], or Y[:, t]"
        )


class GroupEffectSymbols:
    """Group-level random-effect symbols ``b[group]``.

    When multiple random grouping factors exist, only the primary grouping
    factor has a canonical group-level effect vector.
    """

    def __init__(self, model: LMM) -> None:
        self.model = model
        self._by_expression: dict[Any, Any] = {}
        self._by_group: dict[Any, Any] = {}
        if model.group_index is not None and model.random:
            primary = model.random_effect(model.group_index)
            self._by_expression[primary] = model.group_index
            self._by_group[model.group_index] = primary

    def _canonical_group(self, group: Any) -> Any:
        """Normalize a group definition or name to its canonical index symbol."""
        index = getattr(group, "index", group)
        name = getattr(group, "name", group)
        if self.model.group_index is not None and (
            index == self.model.group_index or name == self.model.primary_group
        ):
            return self.model.group_index
        return index

    def require_moment_group(self, group: Any) -> Any:
        """Return the canonical group index when it is valid for effect moments."""
        canonical = self._canonical_group(group)
        groups = {term.group for term in self.model.random}
        if len(groups) <= 1:
            return canonical
        if canonical == self.model.group_index:
            return canonical
        raise NotDefined(
            f"random-effect moments require the primary group index; {group!r} is not primary"
        )

    def __getitem__(self, group: Any) -> Any:
        """Return ``b_group`` for the primary canonical group index."""
        canonical = self.require_moment_group(group)
        existing = self._by_group.get(group)
        if existing is not None:
            return existing
        symbol = self.model.random_effect(canonical)
        self._by_group[group] = symbol
        self._by_group.setdefault(canonical, symbol)
        self._by_expression[symbol] = canonical
        return symbol

    def group_of(self, expression: Any) -> Any:
        """Return the group index for a registered random-effect base."""
        known = self._by_expression.get(expression)
        if known is not None:
            return self._canonical_group(known)
        raise ModelError(f"not a registered random-effect atom: {expression!r}")


@dataclass(frozen=True)
class ModelSymbols:
    """Model symbols and moment operators for blackboard-style work."""

    model: LMM
    Y: ResponseSymbols
    b: GroupEffectSymbols
    i: Any
    t: Any
    X: Callable[[Any], Any]
    Z: Callable[[Any], Any]
    R: Callable[[Any], Any]
    G: Any
    beta: sp.Matrix

    def E(self, expr: Any, *, given: Any = None) -> Any:
        """Return the expectation of a linear expression in canonical atoms."""
        return _expectation(self, expr, given=given)

    def Var(self, expr: Any, *, given: Any = None) -> Any:
        """Return the variance of a linear expression in canonical atoms."""
        return _covariance(self, expr, expr, given=given)

    def Cov(self, expr1: Any, expr2: Any, *, given: Any = None) -> Any:
        """Return the covariance of two linear expressions in canonical atoms."""
        return _covariance(self, expr1, expr2, given=given)

    def total_variance(self, expr: Any, *, given: Any = None) -> Any:
        """Return ``Var(E(expr|given)) + E(Var(expr|given))``."""
        if given is None:
            return self.Var(expr)
        conditional_mean = self.E(expr, given=given)
        conditional_variance = self.Var(expr, given=given)
        return self.Var(conditional_mean) + self.E(conditional_variance)

    def total_covariance(self, expr1: Any, expr2: Any, *, given: Any = None) -> Any:
        """Return the law-of-total-covariance decomposition."""
        if given is None:
            return self.Cov(expr1, expr2)
        conditional_mean1 = self.E(expr1, given=given)
        conditional_mean2 = self.E(expr2, given=given)
        conditional_covariance = self.Cov(expr1, expr2, given=given)
        return self.Cov(conditional_mean1, conditional_mean2) + self.E(conditional_covariance)


def symbols(model: LMM) -> ModelSymbols:
    """Return blackboard symbols for `model`."""
    coefficients = model.fixed.coefficients()
    beta = (
        sp.zeros(len(coefficients), 1)
        if not coefficients
        else sp.Matrix(len(coefficients), 1, coefficients)
    )
    term = model._primary_random_term()
    G = sp.zeros(0, 0) if term is None else term.cov.dense()
    return ModelSymbols(
        model=model,
        Y=ResponseSymbols(model),
        b=GroupEffectSymbols(model),
        i=model.group_index,
        t=model.occasion_index,
        X=model.fixed_design,
        Z=model.random_design,
        R=model.residual_covariance,
        G=G,
        beta=beta,
    )


def _symbols_method(self: LMM) -> ModelSymbols:
    """Return blackboard symbols for this model."""
    return symbols(self)


if not hasattr(LMM, "symbols"):
    # ``model.py`` is controller-owned; register the facade lazily from here.
    LMM.symbols = _symbols_method  # type: ignore[attr-defined]


def _response_matrix(symbols: ModelSymbols) -> Any | None:
    """Return the canonical response matrix, or None when it is undefined."""
    try:
        return symbols.model.response_matrix()
    except NotDefined:
        return None


def _is_effect_base(expression: Any) -> bool:
    """Return whether `expression` looks like a random-effect base symbol."""
    return isinstance(expression, MatrixSymbol) and expression.name.startswith("b_")


def _match_atom(symbols: ModelSymbols, expr: Any) -> _Atom | None:
    """Return the canonical random atom represented by `expr`, if any."""
    Y = _response_matrix(symbols)
    if isinstance(expr, MatrixElement):
        base = expr.args[0]
        if _is_effect_base(base):
            group = symbols.b.group_of(base)
            return _Atom("effect_entry", group=group, component=expr.args[1])
        if Y is not None and base == Y:
            return _Atom("entry", group=expr.args[1], occasion=expr.args[2])
        return None
    if isinstance(expr, MatrixSlice):
        if Y is None or expr.args[0] != Y:
            return None
        row_slice = expr.args[1]
        col_slice = expr.args[2]
        rows, cols = Y.shape
        if expr.rows == 1 and col_slice[0] == 0 and col_slice[1] == cols:
            return _Atom("group_row", group=row_slice[0], orientation="row")
        if expr.cols == 1 and row_slice[0] == 0 and row_slice[1] == rows:
            return _Atom("occasion_column", occasion=col_slice[0], orientation="col")
        return _Atom("unsupported", orientation="matrix")
    if isinstance(expr, MatrixSymbol):
        if _is_effect_base(expr):
            group = symbols.b.group_of(expr)
            return _Atom("effect_vector", group=group, orientation="col")
        if Y is not None and expr == Y:
            return _Atom("response_matrix", orientation="matrix")
        return None
    if isinstance(expr, Transpose):
        inner = _match_atom(symbols, expr.args[0])
        if inner is None:
            return None
        if inner.kind == "effect_vector" and inner.orientation == "col":
            return _Atom("effect_vector", group=inner.group, orientation="row")
        if inner.kind == "occasion_column":
            return _Atom("occasion_column", occasion=inner.occasion, orientation="row")
        return None
    return None


def _contains_random(symbols: ModelSymbols, expr: Any) -> bool:
    """Return whether `expr` contains a canonical random atom."""
    return any(_match_atom(symbols, node) is not None for node in sp.preorder_traversal(expr))


def _product(values: list[Any]) -> Any:
    """Return the ordered product of `values`, or one for an empty list."""
    result: Any = sp.Integer(1)
    for value in values:
        result = value if result == 1 else result * value
    return result


def _parse_term(symbols: ModelSymbols, term: Any) -> _Term | None:
    """Parse one additive term, returning None for deterministic terms."""
    atom = _match_atom(symbols, term)
    if atom is not None:
        if atom.kind in _MATRIX_KINDS:
            raise UnsupportedMoment("moment engine does not support the full response matrix")
        return _Term(sp.Integer(1), atom, _default_side(atom))
    if not _contains_random(symbols, term):
        return None
    if not term.is_Mul:
        raise UnsupportedMoment("moment engine supports linear expressions only")
    factors = list(term.args)
    matches = []
    for index, factor in enumerate(factors):
        candidate = _match_atom(symbols, factor)
        if candidate is not None:
            matches.append((index, candidate))
    if len(matches) != 1:
        raise UnsupportedMoment("moment engine supports linear expressions only")
    index, matched = matches[0]
    if matched.kind in _MATRIX_KINDS:
        raise UnsupportedMoment("moment engine supports linear expressions only")
    before = factors[:index]
    after = factors[index + 1 :]
    if matched.orientation == "scalar":
        coefficient = _product(before + after)
        if _is_matrix(coefficient):
            raise UnsupportedMoment(
                "matrix coefficients on scalar response atoms are not supported"
            )
        if _contains_random(symbols, coefficient):
            raise UnsupportedMoment("moment engine supports linear expressions only")
        return _Term(coefficient, matched, "scalar")
    if matched.orientation == "col":
        if any(_is_matrix(value) for value in after):
            raise UnsupportedMoment("vector combinations are not supported")
        coefficient = _product(before + [_product(after)])
        return _Term(coefficient, matched, "left")
    if any(_is_matrix(value) for value in before):
        raise UnsupportedMoment("vector combinations are not supported")
    coefficient = _product(after + [_product(before)])
    return _Term(coefficient, matched, "right")


def _default_side(atom: _Atom) -> str:
    """Return the natural coefficient side for `atom`."""
    if atom.orientation == "col":
        return "left"
    if atom.orientation == "row":
        return "right"
    return "scalar"


def _decompose(symbols: ModelSymbols, expr: Any) -> tuple[list[Any], list[_Term]]:
    """Split `expr` into deterministic constants and linear random terms."""
    expanded = sp.expand(expr)
    terms = expanded.args if expanded.is_Add else (expanded,)
    constants: list[Any] = []
    parsed: list[_Term] = []
    for term in terms:
        value = _parse_term(symbols, term)
        if value is None:
            constants.append(term)
        else:
            parsed.append(value)
    if _is_matrix(expr) and len(parsed) > 1:
        raise UnsupportedMoment("vector combinations are not supported")
    return constants, parsed


def _given_group(symbols: ModelSymbols, given: Any) -> Any | None:
    """Return the conditioned group index for a single random-effect vector."""
    if given is None:
        return None
    atom = _match_atom(symbols, given)
    if atom is not None and atom.kind == "effect_vector":
        return symbols.b.require_moment_group(atom.group)
    raise UnsupportedMoment("given must be a single group random-effect vector b[g]")


def _effect_dim(symbols: ModelSymbols) -> int:
    """Return the primary random-effect dimension."""
    term = symbols.model._primary_random_term()
    return 0 if term is None else term.dim


def _conditional_mean(symbols: ModelSymbols, group: Any) -> Any:
    """Return ``X_g beta + Z_g b_g`` as a column vector."""
    return symbols.model.group_mean(group) + symbols.model.random_design(group) * symbols.b[group]


def _atom_expectation(symbols: ModelSymbols, atom: _Atom, given: Any | None) -> Any:
    """Return the expectation of one canonical atom."""
    model = symbols.model
    if atom.kind == "group_row":
        if given is not None and given == atom.group:
            return _conditional_mean(symbols, atom.group).T
        return model.group_mean(atom.group).T
    if atom.kind == "occasion_column":
        if given is not None:
            raise UnsupportedMoment("conditional occasion columns are not supported")
        return model.time_mean(atom.occasion)
    if atom.kind == "entry":
        if given is not None and given == atom.group:
            row = model._entry_row(atom.group, atom.occasion)
            return _conditional_mean(symbols, atom.group)[row, 0]
        return model.entry_mean(atom.group, atom.occasion)
    if atom.kind == "effect_vector":
        if given is not None and given == atom.group:
            value = symbols.b[atom.group]
            return value.T if atom.orientation == "row" else value
        zero = _zero_matrix(_effect_dim(symbols), 1)
        return zero.T if atom.orientation == "row" else zero
    if atom.kind == "effect_entry":
        if given is not None and given == atom.group:
            return symbols.b[atom.group][atom.component, 0]
        return sp.Integer(0)
    raise UnsupportedMoment("moment engine does not support the full response matrix")


def _expectation(symbols: ModelSymbols, expr: Any, *, given: Any = None) -> Any:
    """Return the expectation of a linear expression."""
    given_group = _given_group(symbols, given)
    constants, terms = _decompose(symbols, expr)
    total: Any = None
    for constant in constants:
        total = constant if total is None else total + constant
    for term in terms:
        base = _atom_expectation(symbols, term.atom, given_group)
        if term.side == "left":
            value = base if term.coeff == 1 else term.coeff * base
        elif term.side == "right":
            value = base if term.coeff == 1 else base * term.coeff
        else:
            value = base if term.coeff == 1 else term.coeff * base
        total = value if total is None else total + value
    if total is None:
        return _zero_for_expr(expr)
    return sp.expand(total)


def _group_covariance(symbols: ModelSymbols, group1: Any, group2: Any, given: Any | None) -> Any:
    """Return the covariance block between two group response vectors."""
    model = symbols.model
    if group1 == group2:
        if given is not None and given == group1:
            return model.residual_covariance(group1)
        return model.group_covariance(group1)
    if given is not None and (given == group1 or given == group2):
        return _zero_matrix(model.group_size(group1), model.group_size(group2))
    return model.group_cross_covariance(group1, group2)


def _entry_covariance(
    symbols: ModelSymbols,
    group1: Any,
    occasion1: Any,
    group2: Any,
    occasion2: Any,
    given: Any | None,
) -> Any:
    """Return the covariance between two canonical response entries."""
    model = symbols.model
    if group1 == group2:
        if given is not None and given == group1:
            value = model.residual_covariance(group1)
            return value[
                model._entry_row(group1, occasion1),
                model._entry_row(group1, occasion2),
            ]
        return model.entry_covariance(group1, occasion1, occasion2)
    if given is not None and (given == group1 or given == group2):
        return sp.Integer(0)
    return model.entry_cross_covariance(group1, occasion1, group2, occasion2)


def _atom_covariance(symbols: ModelSymbols, atom1: _Atom, atom2: _Atom, given: Any | None) -> Any:
    """Return the raw covariance block between two canonical atoms."""
    model = symbols.model
    q = _effect_dim(symbols)
    if atom1.kind == "response_matrix" or atom2.kind == "response_matrix":
        raise UnsupportedMoment("moment engine does not support the full response matrix")
    if atom1.kind == "group_row" and atom2.kind == "group_row":
        return _group_covariance(symbols, atom1.group, atom2.group, given)
    if atom1.kind == "occasion_column" and atom2.kind == "occasion_column":
        if given is not None:
            raise UnsupportedMoment("conditional occasion columns are not supported")
        return model.time_covariance(atom1.occasion, atom2.occasion)
    if atom1.kind == "entry" and atom2.kind == "entry":
        return _entry_covariance(
            symbols,
            atom1.group,
            atom1.occasion,
            atom2.group,
            atom2.occasion,
            given,
        )
    if atom1.kind == "effect_vector" and atom2.kind == "effect_vector":
        if atom1.group != atom2.group:
            return _zero_matrix(q, q)
        if given is not None and given == atom1.group:
            return _zero_matrix(q, q)
        return symbols.G
    if atom1.kind == "effect_entry" and atom2.kind == "effect_entry":
        if atom1.group != atom2.group:
            return sp.Integer(0)
        if given is not None and given == atom1.group:
            return sp.Integer(0)
        return symbols.G[atom1.component, atom2.component]
    if atom1.kind == "group_row" and atom2.kind == "effect_vector":
        if atom1.group != atom2.group:
            return _zero_matrix(model.group_size(atom1.group), q)
        if given is not None and given == atom1.group:
            return _zero_matrix(model.group_size(atom1.group), q)
        return model.random_design(atom1.group) * symbols.G
    if atom1.kind == "effect_vector" and atom2.kind == "group_row":
        return _transpose(_atom_covariance(symbols, atom2, atom1, given))
    if atom1.kind == "entry" and atom2.kind == "effect_vector":
        if atom1.group != atom2.group:
            return _zero_matrix(1, q)
        if given is not None and given == atom1.group:
            return _zero_matrix(1, q)
        value = model.random_design(atom1.group) * symbols.G
        return value[model._entry_row(atom1.group, atom1.occasion), :]
    if atom1.kind == "effect_vector" and atom2.kind == "entry":
        return _transpose(_atom_covariance(symbols, atom2, atom1, given))
    if atom1.kind == "group_row" and atom2.kind == "entry":
        value = _group_covariance(symbols, atom1.group, atom2.group, given)
        return value[:, model._entry_row(atom2.group, atom2.occasion)]
    if atom1.kind == "entry" and atom2.kind == "group_row":
        return _transpose(_atom_covariance(symbols, atom2, atom1, given))
    raise UnsupportedMoment("moment engine supports linear expressions only")


def _sandwich(term1: _Term, term2: _Term, base: Any) -> Any:
    """Apply coefficient matrices to a raw atom covariance block."""
    if term1.side == "scalar" and term2.side == "scalar":
        return _chain(term1.coeff, term2.coeff, base)
    if term1.side == "left" and term2.side == "left":
        return _chain(term1.coeff, base, _transpose(term2.coeff))
    if term1.side == "right" and term2.side == "right":
        return _chain(_transpose(term1.coeff), base, term2.coeff)
    if term1.side == "left" and term2.side == "right":
        return _chain(term1.coeff, base, term2.coeff)
    if term1.side == "right" and term2.side == "left":
        return _chain(_transpose(term1.coeff), base, _transpose(term2.coeff))
    if term1.side == "scalar" and term2.side == "left":
        return _chain(term1.coeff, base, _transpose(term2.coeff))
    if term1.side == "left" and term2.side == "scalar":
        return _chain(term1.coeff, base, term2.coeff)
    if term1.side == "scalar" and term2.side == "right":
        return _chain(term1.coeff, base, term2.coeff)
    if term1.side == "right" and term2.side == "scalar":
        return _chain(_transpose(term1.coeff), base, term2.coeff)
    raise UnsupportedMoment("moment engine supports linear expressions only")


def _covariance(symbols: ModelSymbols, expr1: Any, expr2: Any, *, given: Any = None) -> Any:
    """Return the covariance of two linear expressions."""
    given_group = _given_group(symbols, given)
    _, terms1 = _decompose(symbols, expr1)
    _, terms2 = _decompose(symbols, expr2)
    total: Any = None
    for term1 in terms1:
        for term2 in terms2:
            base = _atom_covariance(symbols, term1.atom, term2.atom, given_group)
            value = _sandwich(term1, term2, base)
            total = value if total is None else total + value
    if total is None:
        result = _zero_covariance(expr1, expr2)
    else:
        result = sp.expand(total)
    if not _is_matrix(expr1) and not _is_matrix(expr2):
        result = _scalarize(result)
    return result
