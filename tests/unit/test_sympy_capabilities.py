"""Spike tests for the SymPy capabilities symlmm depends on."""

import pytest
import sympy as sp
from sympy.core.sympify import SympifyError
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix
from sympy.matrices.expressions.diagonal import DiagonalOf


def test_matmul_entry_expands_to_sum() -> None:
    n, q = sp.symbols("n q", positive=True, integer=True)
    Z = sp.MatrixSymbol("Z", n, q)
    G = sp.MatrixSymbol("G", q, q)
    V = Z * G * Z.T

    assert V[0, 1].has(sp.Sum)


def test_blockdiag_accepts_varargs_but_not_lists() -> None:
    A = sp.MatrixSymbol("A", 2, 2)
    B = sp.MatrixSymbol("B", 3, 3)

    block = BlockDiagMatrix(A, B)
    assert block.shape == (5, 5)
    assert block[0, 0] == A[0, 0]
    assert block[2, 2] == B[0, 0]

    with pytest.raises(SympifyError):
        BlockDiagMatrix([A, B])


def test_elementwise_apply_function() -> None:
    n = sp.Symbol("n", positive=True, integer=True)
    x = sp.MatrixSymbol("x", n, 1)
    f = ElementwiseApplyFunction(sp.exp, x)

    assert f.shape == (n, 1)
    assert f[0, 0] == sp.exp(x[0, 0])


def test_diagonal_of_is_a_column_vector() -> None:
    n = sp.Symbol("n", positive=True, integer=True)
    V = sp.MatrixSymbol("V", n, n)
    diagonal = DiagonalOf(V)

    assert diagonal.shape == (n, 1)
    assert diagonal[0, 0] == V[0, 0]


def test_kronecker_delta_sum_collapses_after_index_substitution() -> None:
    i, j = sp.symbols("i j", integer=True)
    tau2 = sp.Symbol("tau2", positive=True)
    expr = sp.Sum(tau2 * sp.KroneckerDelta(i, j), (i, 0, 3)).doit()

    assert sp.simplify(expr.subs(j, 1)) == tau2


def test_symbolic_matrix_dimensions() -> None:
    n = sp.Symbol("n", positive=True, integer=True)
    A = sp.MatrixSymbol("A", n, n)

    assert A.shape == (n, n)
