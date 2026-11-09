"""Manual exploration of the SymPy capabilities required by symlmm."""

import sympy as sp
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix
from sympy.matrices.expressions.diagonal import DiagonalOf


def main() -> None:
    n, q = sp.symbols("n q", positive=True, integer=True)
    Z = sp.MatrixSymbol("Z", n, q)
    G = sp.MatrixSymbol("G", q, q)
    V = Z * G * Z.T
    print("MatMul entry:", V[0, 1], "| has Sum:", V[0, 1].has(sp.Sum))

    A = sp.MatrixSymbol("A", 2, 2)
    B = sp.MatrixSymbol("B", 3, 3)
    print("BlockDiag shape:", BlockDiagMatrix(A, B).shape)

    x = sp.MatrixSymbol("x", n, 1)
    print("Elementwise:", ElementwiseApplyFunction(sp.exp, x).shape)

    print("DiagonalOf:", DiagonalOf(V).shape, "| entry:", DiagonalOf(V)[0, 0])

    i, j = sp.symbols("i j", integer=True)
    tau2 = sp.Symbol("tau2", positive=True)
    expr = sp.Sum(tau2 * sp.KroneckerDelta(i, j), (i, 0, 3)).doit()
    print("Delta sum:", expr)
    print("Delta sum at j=1:", sp.simplify(expr.subs(j, 1)))


if __name__ == "__main__":
    main()
