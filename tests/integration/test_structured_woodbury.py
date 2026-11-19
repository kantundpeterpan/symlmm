import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, Group, RandomEffects


def test_ri_iid_shrinkage_closed_form_without_dense():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    i = sp.Symbol("i", integer=True)
    j = sp.Symbol("j", integer=True)

    model = LMM(
        fixed=Fixed(),
        random=[
            RandomEffects(
                group=Group("subject"),
                intercept=True,
                cov=Scalar(tau2),
            )
        ],
        residual=IID(var=sigma2),
    )
    block = model.block("subject", i)
    S = block.shrinkage("subject")
    expected = tau2 / (sigma2 + block.size() * tau2)

    assert S.shape == (1, block.size())
    assert S.has(sp.OneMatrix(1, block.size()))
    assert sp.simplify(S[0, j] - expected) == 0

    precision = block.precision()
    assert precision.shape == (block.size(), block.size())
    assert not any(
        isinstance(symbol, sp.Inverse) and symbol.shape == (block.size(), block.size())
        for symbol in precision.atoms(sp.Inverse)
    )
