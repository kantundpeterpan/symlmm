import sympy as sp

from symlmm.blocks import Block
from symlmm.covariances import IID, Unstructured
from symlmm.model import LMM
from symlmm.specs import Covariate, Fixed, Group, RandomEffects, RandomSlope


def _ri_rs_model():
    b0, b1 = sp.symbols("b0 b1", real=True)
    tau0, tau1, sigma2 = sp.symbols("tau0 tau1 sigma2", positive=True)
    rho_b = sp.Symbol("rho_b", real=True)
    i = sp.Symbol("i", integer=True)

    subject = Group("subject", index=i)
    time = Covariate("time")

    return LMM(
        fixed=Fixed(intercept=b0, slopes={time: b1}),
        random=[
            RandomEffects(
                group=subject,
                intercept=True,
                slopes=(RandomSlope(time),),
                cov=Unstructured.from_sd_corr([tau0, tau1], rho_b),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    ), i


def test_symbolic_block_moments_do_not_access_rows(monkeypatch):
    model, i = _ri_rs_model()
    block = model.block("subject", i)

    def fail(self):
        raise AssertionError("Block.rows() called from symbolic path")

    monkeypatch.setattr(Block, "rows", fail)

    assert block.shrinkage("subject").shape == (2, block.size())
    assert block.blup_mean("subject").shape == (2, 1)
    assert block.blup_covariance("subject").shape == (2, 2)
