"""Multiple random-term assembly tests."""

import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Fixed, RandomEffects


def test_two_random_terms_sum():
    b0 = sp.Symbol("b0")
    tau_a = sp.Symbol("tau_a", positive=True)
    tau_b = sp.Symbol("tau_b", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="subject", intercept=True, cov=Scalar(tau_a)),
            RandomEffects(group="item", intercept=True, cov=Scalar(tau_b)),
        ],
        residual=IID(var=sigma2),
    )
    i, j = model.indices()
    expected = (
        tau_a * sp.KroneckerDelta(model.group(i, "subject"), model.group(j, "subject"))
        + tau_b * sp.KroneckerDelta(model.group(i, "item"), model.group(j, "item"))
        + sigma2 * sp.KroneckerDelta(i, j)
    )
    assert sp.simplify(model.covariance_entry(i, j) - expected) == 0
