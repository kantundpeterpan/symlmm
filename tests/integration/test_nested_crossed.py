"""Nested and crossed grouping-factor label tests."""

import sympy as sp

from symlmm.covariances import IID, Scalar
from symlmm.model import LMM
from symlmm.specs import Crossed, Fixed, Nested, RandomIntercept


def test_nested_and_crossed_labels_feed_sum_form():
    b0 = sp.Symbol("b0")
    tau_v = sp.Symbol("tau_v", positive=True)
    tau_x = sp.Symbol("tau_x", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomIntercept(group="visit", cov=Scalar(tau_v)),
            RandomIntercept(group="item", cov=Scalar(tau_x)),
        ],
        residual=IID(var=sigma2),
        groups=(Nested("visit", within="subject"), Crossed("subject", "item")),
    )
    i, j = model.indices()
    assert model.group(i, "visit").base == sp.IndexedBase("subject:visit", shape=(model.n,))
    assert model.group(i, "item").base == sp.IndexedBase("subject:item", shape=(model.n,))

    entry = model.covariance_entry(i, j)
    labels = {indexed.base.name for indexed in entry.atoms(sp.Indexed)}
    assert "subject:visit" in labels
    assert "subject:item" in labels
    expected = (
        tau_v * sp.KroneckerDelta(model.group(i, "visit"), model.group(j, "visit"))
        + tau_x * sp.KroneckerDelta(model.group(i, "item"), model.group(j, "item"))
        + sigma2 * sp.KroneckerDelta(i, j)
    )
    assert sp.simplify(entry - expected) == 0
