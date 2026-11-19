import pytest
import sympy as sp

from symlmm.blocks import (
    Block,
    block_coefficients,
    block_cross_residual,
    block_mean_symbol,
    block_random_design,
    block_residual,
    block_size_symbol,
    block_term_design,
)
from symlmm.covariances import IID, Diagonal, Scalar, Unstructured
from symlmm.decomposition import terms
from symlmm.errors import ModelError, NotDefined
from symlmm.model import LMM
from symlmm.specs import (
    Covariate,
    Fixed,
    Group,
    Nested,
    RandomEffects,
    RandomSlope,
)


def _hierarchical():
    b0 = sp.Symbol("b0")
    t2 = sp.Symbol("t2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="F", intercept=True, cov=Scalar(t2)),
            RandomEffects(group=Nested("H", "G"), intercept=True, cov=Scalar(t2)),
        ],
        residual=IID(var=s2),
        groups=[Group("F"), Nested("G", "F")],
        n=4,
    )


def test_block_dense_covariance_matches_submatrix():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    block = Block(dense, dense.factor("H"), 0)
    rows = block.rows()
    assert rows == [0, 1]
    assert block.covariance() == dense.matrix_covariance()[rows, :][:, rows]


def test_block_cross_covariance_matches_submatrix():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 0, 0], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    left = Block(dense, dense.factor("H"), 0)
    right = Block(dense, dense.factor("H"), 1)
    assert (
        left.cross_covariance(right) == dense.matrix_covariance()[left.rows(), :][:, right.rows()]
    )
    assert left.cross_covariance(right) != sp.zeros(2, 2)


def test_block_mean_and_design():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    block = Block(dense, dense.factor("H"), 0)
    assert block.mean() == dense.matrix_mean()[block.rows(), :]
    assert block.design() == dense.matrix_x()[block.rows(), :]


def test_block_dense_random_design_residual_and_effect():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    block = Block(dense, dense.factor("H"), 0)
    rows = block.rows()
    assert block.size() == 2
    assert block.random_design() == dense.matrix_z()[rows, :]
    assert block.residual() == dense.matrix_r()[rows, :][:, rows]
    assert block.random_effect() == block_coefficients(dense.factor("H"), 0, 1)


def test_symbolic_block_covariance_shape():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    assert block.covariance().shape == (block.size(), block.size())


def test_symbolic_block_size_mean_and_design():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    factor = model.factor("H")
    block = Block(model, factor, level)
    p = len(model.fixed.coefficients())
    assert block.size() == block_size_symbol(factor, level)
    assert block.mean() == block_mean_symbol(factor, level, p) * sp.Matrix(
        model.fixed.coefficients()
    )
    assert block.design() == block_mean_symbol(factor, level, p)


def test_symbolic_random_design_and_residual():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    factor = model.factor("H")
    block = Block(model, factor, level)
    q = model.dimensions()[1]
    assert block.random_design() == block_random_design(factor, level, q)
    assert block.residual() == block_residual(factor, level)


def test_symbolic_random_effect_and_missing_term():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    assert block.random_effect() == block_coefficients(model.factor("H"), level, 1)
    with pytest.raises(NotDefined, match="no random-effect"):
        Block(model, model.factor("G"), level).random_effect()


def test_symbolic_cross_covariance_uses_term_designs():
    model = _hierarchical()
    left_factor = model.factor("H")
    right_factor = model.factor("F")
    left_level = sp.Symbol("h", integer=True)
    right_level = sp.Symbol("f", integer=True)
    left = Block(model, left_factor, left_level)
    right = Block(model, right_factor, right_level)
    expected = block_cross_residual(left_factor, left_level, right_factor, right_level)
    for term in terms(model):
        expected += (
            block_term_design(left_factor, left_level, term)
            * term.random.cov.dense()
            * block_term_design(right_factor, right_level, term).T
        )
    assert left.cross_covariance(right) == expected


def test_cross_covariance_requires_the_same_model():
    columns = {"F": [0, 0, 1, 1], "G": [0, 0, 0, 0], "H": [0, 0, 1, 1]}
    left_model = _hierarchical().dense(**columns)
    right_model = _hierarchical().dense(**columns)
    left = Block(left_model, left_model.factor("H"), 0)
    right = Block(right_model, right_model.factor("H"), 1)
    with pytest.raises(ModelError, match="same model"):
        left.cross_covariance(right)


def test_block_rows_require_dense_mode():
    model = _hierarchical()
    block = Block(model, model.factor("H"), 0)
    with pytest.raises(NotDefined, match="dense"):
        block.rows()


def test_block_covariance_terms_sum_to_total():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    terms = block.covariance_terms()
    zero = sp.zeros(block.size(), block.size())
    assert sum(terms.values(), zero) == block.covariance()
    assert list(terms)[-1] == "residual"


def test_block_partial_covariance_excludes_term():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    terms = block.covariance_terms()
    excluded = next(iter(terms))
    assert block.partial_covariance(block, exclude=[excluded]) == (
        block.covariance() - terms[excluded]
    )


def test_block_partial_covariance_excludes_everything():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    exclude = tuple(block.covariance_terms())
    assert block.partial_covariance(block, exclude=exclude) == sp.zeros(block.size(), block.size())


def test_block_unknown_exclusion_raises():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    with pytest.raises(ModelError, match="unknown term"):
        block.partial_covariance(block, exclude=["missing"])


def test_block_partial_covariance_distinct_blocks_excludes_everything():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 0, 0], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    left = Block(dense, dense.factor("H"), 0)
    right = Block(dense, dense.factor("H"), 1)
    assert left.rows() != right.rows()
    exclude = tuple(left.covariance_terms())
    assert left.partial_covariance(right, exclude=exclude) == sp.zeros(2, 2)


def test_symbolic_covariance_terms_sum_to_total():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    values = block.covariance_terms()
    zero = sp.ZeroMatrix(block.size(), block.size())
    assert sum(values.values(), zero) == block.covariance()
    assert list(values)[-1] == "residual"


def test_symbolic_partial_covariance_excludes_everything():
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    exclude = tuple(block.covariance_terms())
    assert block.partial_covariance(block, exclude=exclude) == sp.ZeroMatrix(
        block.size(), block.size()
    )


def test_random_design_rejects_unknown_terms():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 1, 0, 1])
    block = Block(dense, dense.factor("H"), 0)
    with pytest.raises(ModelError, match="unknown term"):
        block.random_design(term="missing")
    with pytest.raises(ModelError, match="unknown term"):
        block.random_design(term=terms(model)[0])


def _two_factor():
    b0 = sp.Symbol("b0")
    f2 = sp.Symbol("f2", positive=True)
    g2 = sp.Symbol("g2", positive=True)
    s2 = sp.Symbol("s2", positive=True)
    return LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(group="F", intercept=True, cov=Scalar(f2)),
            RandomEffects(group="G", intercept=True, cov=Scalar(g2)),
        ],
        residual=IID(var=s2),
        n=6,
    )


def test_cross_block_covariance_terms_sum_to_cross_covariance():
    model = _two_factor()
    dense = model.dense(
        F=[0, 0, 0, 1, 1, 1],
        G=[0, 1, 2, 0, 1, 2],
    )
    left = Block(dense, dense.factor("F"), 0)
    right = Block(dense, dense.factor("G"), 0)
    assert left.rows() != right.rows()
    values = left.covariance_terms(right)
    assert list(values)[-1] == "residual"
    zero = sp.zeros(left.size(), right.size())
    assert sum(values.values(), zero) == left.cross_covariance(right)
    assert values["residual"] == left.cross_residual(right)


def test_model_covariance_terms_dispatches_cross_block_terms():
    model = _two_factor()
    dense = model.dense(
        F=[0, 0, 0, 1, 1, 1],
        G=[0, 1, 2, 0, 1, 2],
    )
    left = Block(dense, dense.factor("F"), 0)
    right = Block(dense, dense.factor("G"), 0)
    assert dense.covariance_terms(left, right) == left.covariance_terms(right)


def test_symbolic_cross_block_covariance_terms_sum_to_total():
    model = _two_factor()
    f_level = sp.Symbol("f", integer=True)
    g_level = sp.Symbol("g", integer=True)
    left = Block(model, model.factor("F"), f_level)
    right = Block(model, model.factor("G"), g_level)
    values = left.covariance_terms(right)
    zero = sp.ZeroMatrix(left.size(), right.size())
    assert sum(values.values(), zero) == left.cross_covariance(right)
    assert values["residual"] == block_cross_residual(
        model.factor("F"), f_level, model.factor("G"), g_level
    )


def _ri_rs_symbolic_model():
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


def test_block_response_is_symbolic():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    Y = block.response()

    assert Y.shape == (block.size(), 1)
    assert Y.name == "Y_subject:i"


def test_block_term_resolves_single_term():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    assert block.term().name == "subject"
    assert block.term("subject").name == "subject"
    assert block.random_effect("subject") == block_coefficients(model.factor("subject"), i, 2)


def test_block_term_unknown_name_raises():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    with pytest.raises(ModelError, match="unknown random-effect term"):
        block.term("missing")


def test_block_design_columns_for_ri_and_rs():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    i = sp.Symbol("i", integer=True)
    ri_model = LMM(
        fixed=Fixed(),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                cov=Scalar(tau2),
            )
        ],
        residual=IID(var=sigma2),
    )
    ri_block = ri_model.block("subject", i)
    ri_columns = ri_block.design_columns("subject")

    assert len(ri_columns) == 1
    assert ri_columns[0] == sp.OneMatrix(ri_block.size(), 1)

    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)
    columns = block.design_columns("subject")

    assert len(columns) == 2
    assert columns[0] == sp.OneMatrix(block.size(), 1)
    assert columns[1].shape == (block.size(), 1)
    assert columns[1] != columns[0]


@pytest.mark.parametrize(
    "method",
    ["precision", "shrinkage", "blup_covariance", "blup_mean"],
)
def test_block_structured_queries_reject_multiple_random_terms(method):
    model = _hierarchical()
    level = sp.Symbol("h", integer=True)
    block = Block(model, model.factor("H"), level)
    args = () if method == "precision" else (block.term().name,)

    with pytest.raises(NotDefined, match="single random-effect term"):
        getattr(block, method)(*args)


@pytest.mark.parametrize("within", [True, False])
def test_dense_design_columns_and_structured_design_match_matrix_z(within):
    b0 = sp.Symbol("b0")
    tau0, tau1, sigma2 = sp.symbols("tau0 tau1 sigma2", positive=True)
    model = LMM(
        fixed=Fixed(intercept=b0),
        random=[
            RandomEffects(
                group=Group("subject"),
                intercept=True,
                slopes=(RandomSlope("time", within=within),),
                cov=Diagonal([tau0, tau1]),
            )
        ],
        residual=IID(var=sigma2),
        coordinate="time",
    )
    dense = model.dense(subject=[0, 0, 1, 1], time=[0, 1, 0, 2])
    block = dense.block("subject", 1)
    term = block.term("subject")
    rows = block.rows()
    labels = sorted(set(dense._dense_group_codes(term.factor)))
    start = dense._random_term_offset(term.random) + labels.index(block.level) * term.random.dim
    expected = dense.matrix_z()[rows, start : start + term.random.dim]

    columns = block.design_columns("subject")

    assert block.structured_random_design("subject").as_explicit() == expected
    for index, column in enumerate(columns):
        assert sp.Matrix(column) == expected[:, index]

    # Locks the within-resolution mirrored from RandomEffects.z in specs.py.
    term_design = sp.Matrix.vstack(*(term.random.z(dense, row).T for row in rows))
    assert sp.Matrix.hstack(*(sp.Matrix(column) for column in columns)) == term_design


def test_block_precision_matches_dense_inverse():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 1, 1, 1],
        time=[0, 1, 2, 0, 1, 2],
    )
    block = dense.block("subject", 0)
    expected = dense.matrix_covariance()[0:3, 0:3].inv()

    assert (block.precision() - expected).equals(sp.zeros(3, 3))


def test_block_precision_has_no_n_by_n_inverse_symbol():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    precision = block.precision()

    assert precision.shape == (block.size(), block.size())
    assert not any(
        isinstance(symbol, sp.Inverse) and symbol.shape == (block.size(), block.size())
        for symbol in precision.atoms(sp.Inverse)
    )


def test_block_shrinkage_uses_structured_design():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    Z = block.structured_random_design("subject")
    S = block.shrinkage("subject")

    assert Z.shape == (block.size(), 2)
    assert Z[0, 0] == 1
    assert S.shape == (2, block.size())


def test_block_shrinkage_ri_closed_form():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    i = sp.Symbol("i", integer=True)
    model = LMM(
        fixed=Fixed(),
        random=[
            RandomEffects(
                group="subject",
                intercept=True,
                cov=Scalar(tau2),
            )
        ],
        residual=IID(var=sigma2),
    )
    block = model.block("subject", i)

    S = block.shrinkage("subject")
    expected = (tau2 / (sigma2 + block.size() * tau2)) * sp.OneMatrix(1, block.size())

    assert S.shape == (1, block.size())
    assert sp.simplify((S - expected)[0, 0]) == 0
    assert not any(
        isinstance(symbol, sp.Inverse) and symbol.shape == (block.size(), block.size())
        for symbol in S.atoms(sp.Inverse)
    )


def test_block_dense_shrinkage_matches_global_moments():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 0)
    rows = block.rows()
    term = block.term("subject")
    start = dense._random_term_offset(term.random)
    columns = slice(start, start + term.random.dim)

    G = dense.matrix_g()[columns, columns]
    Z = dense.matrix_z()[rows, columns]
    V = dense.matrix_covariance()[rows, :][:, rows]
    expected = G * Z.T * V.inv()

    assert (block.shrinkage("subject") - expected).equals(sp.zeros(2, block.size()))


def test_dense_block_random_design_uses_term_level_columns():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 1)
    rows = block.rows()
    Z = dense.matrix_z()
    term = block.term("subject")
    start = dense._random_term_offset(term.random) + term.random.dim

    actual = block.random_design(term="subject")

    assert rows == [4, 5, 6, 7]
    assert actual == Z[rows, start : start + term.random.dim]
    assert actual != sp.zeros(block.size(), term.random.dim)
    assert actual != Z[rows, : term.random.dim]


def test_dense_block_random_design_uses_term_level_columns_with_reversed_rows():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[1, 1, 1, 1, 0, 0, 0, 0],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 1)
    rows = block.rows()
    Z = dense.matrix_z()
    term = block.term("subject")
    start = dense._random_term_offset(term.random) + term.random.dim

    assert rows == [0, 1, 2, 3]
    assert block.random_design(term="subject") == Z[rows, start : start + term.random.dim]


def test_dense_block_random_design_uses_full_layout_for_other_factor_terms():
    model = _hierarchical()
    dense = model.dense(F=[0, 0, 1, 1], G=[0, 0, 0, 0], H=[0, 0, 1, 1])
    block = dense.block("H", 1)
    rows = block.rows()
    Z = dense.matrix_z()
    term = next(item for item in terms(dense) if item.name == "F")
    start = dense._random_term_offset(term.random)
    stop = start + dense.levels(term.factor.name) * term.random.dim

    actual = block.random_design(term="F")

    assert rows == [2, 3]
    assert actual == Z[rows, start:stop]
    assert actual != sp.zeros(block.size(), term.random.dim)
    assert actual != Z[rows, start : start + term.random.dim]


def test_block_dense_shrinkage_level_one_matches_global_moments():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 1)
    rows = block.rows()
    term = block.term("subject")
    start = dense._random_term_offset(term.random) + term.random.dim
    columns = slice(start, start + term.random.dim)

    G = dense.matrix_g()[columns, columns]
    Z = dense.matrix_z()[rows, columns]
    V = dense.matrix_covariance()[rows, :][:, rows]
    expected = G * Z.T * V.inv()

    assert (block.shrinkage("subject") - expected).equals(sp.zeros(2, block.size()))


def test_block_blup_has_symbolic_shape():
    model, i = _ri_rs_symbolic_model()
    block = model.block("subject", i)

    b_hat = block.blup_mean("subject")

    assert b_hat.shape == (2, 1)
    assert block.response() in b_hat.free_symbols


def test_block_blup_matches_global_dense():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 0)
    y = sp.MatrixSymbol("y", dense.n, 1)

    actual = block.blup_mean("subject").subs({block.response(): y[0:4, :]})
    expected = dense.blup_mean(form="matrix")[0:2, :]

    assert (actual - expected).equals(sp.zeros(2, 1))


def test_block_blup_covariance_matches_global_dense():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 0)

    actual = block.blup_covariance("subject")
    expected = dense.blup_covariance(form="matrix")[0:2, 0:2]

    assert (actual - expected).equals(sp.zeros(2, 2))


def test_block_blup_mean_level_one_matches_global_dense():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 1)
    y = sp.MatrixSymbol("y", dense.n, 1)

    actual = block.blup_mean("subject").subs({block.response(): y[4:8, :]})
    expected = dense.blup_mean(form="matrix")[2:4, :]

    assert (actual - expected).equals(sp.zeros(2, 1))


def test_block_blup_covariance_level_one_matches_global_dense():
    model, _ = _ri_rs_symbolic_model()
    dense = model.dense(
        subject=[0, 0, 0, 0, 1, 1, 1, 1],
        time=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    block = dense.block("subject", 1)

    actual = block.blup_covariance("subject")
    expected = dense.blup_covariance(form="matrix")[2:4, 2:4]

    assert (actual - expected).equals(sp.zeros(2, 2))


def test_block_blup_mean_without_fixed_effects():
    tau2 = sp.Symbol("tau2", positive=True)
    sigma2 = sp.Symbol("sigma2", positive=True)
    model = LMM(
        fixed=Fixed(),
        random=[RandomEffects(group="subject", intercept=True, cov=Scalar(tau2))],
        residual=IID(var=sigma2),
        coordinate="time",
    )
    g = sp.Symbol("g", integer=True)
    block = model.block("subject", g)

    b_hat = block.blup_mean("subject")

    assert b_hat.shape == (1, 1)
    assert block.response() in b_hat.free_symbols
