# Blackboard-Style Symbolic Moments Design

**Status:** proposed
**Scope:** a pedagogical facade over the LMM moment engine. It returns model
symbols and supports blackboard-style `E`, `Var`, and `Cov` manipulation,
including the law of total variance.

## Goal

Let a user define a model once and work with its moments the way they would on a
blackboard:

```python
model = LMM(
    fixed=Fixed(intercept=b0, slopes={"time": b1}),
    random=[RandomEffects(
        group="subject",
        intercept=True,
        slopes=(RandomSlope("time"),),
        cov=G,
    )],
    residual=IID(var=sigma2),
)

syms = model.symbols()

syms.Y[i]                       # Y_i
syms.eps[i]                     # epsilon_i
syms.b["subject"][level, a]     # b_{subject, level, a}
syms.beta                       # fixed coefficients
syms.X(i)                       # fixed design row
syms.Z(i)                       # random design row

syms.E(syms.Y[i])
syms.Var(syms.Y[i])
syms.Cov(syms.Y[i], syms.Y[j])
syms.Var(syms.Y[i], given=syms.b)
syms.E(syms.Var(syms.Y[i], given=syms.b))
syms.Var(syms.E(syms.Y[i], given=syms.b))
syms.total_variance(syms.Y[i], given=syms.b)
```

The facade returns ordinary SymPy expressions. It is not a new random-variable
distribution system.

## Relationship to sympy.stats

`sympy.stats` is not suitable for symbolic indexed random effects:

- `RandomSymbol` is not subscriptable.
- `Normal(Indexed("b", g), ...)` rejects indexed names.
- `MultivariateNormal` components work for concrete indices, but
  `covariance(B[0], B[1])` is unevaluated and `B[g]` with symbolic `g` raises
  `NotImplementedError`.

The blackboard facade therefore uses SymPy expressions plus a small moment
engine. `sympy.stats` remains useful for concrete teaching examples and matrix
derivations, not as the blackboard substrate.

## Model symbols

`LMM.symbols()` returns `ModelSymbols`:

| Symbol | Meaning |
|---|---|
| `Y[i]` | observation response |
| `eps[i]` | residual |
| `b[group][level, coefficient]` | random effect |
| `beta` | fixed-effect coefficient vector |
| `X(i)` | fixed design row |
| `Z(i)` | random design row |
| `G` / `R` | symbolic random/residual covariance matrices |
| `i`, `j` | observation indices |

`b` is a `RandomEffectCollection`; passing `given=syms.b` means conditioning on
all random effects. The collection is also indexable by group.

The conditional-mean symbol for a random block changes from
`b_{group}_{term_index}` to `b_{group}` because duplicate blocks for one group
are rejected. This makes the blackboard symbol the same as the engine symbol.

## Moment engine

The engine supports **linear expressions in model atoms**.

### Atoms

- `Y[i]`: response.
- `eps[i]`: residual.
- `b[group][level, a]`: random effect.
- Deterministic symbols (`beta`, `x`, design entries) are constants.

### Expectation

- `E(Y[i]) = model.mean(i)`
- `E(eps[i]) = 0`
- `E(b[group][level, a]) = 0`
- `E(constant) = constant`
- `E(a*X + b*Y) = a*E(X) + b*E(Y)`

### Covariance

- `Cov(Y[i], Y[j]) = model.covariance_entry(i, j)`
- `Cov(eps[i], eps[j]) = model.residual.entry(i, j, model)`
- `Cov(b[group][l, a], b[group][l, b]) = G[a, b]`
- different groups or levels have zero covariance
- `Cov(Y[i], eps[j]) = R[i, j]`
- `Cov(Y[i], b[group][l, a]) = (G z_i)[a]` when `l == group(i)`, else zero
- `Cov(b, eps) = 0`
- `Var(X) = Cov(X, X)`

Covariance is bilinear over linear combinations:

```text
Cov(sum a_k A_k, sum b_l B_l) = sum_kl a_k b_l Cov(A_k, B_l)
```

### Conditional moments

`given=syms.b` conditions on all random effects:

- `E(Y[i] | b) = model.conditional_mean(i, form="sum")`
- `Cov(Y[i], Y[j] | b) = model.conditional_covariance(i, j, form="sum")`
- `E(b | b) = b`
- `Cov(b, anything | b) = 0`
- residual and fixed-effect rules are unchanged.

### Law of total variance

```python
syms.total_variance(expr, given=syms.b)
# Var(E(expr | b)) + E(Var(expr | b))
```

For `Y[i]` this produces the familiar decomposition:

```text
Var(Y_i) = [Z G Z']_ii + R_ii
```

Total covariance is analogous:

```python
syms.total_covariance(expr1, expr2, given=syms.b)
# Cov(E(expr1 | b), E(expr2 | b)) + E(Cov(expr1, expr2 | b))
```

## Supported scope

- Linear expressions in `Y`, `eps`, and `b`.
- Linear combinations, scalar multiplication, addition, subtraction.
- Conditional expectation/covariance given all random effects.
- LMM first; marginal and GEE can implement the same protocol later.

Not supported in the first version:

- nonlinear expressions such as `Y[i]**2` or `Y[i]*Y[j]`;
- conditioning on individual random effects;
- non-Gaussian random-effect distributions;
- distribution density or sampling;
- automatic simplification of arbitrary SymPy expressions.

Unsupported expressions raise `UnsupportedMoment` from `symlmm.errors`.

## Testing

- symbols are model-consistent and indexed as expected;
- `E`, `Var`, `Cov` for atoms and linear combinations;
- random-effect covariance and cross-covariance with response/residual;
- multiple groups and joint intercept/slope blocks;
- conditional moments and total variance;
- total covariance;
- nonlinear expression rejection;
- comparison against `model.mean`, `model.covariance`, and dense `Z G Z' + R`.

## Risks

- The facade must not duplicate model math. It should delegate to the existing
  moment engine and covariance structures.
- Linear-only scope must be explicit; nonlinear use should fail loudly, not
  return a wrong expression.
- Changing the conditional-mean symbol to `b_{group}` is a symbolic breaking
  change; tests and docs must update.
- Users may expect `sympy.stats` semantics; documentation must state the
  distinction.
