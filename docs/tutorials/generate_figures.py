"""Generate figures for the factor-first tutorial.

Run from the repository root:

    uv run --with matplotlib python docs/tutorials/generate_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import symlmm as sl

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def numeric(expr, values):
    """Return a float for a scalar SymPy expression."""
    return float(sp.N(expr.subs(values)))


def numeric_matrix(expr, values):
    """Return a NumPy matrix for a SymPy matrix expression."""
    matrix = expr.subs(values)
    return np.array(matrix.tolist(), dtype=float)


# ---------------------------------------------------------------------------
# Hierarchical model: schools -> classes -> students
# ---------------------------------------------------------------------------

b0 = sp.Symbol("b0")
s2_school, s2_class, s2_student, s2 = sp.symbols(
    "s2_school s2_class s2_student s2", positive=True
)

school = sl.Group("school")
klass = sl.Nested("class", "school")
student = sl.Nested("student", "class")

hier = sl.LMM(
    fixed=sl.Fixed(intercept=b0),
    random=[
        sl.RandomEffects(group=school, intercept=True, cov=sl.Scalar(s2_school)),
        sl.RandomEffects(group=klass, intercept=True, cov=sl.Scalar(s2_class)),
        sl.RandomEffects(group=student, intercept=True, cov=sl.Scalar(s2_student)),
    ],
    residual=sl.IID(var=s2),
    groups=[school, klass, student],
    n=8,
)

hier_dense = hier.dense(
    school=[0, 0, 0, 0, 1, 1, 1, 1],
    **{"class": [0, 0, 1, 1, 0, 0, 1, 1]},
    student=[0, 1, 2, 3, 4, 5, 6, 7],
)

hier_values = {
    b0: 0.0,
    s2_school: 0.6,
    s2_class: 0.3,
    s2_student: 0.8,
    s2: 0.4,
}

hier_cov = numeric_matrix(hier_dense.matrix_covariance(), hier_values)
hier_labels = [
    "S0 C0 U0", "S0 C0 U1", "S0 C1 U2", "S0 C1 U3",
    "S1 C0 U4", "S1 C0 U5", "S1 C1 U6", "S1 C1 U7",
]

fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(hier_cov, cmap="viridis")
ax.set_title("Hierarchical covariance matrix")
ax.set_xticks(range(8), hier_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(8), hier_labels, fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8, label="Covariance")
fig.tight_layout()
fig.savefig(OUT / "hierarchical_covariance.png", dpi=160)
plt.close(fig)

# Variance-component decomposition for four representative pairs.
pairs = [
    ("same student", (0, 0)),
    ("same class", (0, 1)),
    ("same school\nother class", (0, 2)),
    ("other school", (0, 4)),
]
component_order = ["school", "school:class", "school:class:student", "residual"]
component_labels = ["school", "class", "student", "residual"]
colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

contributions = {label: [] for label in component_labels}
for _, (i, j) in pairs:
    terms = hier_dense.covariance_terms(i, j)
    for key, label in zip(component_order, component_labels, strict=True):
        contributions[label].append(numeric(terms.get(key, 0), hier_values))

fig, ax = plt.subplots(figsize=(8.0, 4.6))
x = np.arange(len(pairs))
bottom = np.zeros(len(pairs))
for label, color in zip(component_labels, colors, strict=True):
    values = np.array(contributions[label])
    ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.62)
    bottom += values
ax.set_xticks(x, [label for label, _ in pairs])
ax.set_ylabel("Covariance contribution")
ax.set_title("Hierarchical covariance by variance component")
ax.legend(frameon=False, ncol=2)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "hierarchical_components.png", dpi=160)
plt.close(fig)

print("hierarchical covariance:\n", hier_cov.round(3))
print("hierarchical terms (0, 1):", hier_dense.covariance_terms(0, 1))


# ---------------------------------------------------------------------------
# Longitudinal model: subjects observed over time
# ---------------------------------------------------------------------------

beta0, beta1, beta2 = sp.symbols("beta0 beta1 beta2")
tau0, tau1, sigma2, rho = sp.symbols("tau0 tau1 sigma2 rho", positive=True)

subject = sl.Group("subject")
time = sl.Covariate("time")
visit = sl.Position("time")
group = sl.Covariate("group")

long = sl.LMM(
    fixed=sl.Fixed(intercept=beta0, slopes={time: beta1, group: beta2}),
    random=[
        sl.RandomEffects(
            group=subject,
            intercept=True,
            slopes=(sl.RandomSlope(time),),
            cov=sl.Diagonal([tau0, tau1]),
        )
    ],
    residual=sl.AR1(group=subject, coordinate=time, var=sigma2, rho=rho),
    coordinate="time",
    n=8,
)

long_dense = long.dense(
    subject=[0, 0, 0, 0, 1, 1, 1, 1],
    time=[0, 1, 2, 3, 0, 1, 2, 3],
    group=[0, 0, 0, 0, 1, 1, 1, 1],
)
panel = long_dense.panel(subject, visit)

long_values = {
    beta0: 2.0,
    beta1: 0.5,
    beta2: 1.0,
    tau0: 1.0,
    tau1: 0.25,
    sigma2: 0.5,
    rho: 0.7,
}

long_cov = numeric_matrix(long_dense.matrix_covariance(), long_values)
long_labels = [f"S0 T{t}" for t in range(4)] + [f"S1 T{t}" for t in range(4)]

fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(long_cov, cmap="magma")
ax.set_title("Longitudinal covariance matrix")
ax.set_xticks(range(8), long_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(8), long_labels, fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8, label="Covariance")
fig.tight_layout()
fig.savefig(OUT / "longitudinal_covariance.png", dpi=160)
plt.close(fig)

times = np.arange(4)
mean0 = np.array([numeric(panel.entry_mean(0, t), long_values) for t in times])
mean1 = np.array([numeric(panel.entry_mean(1, t), long_values) for t in times])
variance = np.array(
    [numeric(panel.entry_covariance(0, t, t), long_values) for t in times]
)
sd = np.sqrt(variance)
lag_cov = np.array(
    [numeric(panel.entry_covariance(0, 0, t), long_values) for t in times]
)

block = long_dense.block(subject, 0)
block_cov = numeric_matrix(block.covariance(), long_values)
block_sd = np.sqrt(np.diag(block_cov))
block_corr = block_cov / np.outer(block_sd, block_sd)

fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))

ax = axes[0, 0]
ax.plot(times, mean0, "o-", label="group 0")
ax.plot(times, mean1, "o-", label="group 1")
ax.fill_between(times, mean0 - 1.96 * sd, mean0 + 1.96 * sd, alpha=0.15)
ax.set_title("Mean trajectories")
ax.set_xlabel("time")
ax.set_ylabel("E[Y]")
ax.legend(frameon=False)
ax.grid(alpha=0.25)

ax = axes[0, 1]
ax.plot(times, variance, "o-", color="#2a9d8f")
ax.set_title("Within-subject variance")
ax.set_xlabel("time")
ax.set_ylabel("Var(Y)")
ax.grid(alpha=0.25)

ax = axes[1, 0]
ax.plot(times, lag_cov, "o-", color="#e76f51")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Covariance with baseline")
ax.set_xlabel("lag")
ax.set_ylabel("Cov(Y_0, Y_t)")
ax.grid(alpha=0.25)

ax = axes[1, 1]
im = ax.imshow(block_corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_title("Within-subject correlation")
ax.set_xticks(times, [f"T{t}" for t in times])
ax.set_yticks(times, [f"T{t}" for t in times])
fig.colorbar(im, ax=ax, shrink=0.8)

fig.tight_layout()
fig.savefig(OUT / "longitudinal_moments.png", dpi=160)
plt.close(fig)

print("longitudinal covariance:\n", long_cov.round(3))
print("entry mean (subject 1, t=1):", panel.entry_mean(1, 1))
print("entry variance (subject 0, t=2):", panel.entry_covariance(0, 2, 2))
print("entry covariance (subject 0, 0, 2):", panel.entry_covariance(0, 0, 2))
print("terms (0, 1):", long_dense.covariance_terms(0, 1))
