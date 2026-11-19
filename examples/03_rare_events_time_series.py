"""
MESS on sensor time series: keeping rare events.

A machine with 40 sensors switches between a few operating regimes. Regimes
recur many times, and one short fault (1 % of the time) occurs once. A small,
uniformly or randomly chosen subset of time steps easily misses the fault,
while MESS, which keeps states that are *not* recurrent, picks it up even
with a handful of snapshots.

Also shown: on noisy data the automatic eps is generous (noise makes many
states look new); ``tol=...`` or ``n_samples=...`` give tighter control.

Run:  python examples/03_rare_events_time_series.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mess import MESS, build_pod_basis

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(1)

# ---------------------------------------------------------------- data
N_SENSORS = 40
patterns = rng.standard_normal((N_SENSORS, 4))      # 3 regimes + 1 fault signature
REGIMES = ["idle", "load", "high load", "FAULT"]
plan = [(0, 600), (1, 500), (0, 400), (3, 30), (1, 600), (2, 400), (0, 470)]
label = np.concatenate([[r] * n for r, n in plan])
t = np.arange(label.size)
X = (patterns[:, label] * (1 + 0.1 * np.sin(2 * np.pi * t / 50))[None, :]
     + 0.02 * rng.standard_normal((N_SENSORS, label.size)))
print(f"data: {N_SENSORS} sensors x {label.size} time steps, "
      f"fault = {np.sum(label == 3)} steps\n")


def per_step_error(subset, rank=8):
    V, _ = build_pod_basis(subset, rank=min(rank, subset.shape[1]))
    return np.linalg.norm(X - V @ (V.T @ X), axis=0) / np.linalg.norm(X, axis=0)


# ------------------------------------------------- equal-budget comparison
K = 8
mess = MESS(X)
mess.sampling(n_samples=K)
idx = {
    "MESS": mess.indices_,
    "uniform": np.linspace(0, label.size - 1, K).astype(int),
    "random": np.sort(rng.choice(label.size, K, replace=False)),
}
print(f"{'K = ' + str(K):10s} {'regimes found':32s} {'mean err':>9s} {'fault err':>10s}")
errs = {}
for name, ii in idx.items():
    errs[name] = per_step_error(X[:, ii])
    found = ", ".join(REGIMES[r] for r in sorted(set(label[ii])))
    print(f"{name:10s} {found:32s} {errs[name].mean():9.3f} "
          f"{errs[name][label == 3].mean():10.3f}")

# ---------------------------------------------------- eps modes on noisy data
print("\neps modes on this (noisy) data:")
for kw in ({}, {"tol": 0.05}, {"n_samples": K}):
    sel = mess.sampling(**kw)
    name = str(kw) if kw else "auto"
    print(f"  {name:20s} -> {sel.shape[1]:4d} snapshots (eps={mess.eps_:.3f})")

# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
ax = axes[0]
ax.plot(t, X[:5].T, lw=0.6)
for name, marker, y, c in (("MESS", "v", 0.97, "k"), ("uniform", "^", 0.03, "0.5")):
    ax.plot(t[idx[name]], np.full(K, y), marker, color=c, ls="none", ms=8, label=name,
            transform=ax.get_xaxis_transform())
ax.axvspan(t[label == 3][0], t[label == 3][-1], color="red", alpha=0.15, label="fault")
ax.set_ylabel("sensor signals (5 of 40)")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False)
ax.set_title(f"Which {K} time steps to keep?", loc="left")

ax = axes[1]
for name in idx:
    ax.plot(t, errs[name], lw=1, label=name)
ax.set_ylim(0, 1.05)
ax.set_xlabel("time step")
ax.set_ylabel("rel. reconstruction error")
ax.legend(loc="upper right", ncol=3)
fig.tight_layout()
fig.savefig(OUT / "03_rare_events.png", dpi=150)
print(f"\nfigure written to {OUT / '03_rare_events.png'}")
if "agg" not in plt.get_backend().lower():
    plt.show()
