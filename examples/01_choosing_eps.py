"""
Choosing eps — all the ways to tell MESS how aggressive it should be.

The data is a synthetic field: a Gaussian pulse that travels at varying speed
on top of a slowly breathing background mode. The script compares the
different eps specifications and plots the "kept snapshots vs. eps" curve
that the automatic mode uses.

Run:  python examples/01_choosing_eps.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mess import MESS
from mess.mess_core import _projection_error

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
x = np.linspace(0, 1, 300)
t = np.linspace(0, 1, 1000)
centre = 0.1 + 0.8 * (t + 0.15 * np.sin(6 * np.pi * t))       # non-uniform speed
data = (np.exp(-((x[:, None] - centre[None, :]) ** 2) / 0.002)
        + 0.3 * np.cos(np.pi * x)[:, None] * np.cos(2 * np.pi * t)[None, :])
print(f"data: {data.shape[0]} features x {data.shape[1]} snapshots\n")

# ------------------------------------------------- compare eps modes
mess = MESS(data)
modes = {
    "auto (default)":     {},
    "eps=0.05":           {"eps": 0.05},
    "quantile=0.02":      {"quantile": 0.02},
    "n_samples=60":       {"n_samples": 60},
    "keep_ratio=0.1":     {"keep_ratio": 0.1},
    "tol=1e-2":           {"tol": 1e-2},
}
print(f"{'mode':18s} {'eps':>8s} {'kept':>6s} {'proj. error':>12s} {'time':>8s}")
print("-" * 56)
for label, kw in modes.items():
    t0 = time.perf_counter()
    sel = mess.sampling(**kw)
    dt = time.perf_counter() - t0
    err = _projection_error(data, sel)
    print(f"{label:18s} {mess.eps_:8.4f} {sel.shape[1]:6d} {err:12.2e} {dt * 1e3:6.1f}ms")

# ------------------------------------------- the same quantile is portable
print("\nThe same quantile on rescaled data gives the same selection:")
for scale in (1e-3, 1.0, 1e3):
    n = MESS(scale * data).sampling(quantile=0.02).shape[1]
    print(f"  data x {scale:g}: {n} snapshots")

# ---------------------------------------------------------- the sweep
sweep = mess.eps_sweep()
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(sweep.eps, sweep.n_kept, "o-", ms=3, label="kept snapshots")
ax.semilogx(sweep.eps[sweep.knee], sweep.n_kept[sweep.knee], "o", ms=10,
            mfc="none", mew=2, label=f"knee ('auto'): eps={sweep.eps[sweep.knee]:.3f}")
ax.set_xlabel("eps (normalized distance threshold)")
ax.set_ylabel("number of kept snapshots")
ax.set_title("MESS.eps_sweep()")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "01_eps_sweep.png", dpi=150)
print(f"\nfigure written to {OUT / '01_eps_sweep.png'}")
if "agg" not in plt.get_backend().lower():
    plt.show()
