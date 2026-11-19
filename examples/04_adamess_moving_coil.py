"""
AdaMESS: a coil passing over an observation region.

Full-order "solver": the magnetic flux density of a small coil (magnetic
dipole) that moves along a straight line at height h over a 30 x 30 grid of
observation points. Far away the field is weak and nearly static; while the
coil passes over the region it changes quickly. AdaMESS decides on the fly
where along the path the next (expensive) solve should happen.

The script shows
  * where AdaMESS places its solves and how the step size evolves,
  * the per-window diagnostics (MESS keep ratio, gradient brake),
  * the cost and POD accuracy compared with the dense uniform sweep you would
    otherwise run at the minimum step size.

Run:  python examples/04_adamess_moving_coil.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mess import AdaMESS, build_pod_basis

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# -------------------------------------------------------- full-order model
g = np.linspace(-0.1, 0.1, 30)                     # observation grid [m]
GX, GY = np.meshgrid(g, g)
POINTS = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
MOMENT = np.array([0.0, 0.0, 1.0])
HEIGHT = 0.05                                      # coil height above the grid [m]
S0, S1 = -1.0, 1.0                                 # coil path along x [m]


def solve(s):
    """Dipole field B (up to mu0/4pi) at all observation points, flattened."""
    r = POINTS - np.array([s, 0.0, HEIGHT])
    d = np.linalg.norm(r, axis=1, keepdims=True)
    B = (3 * r * (r @ MOMENT)[:, None] / d**2 - MOMENT) / d**3
    return B.ravel()                               # 3 * 900 = 2700 dof


# ------------------------------------------------------------ AdaMESS run
ada = AdaMESS(solve, S0, S1, target_keep=0.85, beta_grad=5.0)
res = ada.run()
pos = np.asarray(res.positions)
print(f"AdaMESS: {res.n_fom_solves} solves "
      f"(step range {np.diff(pos).min():.4f} .. {np.diff(pos).max():.4f} m)")

# --------------------------------------- reference: dense uniform sweep
s_dense = np.arange(S0, S1 + 1e-12, ada.step_min)
S_dense = np.column_stack([solve(s) for s in s_dense])
print(f"dense uniform sweep at step_min: {len(s_dense)} solves")

s_test = np.linspace(S0, S1, 2500)                 # independent test positions
S_test = np.column_stack([solve(s) for s in s_test])


def pod_error(snapshots, rank):
    V, _ = build_pod_basis(snapshots, rank=rank)
    return np.linalg.norm(S_test - V @ (V.T @ S_test)) / np.linalg.norm(S_test)


print(f"\n{'POD rank':>8s} {'AdaMESS':>10s} {'dense':>10s}")
for r in (10, 20, 40, 60):
    print(f"{r:8d} {pod_error(res.snapshots, r):10.2e} {pod_error(S_dense, r):10.2e}")
print(f"\nsolves: {res.n_fom_solves} vs {len(s_dense)} "
      f"({100 * res.n_fom_solves / len(s_dense):.0f} %)")

# ------------------------------------------------------------------ plots
bz_centre = S_test[2 * 900 + 465]                  # B_z near the grid centre
diag_pos = np.array([d["pos"] for d in res.diagnostics])
keep = np.array([d["keep_ratio"] for d in res.diagnostics])
brake = np.array([d["factor_grad"] for d in res.diagnostics])

fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
ax = axes[0]
ax.plot(s_test, bz_centre, color="0.3")
ax.vlines(pos, bz_centre.min(), bz_centre.min() + 0.08 * np.ptp(bz_centre),
          color="C2", lw=0.7)
ax.set_ylabel("B_z at grid centre")
ax.set_title(f"AdaMESS sample positions (N = {res.n_fom_solves})")

ax = axes[1]
ax.semilogy(pos[:-1], np.diff(pos), ".-", color="C2", ms=3)
ax.axhline(ada.step_min, color="0.6", ls="--", lw=1, label="step_min / step_max")
ax.axhline(ada.step_max, color="0.6", ls="--", lw=1)
ax.set_ylabel("step size [m]")
ax.legend(loc="upper right")

ax = axes[2]
ax.plot(diag_pos, keep, label="MESS keep ratio")
ax.plot(diag_pos, brake, label="gradient brake factor")
ax.axhline(ada.target_keep, color="0.6", ls="--", lw=1, label="target_keep")
ax.set_xlabel("coil position s [m]")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", ncol=3)
fig.tight_layout()
fig.savefig(OUT / "04_adamess_coil.png", dpi=150)
print(f"\nfigure written to {OUT / '04_adamess_coil.png'}")
if "agg" not in plt.get_backend().lower():
    plt.show()
