"""
MESS as a preprocessing step for POD on a PDE trajectory.

Viscous Burgers equation  u_t + u u_x = nu u_xx  on a periodic domain. A smooth
initial condition steepens into a shock within the first part of the
simulation, stays in place at x = pi and then only decays slowly. Uniform time
sampling therefore wastes most snapshots on the uneventful decay phase.

We keep the same number of snapshots with
  * uniform sub-sampling,
  * random sub-sampling,
  * MESS (``n_samples=...``, state analysis),
and compare how well a POD basis built from each subset reconstructs the
*full* trajectory.

Run:  python examples/02_burgers_snapshots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from mess import MESS, build_pod_basis

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------- full-order model
N, NU, T_END, N_T = 256, 0.004, 10.0, 2000
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
dx = x[1] - x[0]


def rhs(_, u):
    up, um = np.roll(u, -1), np.roll(u, 1)
    flux = 0.25 * (up**2 - um**2) / dx            # central, conservative
    return -flux + NU * (up - 2 * u + um) / dx**2


t_eval = np.linspace(0, T_END, N_T)
sol = solve_ivp(rhs, (0, T_END), np.sin(x), t_eval=t_eval,
                method="LSODA", rtol=1e-6, atol=1e-8)
U = sol.y                                          # (N, N_T)
print(f"trajectory: {U.shape[0]} dof x {U.shape[1]} time steps")


# ------------------------------------------------------------ comparison
def pod_error(subset, rank):
    V, _ = build_pod_basis(subset, rank=rank)
    return np.linalg.norm(U - V @ (V.T @ U)) / np.linalg.norm(U)


K = 40
mess = MESS(U)
subsets = {
    "uniform": U[:, np.linspace(0, N_T - 1, K).astype(int)],
    "random": U[:, np.sort(np.random.default_rng(0).choice(N_T, K, replace=False))],
    "MESS": mess.sampling(n_samples=K),
}
print(f"MESS kept {subsets['MESS'].shape[1]} snapshots (eps={mess.eps_:.4f})\n")

ranks = np.arange(1, K + 1)
errors = {name: [pod_error(S, r) for r in ranks] for name, S in subsets.items()}
print(f"{'rank':>5s}" + "".join(f"{n:>12s}" for n in subsets))
for r in (5, 10, 20, 30, 40):
    print(f"{r:5d}" + "".join(f"{errors[n][r - 1]:12.2e}" for n in subsets))

# ------------------------------------------------------------------ plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.pcolormesh(t_eval, x, U, shading="auto", cmap="RdBu_r")
for ti in t_eval[mess.indices_]:
    ax.axvline(ti, color="k", lw=0.5, alpha=0.5)
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title("Burgers trajectory, lines = MESS snapshots")

ax = axes[1]
for name, err in errors.items():
    ax.semilogy(ranks, err, lw=2, label=f"{name} ({K} snapshots)")
ax.set_xlabel("POD rank")
ax.set_ylabel("relative reconstruction error (full trajectory)")
ax.set_title("POD from a subset of the snapshots")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "02_burgers.png", dpi=150)
print(f"\nfigure written to {OUT / '02_burgers.png'}")
if "agg" not in plt.get_backend().lower():
    plt.show()
