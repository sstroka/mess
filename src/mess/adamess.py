"""
Adaptive Maximum Entropy Snapshot Sampling (AdaMESS).

AdaMESS extends MESS from a *filter on a given sequence* to an *adaptive
generator*: instead of sampling a pre-existing data matrix, it decides *where*
along a 1-D parameter axis the next (expensive) full-order snapshot should be
computed. A sliding window of recent snapshots feeds a recurrence/MESS analysis
and a normalized-gradient brake, which together steer the local step size.

The algorithm is solver-agnostic. The caller supplies a `solve(s) -> vector`
callable that returns the full-order solution at parameter value ``s`` (e.g. a
field solve at a given coil position). AdaMESS treats it as a black box, and a
POD basis can be built directly from the sampled snapshots.

Reference
---------
S. Stroka et al., "Adaptive Maximum Entropy Snapshot Sampling for Memory
Efficient Reduced-Basis Generation", IEEE CEFC 2026.

Builds on the original MESS method:
F. Kasolis and M. Clemens, "Maximum Entropy Snapshot Sampling for Reduced
Basis Generation", arXiv:2005.01280, 2020.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# Recurrence / MESS primitives
# ---------------------------------------------------------------------------
def recurrence_matrix_from_columns(M, percentile=10.0):
    """Build the normalized boolean recurrence matrix from an ``(n_dof, n)``
    matrix ``M``.

    1. pairwise euclidean distances of the columns
    2. normalize by the maximum distance
    3. threshold at the ``percentile``-th percentile of the upper triangle

    Returns ``(R, epsilon, rr)``: boolean matrix, threshold, recurrence rate.
    """
    if M.shape[1] < 2:
        return np.zeros((0, 0), dtype=bool), 0.0, 0.0

    dist_vec = pdist(np.abs(M).T, metric="euclidean")
    D = squareform(dist_vec)
    max_val = D.max()
    if max_val > 0:
        D /= max_val

    triu = D[np.triu_indices_from(D, k=1)]
    epsilon = float(np.percentile(triu, percentile))
    R = D < epsilon
    rr = float(R[np.triu_indices_from(R, k=1)].mean())
    return R, epsilon, rr


def mess_local(recurrence_matrix):
    """Greedy MESS for the sliding window.

    Expects an ``(n, n)`` boolean matrix, returns an ``(n,)`` boolean vector.
    Uses the Welford-style update ``v <- (k^2 v + d) / (k+1)^2``.
    """
    n = recurrence_matrix.shape[1]
    idx = np.zeros(n, dtype=bool)
    if n == 0:
        return idx
    idx[0] = True
    v, k = 1.0, 1

    for cnt in range(n - 1):
        d = 2 * np.sum(recurrence_matrix[cnt + 1, idx]) + 1
        if (d - (2 * k + 1) * v) < 0:
            idx[cnt + 1] = True
            v = (k * k * v + d) / ((k + 1) ** 2)
            k += 1
    return idx


# ---------------------------------------------------------------------------
# Step-size control
# ---------------------------------------------------------------------------
def gradient_brake_normalized(grad_norms_per_step, beta_grad=1.0,
                              min_factor=0.3, quiet_rel=1e-3):
    """Brake derived from step-normalized gradients.

    Returns ``factor_grad`` in ``(0, 1]``: never expanding.

    Robustness:
      - reference = median of the previous gradients (outlier-robust)
      - ``quiet_rel``: in field-poor zones (median negligible against the
        maximum) no braking is applied, so noise ratios of tiny gradients
        cannot produce wild factors (scale independent).
    """
    g = np.asarray(grad_norms_per_step, dtype=float)
    if g.size < 2:
        return 1.0
    last_grad = g[-1]
    ref_grad = np.median(g[:-1])
    g_scale = np.max(np.abs(g))
    if g_scale <= 0.0 or ref_grad <= quiet_rel * g_scale:
        return 1.0
    ratio = last_grad / ref_grad
    if ratio <= 1.0:
        return 1.0
    return float(max(ratio ** (-beta_grad), min_factor))


def update_step_symmetric(step, keep_mask, grad_norms,
                          target_keep=0.85,
                          beta_keep=4.0,
                          beta_grad=1.0,
                          step_min=20.0,
                          step_max=500.0,
                          alpha_smooth=0.8):
    """Symmetric step control (the A4 / ``mess_full`` variant).

    A single keep-ratio exponent ``beta_keep`` reacts the same way in both
    directions: the step shrinks when the local data is too diverse
    (``keep_ratio > target_keep``) and grows when it is too redundant
    (``keep_ratio < target_keep``). The gradient brake never expands the step.

    Combination: ``factor_total = factor_keep * factor_grad``, then exponential
    smoothing and a clip to ``[step_min, step_max]``.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    keep_ratio = keep_mask.mean() if keep_mask.size > 0 else 0.0

    if keep_ratio < 1e-8:
        factor_keep = 2.0
    else:
        factor_keep = (target_keep / keep_ratio) ** beta_keep

    factor_grad = gradient_brake_normalized(grad_norms, beta_grad=beta_grad)
    factor_total = factor_keep * factor_grad

    step = float(step)
    step_raw = float(np.clip(step * factor_total, step_min, step_max))
    alpha = float(np.clip(alpha_smooth, 0.0, 1.0))
    step_new = (1.0 - alpha) * step + alpha * step_raw
    step_new = float(np.clip(step_new, step_min, step_max))

    diag = {
        "keep_ratio": keep_ratio,
        "factor_keep": factor_keep,
        "factor_grad": factor_grad,
        "factor_total": factor_total,
        "step_old": step,
        "step_new": step_new,
    }
    return step_new, diag


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SamplingResult:
    """Uniform return type of all sampling strategies."""
    variant: str
    positions: list                   # parameter coordinates of the snapshots
    snapshots: np.ndarray             # (n_dof, N) array
    step_history: list                # actual step sizes used
    n_fom_solves: int                 # = len(positions)
    diagnostics: list = field(default_factory=list)

    def save(self, path):
        import h5py
        with h5py.File(path, "w") as f:
            f.attrs["variant"] = self.variant
            f.attrs["n_fom_solves"] = self.n_fom_solves
            f.create_dataset("positions", data=np.asarray(self.positions))
            f.create_dataset("step_history", data=np.asarray(self.step_history))
            f.create_dataset("snapshots", data=self.snapshots)

    @classmethod
    def load(cls, path):
        import h5py
        with h5py.File(path, "r") as f:
            return cls(
                variant=str(f.attrs["variant"]),
                positions=list(f["positions"][:]),
                snapshots=f["snapshots"][:],
                step_history=list(f["step_history"][:]),
                n_fom_solves=int(f.attrs["n_fom_solves"]),
            )


# ---------------------------------------------------------------------------
# AdaMESS — adaptive snapshot generator
# ---------------------------------------------------------------------------
class AdaMESS:
    """Adaptive Maximum Entropy Snapshot Sampling.

    Walks a 1-D parameter axis ``[start, end]`` and adaptively places
    full-order snapshots. A sliding window of the most recent snapshots feeds
    a recurrence/MESS keep-ratio and a gradient brake, which jointly steer the
    local step size (see :func:`update_step_symmetric`). This is the symmetric
    A4 variant: a single ``beta_keep`` exponent, raw (non-normalized) gradients,
    and ``alpha_smooth = 0.8``.

    Parameters
    ----------
    solve : callable
        ``solve(s) -> np.ndarray`` returns the full-order solution vector at
        parameter value ``s``. Treated as a black box (any solver/backend).
    start, end : float
        Parameter range to traverse.
    target_keep : float, default 0.85
        Target MESS keep ratio (``zeta`` in the paper). Lower values keep fewer
        snapshots per window -> more aggressive thinning.
    beta_keep : float, default 4.0
        Keep-ratio exponent. Reacts symmetrically: shrink when too diverse,
        grow when too redundant.
    beta_grad : float, default 1.0
        Strength of the gradient brake (``beta`` in the paper).
    chunk_size : int, default 10
        Sliding-window length used for the recurrence/MESS analysis.
    step_min, step_max, step_init : float, optional
        Step-size bounds. If not given, derived from the span:
        0.2 % / 5 % / 0.2 % of ``|end - start|`` respectively.
    epsilon_percentile : float, default 10.0
        Percentile used to set the recurrence threshold.
    alpha_smooth : float, default 0.8
        Exponential smoothing of the step update.

    Examples
    --------
    >>> import numpy as np
    >>> def solve(s):
    ...     x = np.linspace(0, 1, 200)
    ...     return np.exp(-((x - 0.5 * (1 + np.tanh(s))) ** 2) / 0.01)
    >>> ada = AdaMESS(solve, start=-3.0, end=3.0, target_keep=0.75)
    >>> res = ada.run()
    >>> res.snapshots.shape[1] == res.n_fom_solves
    True
    """

    def __init__(self, solve, start, end,
                 target_keep=0.85,
                 beta_keep=4.0,
                 beta_grad=1.0,
                 chunk_size=10,
                 step_min=None,
                 step_max=None,
                 step_init=None,
                 epsilon_percentile=10.0,
                 alpha_smooth=0.8,
                 normalize_gradients=False):
        self.solve = solve
        self.start = float(start)
        self.end = float(end)
        self.target_keep = target_keep
        self.beta_keep = beta_keep
        self.beta_grad = beta_grad
        self.chunk_size = int(chunk_size)
        self.epsilon_percentile = epsilon_percentile
        self.alpha_smooth = alpha_smooth
        self.normalize_gradients = normalize_gradients

        span = abs(self.end - self.start)
        self.step_min = step_min if step_min is not None else 0.002 * span
        self.step_max = step_max if step_max is not None else 0.05 * span
        self.step_init = step_init if step_init is not None else 0.002 * span

    # -- step update used during the streaming loop -------------------------
    def _step_update(self, step, keep_mask, grad_norms):
        return update_step_symmetric(
            step=step,
            keep_mask=keep_mask,
            grad_norms=grad_norms,
            target_keep=self.target_keep,
            beta_keep=self.beta_keep,
            beta_grad=self.beta_grad,
            step_min=self.step_min,
            step_max=self.step_max,
            alpha_smooth=self.alpha_smooth,
        )

    # -- adaptive snapshot generation --------------------------------------
    def run(self, variant_name="AdaMESS"):
        """Generate snapshots adaptively over ``[start, end]``."""
        pos = self.start
        positions, snaps_list, step_history, diags = [], [], [], []
        step = self.step_init
        P_window = None

        while pos < self.end:
            phi = np.asarray(self.solve(pos)).ravel()
            positions.append(pos)
            snaps_list.append(phi)

            # update sliding window
            if P_window is None:
                P_window = phi.reshape(-1, 1)
            else:
                P_window = np.column_stack((P_window, phi))
                if P_window.shape[1] > self.chunk_size:
                    P_window = P_window[:, 1:]

            # warm-up
            if P_window.shape[1] < self.chunk_size:
                step_history.append(step)
                pos += step
                continue

            # analysis: gradients + recurrence
            P_diffs = np.diff(P_window, axis=1)             # (n_dof, chunk-1)
            recent_steps = np.asarray(
                step_history[-(P_window.shape[1] - 1):], dtype=float
            )

            if self.normalize_gradients and recent_steps.size == P_diffs.shape[1]:
                grad_norms = np.linalg.norm(P_diffs, axis=0) / recent_steps
                P_for_R = np.abs(P_diffs) / recent_steps[None, :]
            else:
                grad_norms = np.linalg.norm(P_diffs, axis=0)
                P_for_R = np.abs(P_diffs)

            R, eps, rr = recurrence_matrix_from_columns(
                P_for_R, percentile=self.epsilon_percentile
            )
            keep_mask = mess_local(R) if R.size > 0 else np.zeros(0, dtype=bool)

            step, diag = self._step_update(step, keep_mask, grad_norms)
            diag.update({"rr": rr, "epsilon": eps, "pos": pos})
            diags.append(diag)

            step_history.append(step)
            pos += step

        snaps = (np.column_stack(snaps_list) if snaps_list
                 else np.empty((0, 0)))
        return SamplingResult(
            variant=variant_name,
            positions=positions,
            snapshots=snaps,
            step_history=step_history,
            n_fom_solves=len(positions),
            diagnostics=diags,
        )


# ---------------------------------------------------------------------------
# POD basis helper
# ---------------------------------------------------------------------------
def build_pod_basis(snapshots, rank=None, energy_threshold=1 - 1e-6):
    """POD basis via SVD. Returns ``(V, sigma)``.

    If ``rank is None`` the basis is truncated automatically at
    ``energy_threshold`` of the cumulative singular-value energy.
    """
    U, sigma, _ = np.linalg.svd(snapshots, full_matrices=False)
    if rank is None:
        energy = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
        rank = max(int(np.searchsorted(energy, energy_threshold)) + 1, 1)
    return U[:, :rank], sigma
