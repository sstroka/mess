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

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ._core import greedy_mess, normalized_distance_matrix


# ---------------------------------------------------------------------------
# Recurrence / MESS primitives
# ---------------------------------------------------------------------------
def recurrence_matrix_from_columns(M, percentile=10.0, use_abs=True):
    """Build the normalized boolean recurrence matrix from an ``(n_dof, n)``
    matrix ``M``.

    1. pairwise euclidean distances of the columns (of ``|M|`` if ``use_abs``)
    2. normalize by the maximum distance
    3. threshold at the ``percentile``-th percentile of the upper triangle

    Returns ``(R, epsilon, rr)``: boolean matrix, threshold, recurrence rate.
    """
    M = np.asarray(M)
    if M.shape[1] < 2:
        return np.zeros((0, 0), dtype=bool), 0.0, 0.0

    D = normalized_distance_matrix(np.abs(M) if use_abs else M)
    iu = np.triu_indices_from(D, k=1)
    epsilon = float(np.percentile(D[iu], percentile))
    R = D < epsilon
    rr = float(R[iu].mean())
    return R, epsilon, rr


def mess_local(recurrence_matrix):
    """Greedy MESS for the sliding window (alias of the shared core).

    Expects an ``(n, n)`` boolean matrix, returns an ``(n,)`` boolean vector.
    """
    return greedy_mess(recurrence_matrix)


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
    keep_ratio = float(keep_mask.mean()) if keep_mask.size > 0 else 0.0

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
    step_history: list                # spacing to the next snapshot (per snapshot)
    n_fom_solves: int                 # = len(positions)
    diagnostics: list = field(default_factory=list)

    def save(self, path):
        """Write the result to an HDF5 file (requires ``h5py``)."""
        import h5py
        with h5py.File(path, "w") as f:
            f.attrs["variant"] = self.variant
            f.attrs["n_fom_solves"] = self.n_fom_solves
            f.create_dataset("positions", data=np.asarray(self.positions, dtype=float))
            f.create_dataset("step_history", data=np.asarray(self.step_history, dtype=float))
            f.create_dataset("snapshots", data=self.snapshots)
            if self.diagnostics:
                g = f.create_group("diagnostics")
                for key in self.diagnostics[0]:
                    g.create_dataset(key, data=np.asarray(
                        [d.get(key, np.nan) for d in self.diagnostics], dtype=float))

    @classmethod
    def load(cls, path):
        """Read a result written by :meth:`save`."""
        import h5py
        with h5py.File(path, "r") as f:
            diagnostics = []
            if "diagnostics" in f:
                g = f["diagnostics"]
                cols = {k: g[k][:] for k in g}
                n = len(next(iter(cols.values()))) if cols else 0
                diagnostics = [{k: float(v[i]) for k, v in cols.items()} for i in range(n)]
            return cls(
                variant=str(f.attrs["variant"]),
                positions=[float(p) for p in f["positions"][:]],
                snapshots=f["snapshots"][:],
                step_history=[float(s) for s in f["step_history"][:]],
                n_fom_solves=int(f.attrs["n_fom_solves"]),
                diagnostics=diagnostics,
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
        Parameter range to traverse (``end > start``).
    target_keep : float, default 0.85
        Target MESS keep ratio (``zeta`` in the paper), in ``(0, 1]``. The
        step is scaled by ``(target_keep / keep_ratio) ** beta_keep``, so a
        *higher* value lets the step grow sooner => fewer solves; a lower
        value samples more finely.
    beta_keep : float, default 4.0
        Keep-ratio exponent. Reacts symmetrically: shrink when too diverse,
        grow when too redundant.
    beta_grad : float, default 1.0
        Strength of the gradient brake (``beta`` in the paper).
    chunk_size : int, default 10
        Sliding-window length used for the recurrence/MESS analysis (>= 3).
    step_min, step_max, step_init : float, optional
        Step-size bounds. If not given, derived from the span:
        0.2 % / 5 % / 0.2 % of ``end - start`` respectively.
    epsilon_percentile : float, default 10.0
        Percentile of the window distances used as recurrence threshold.
        Being a percentile, it is independent of the scale of the data.
    alpha_smooth : float, default 0.8
        Exponential smoothing of the step update.
    normalize_gradients : bool, default False
        Divide the snapshot differences by the actual step sizes before the
        analysis (true finite-difference gradients instead of raw changes).
    abs_diffs : bool, default True
        Take ``|P_{i+1} - P_i|`` element-wise before computing distances (as in
        the paper). Set ``False`` to keep the sign information.
    include_end : bool, default True
        Shorten the final step so that a snapshot is computed exactly at
        ``end``. (Before v0.2 the range end was generally not sampled.)
    max_solves : int, optional
        Hard cap on the number of ``solve`` calls.

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
    >>> res.positions[0], res.positions[-1]
    (-3.0, 3.0)
    """

    def __init__(self, solve: Callable[[float], np.ndarray], start: float, end: float,
                 target_keep: float = 0.85,
                 beta_keep: float = 4.0,
                 beta_grad: float = 1.0,
                 chunk_size: int = 10,
                 step_min: float | None = None,
                 step_max: float | None = None,
                 step_init: float | None = None,
                 epsilon_percentile: float = 10.0,
                 alpha_smooth: float = 0.8,
                 normalize_gradients: bool = False,
                 abs_diffs: bool = True,
                 include_end: bool = True,
                 max_solves: int | None = None):
        self.solve = solve
        self.start = float(start)
        self.end = float(end)
        if not self.end > self.start:
            raise ValueError(f"end ({end}) must be greater than start ({start})")
        if not 0.0 < target_keep <= 1.0:
            raise ValueError(f"target_keep must be in (0, 1], got {target_keep}")
        if int(chunk_size) < 3:
            raise ValueError(f"chunk_size must be >= 3, got {chunk_size}")
        if not 0.0 <= epsilon_percentile <= 100.0:
            raise ValueError(f"epsilon_percentile must be in [0, 100], got {epsilon_percentile}")

        self.target_keep = target_keep
        self.beta_keep = beta_keep
        self.beta_grad = beta_grad
        self.chunk_size = int(chunk_size)
        self.epsilon_percentile = epsilon_percentile
        self.alpha_smooth = alpha_smooth
        self.normalize_gradients = normalize_gradients
        self.abs_diffs = abs_diffs
        self.include_end = include_end
        self.max_solves = max_solves

        span = self.end - self.start
        self.step_min = step_min if step_min is not None else 0.002 * span
        self.step_max = step_max if step_max is not None else 0.05 * span
        self.step_init = step_init if step_init is not None else self.step_min
        if not 0.0 < self.step_min <= self.step_max:
            raise ValueError("need 0 < step_min <= step_max")

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

    def _analyse_window(self, window, recent_steps, step, pos):
        P = np.column_stack(window)                       # (n_dof, chunk)
        P_diffs = np.diff(P, axis=1)                      # (n_dof, chunk-1)

        if self.normalize_gradients and recent_steps.size == P_diffs.shape[1]:
            P_diffs = P_diffs / recent_steps[None, :]
        grad_norms = np.linalg.norm(P_diffs, axis=0)

        R, eps, rr = recurrence_matrix_from_columns(
            P_diffs, percentile=self.epsilon_percentile, use_abs=self.abs_diffs
        )
        keep_mask = mess_local(R)

        step, diag = self._step_update(step, keep_mask, grad_norms)
        diag.update({"rr": rr, "epsilon": eps, "pos": pos})
        return step, diag

    # -- adaptive snapshot generation --------------------------------------
    def run(self, variant_name: str = "AdaMESS") -> SamplingResult:
        """Generate snapshots adaptively over ``[start, end]``."""
        pos = self.start
        positions, snaps_list, step_history, diags = [], [], [], []
        step = self.step_init
        window: deque[np.ndarray] = deque(maxlen=self.chunk_size)
        # positions closer than this to `end` are snapped onto it
        snap_tol = 1e-9 * (self.end - self.start)

        while True:
            phi = np.asarray(self.solve(pos)).ravel()
            positions.append(pos)
            snaps_list.append(phi)
            window.append(phi)

            if len(window) == self.chunk_size:
                recent = np.asarray(step_history[-(self.chunk_size - 1):], dtype=float)
                step, diag = self._analyse_window(window, recent, step, pos)
                diags.append(diag)

            done = pos >= self.end or (
                self.max_solves is not None and len(positions) >= self.max_solves)
            next_pos = pos + step
            if not self.include_end and next_pos >= self.end:
                done = True
            if done:
                step_history.append(step)
                break
            if next_pos >= self.end - snap_tol:
                next_pos = self.end
            step_history.append(next_pos - pos)          # actual spacing
            pos = next_pos

        snaps = np.column_stack(snaps_list)
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
    ``energy_threshold`` of the cumulative singular-value energy. An explicit
    ``rank`` is capped at the number of available singular values.
    """
    U, sigma, _ = np.linalg.svd(np.asarray(snapshots), full_matrices=False)
    total = np.sum(sigma ** 2)
    if rank is None:
        if total == 0:
            rank = 1
        else:
            energy = np.cumsum(sigma ** 2) / total
            rank = int(np.searchsorted(energy, energy_threshold)) + 1
    rank = int(min(max(rank, 1), U.shape[1]))
    return U[:, :rank], sigma
