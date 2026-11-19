"""
Maximum Entropy Snapshot Sampling (MESS).

Selects a compact, information-rich subset of snapshots from a data sequence by
greedily keeping only those states that *increase* an estimate of the system's
epsilon-Frobenius entropy. Redundant / frequently recurring states are dropped.

Choosing ``eps``
----------------
The recurrence threshold ``eps`` is the only parameter of MESS, but its
absolute value is hard to pick: distances are normalized by the *largest*
distance, so a single outlier rescales everything, and ``'state'`` and
``'energy'`` analysis need thresholds that differ by orders of magnitude.
:meth:`MESS.sampling` therefore accepts several ways to specify it:

=================  ==========================================================
``eps=0.05``       absolute threshold on the normalized distances (classic)
``quantile=0.02``  ``eps`` = 2 % quantile of all pairwise distances
                   (scale-free, comparable across data sets)
``n_samples=100``  ``eps`` chosen so that ~100 snapshots are kept
``keep_ratio=0.1`` ``eps`` chosen so that ~10 % of the snapshots are kept
``tol=1e-3``       largest ``eps`` whose selection still represents the full
                   sequence with relative projection error <= ``tol``
*nothing*          ``'auto'``: knee of the curve "kept snapshots vs. eps"
=================  ==========================================================

Reference
---------
F. Kasolis and M. Clemens, "Maximum Entropy Snapshot Sampling for Reduced Basis
Generation", arXiv:2005.01280, 2020.  https://arxiv.org/abs/2005.01280
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from ._core import greedy_mess_from_distance, normalized_distance_matrix

AnalysisType = Literal["state", "energy"]

#: Quantile grid used by ``'auto'`` and :meth:`MESS.eps_sweep`.
DEFAULT_QUANTILES = np.logspace(-3, np.log10(0.5), 40)


class EpsSweep(NamedTuple):
    """Result of :meth:`MESS.eps_sweep`."""

    eps: np.ndarray        #: thresholds that were evaluated
    quantile: np.ndarray   #: the same thresholds expressed as distance quantiles
    n_kept: np.ndarray     #: number of selected snapshots for each threshold
    knee: int              #: index of the knee (what ``'auto'`` picks)


def _knee_index(x: np.ndarray, y: np.ndarray) -> int:
    """Kneedle for a decreasing curve: the point furthest below the chord
    between the first and the last point (after normalizing both axes)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(y) == 0 or np.ptp(x) == 0:
        return len(x) // 2
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y.min()) / np.ptp(y)
    return int(np.argmax((1.0 - xn) - yn))


def _projection_error(sequence: np.ndarray, basis_columns: np.ndarray) -> float:
    """Relative Frobenius error of projecting ``sequence`` onto
    ``span(basis_columns)``."""
    norm = np.linalg.norm(sequence)
    if norm == 0:
        return 0.0
    U, s, _ = np.linalg.svd(basis_columns, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return 1.0
    U = U[:, s > s[0] * max(basis_columns.shape) * np.finfo(float).eps]
    residual = sequence - U @ (U.T @ sequence)
    return float(np.linalg.norm(residual) / norm)


class MESS:
    """Maximum Entropy Snapshot Sampling.

    Parameters
    ----------
    sequence : np.ndarray
        Data matrix of shape ``(n_features, n_snapshots)``. Each *column* is one
        snapshot (one system state). The method selects a subset of columns.
    analysis_type : {'state', 'energy'}, default 'state'
        - ``'state'``  : compare snapshots directly (pairwise distance of the
          state vectors). Use this for "which states are redundant?".
        - ``'energy'`` : compare the *change* between consecutive snapshots
          (transition energy ``||x_{i+1} - x_i||^2``). Use this to prioritize
          high-change transitions. Selected transition ``i`` maps to snapshot
          ``i`` (its start).
    distance_method : {'gram', 'pdist'}, default 'gram'
        How pairwise state distances are computed (``'state'`` analysis).
        ``'gram'`` uses one matrix product and is much faster for wide data;
        ``'pdist'`` is the exact (slower) pairwise evaluation.

    Attributes
    ----------
    state_matrix : np.ndarray
        Normalized pairwise-distance matrix (in ``[0, 1]``).
    eps_ : float or None
        Threshold used by the last call to :meth:`sampling`.
    mask_ : np.ndarray of bool or None
        Keep-mask over the columns of ``sequence`` from the last call.
    indices_ : np.ndarray of int or None
        Column indices of the selected snapshots (``np.flatnonzero(mask_)``).
    sampled_sequence_elements : np.ndarray or None
        The selected columns, ``sequence[:, indices_]``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 4 * np.pi, 400)
    >>> data = np.vstack([np.sin(k * t) for k in range(1, 6)])   # (5, 400)
    >>> mess = MESS(data)
    >>> reduced = mess.sampling(n_samples=20)                    # ~20 snapshots
    >>> abs(reduced.shape[1] - 20) <= 2
    True
    >>> reduced = mess.sampling(quantile=0.05)                   # scale-free eps
    >>> reduced = mess.sampling()                                # automatic eps
    """

    def __init__(
        self,
        sequence: np.ndarray,
        analysis_type: AnalysisType = "state",
        distance_method: Literal["gram", "pdist"] = "gram",
    ):
        sequence = np.asarray(sequence)
        if sequence.ndim != 2:
            raise ValueError(
                f"sequence must be 2-D (n_features, n_snapshots), got shape {sequence.shape}"
            )
        if analysis_type not in ("state", "energy"):
            raise ValueError(f"analysis_type must be 'state' or 'energy', got {analysis_type!r}")

        if analysis_type == "energy" and sequence.shape[1] < 2:
            raise ValueError("'energy' analysis needs at least 2 snapshots")

        self.sequence = sequence
        self.analysis_type = analysis_type
        self.eps_: float | None = None
        self.mask_: np.ndarray | None = None
        self.indices_: np.ndarray | None = None
        self.sampled_sequence_elements: np.ndarray | None = None

        if analysis_type == "state":
            self.state_matrix = normalized_distance_matrix(sequence, method=distance_method)
        else:
            # scalar transition energy per step, then |e_i - e_j| for all pairs
            # (1-D, so the exact pairwise difference is cheap)
            energies = np.linalg.norm(np.diff(sequence.astype(float), axis=1), axis=0) ** 2
            self.state_matrix = normalized_distance_matrix(energies[None, :], method="pdist")

        n = self.state_matrix.shape[0]
        # sorted off-diagonal distances: used for quantiles and bisection
        self._distances = np.sort(self.state_matrix[np.triu_indices(n, k=1)])
        self._cache: dict[float, np.ndarray] = {}

    # ------------------------------------------------------------------ core
    @property
    def recurrence_matrix(self) -> np.ndarray | None:
        """``state_matrix < eps_`` (computed on demand; ``None`` before
        :meth:`sampling` has been called)."""
        if self.eps_ is None:
            return None
        return self.state_matrix < self.eps_

    def _mask_for(self, eps: float) -> np.ndarray:
        """Keep-mask over the *rows of the state matrix* for threshold ``eps``."""
        eps = float(eps)
        if eps not in self._cache:
            self._cache[eps] = greedy_mess_from_distance(self.state_matrix, eps)
        return self._cache[eps]

    def _n_kept(self, eps: float) -> int:
        return int(self._mask_for(eps).sum())

    def _to_sequence_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.analysis_type == "energy":
            # n-1 transitions -> n snapshots; transition i selects snapshot i
            mask = np.append(mask, False)
        return mask

    # ----------------------------------------------------- eps resolution
    def _candidates(self) -> np.ndarray:
        """All thresholds that lead to distinct recurrence matrices, in
        increasing order: the distinct distances plus one value above the max."""
        uniq = np.unique(self._distances)
        top = np.nextafter(uniq[-1], np.inf) if uniq.size else 1.0
        return np.append(uniq, top)

    def eps_from_quantile(self, q: float) -> float:
        """Threshold equal to the ``q``-quantile of all pairwise distances."""
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {q}")
        if self._distances.size == 0:
            return 1.0
        eps = float(np.quantile(self._distances, q))
        if eps == 0.0 and self._distances[-1] == 0.0:
            # all snapshots identical: everything is recurrent
            return 1.0
        return eps

    def _eps_for_count(self, target: int) -> float:
        """Bisection over the candidate thresholds for ~``target`` snapshots.

        The kept count decreases (almost) monotonically with ``eps``; among the
        bracketing thresholds the one whose count is closest to ``target`` wins.
        """
        cand = self._candidates()
        lo, hi = 0, len(cand) - 1          # n_kept(cand[lo]) is largest
        if self._n_kept(cand[lo]) <= target:
            return float(cand[lo])
        if self._n_kept(cand[hi]) >= target:
            return float(cand[hi])
        while hi - lo > 1:                 # invariant: n(lo) > target >= n(hi)
            mid = (lo + hi) // 2
            if self._n_kept(cand[mid]) > target:
                lo = mid
            else:
                hi = mid
        n_lo, n_hi = self._n_kept(cand[lo]), self._n_kept(cand[hi])
        return float(cand[lo] if (n_lo - target) < (target - n_hi) else cand[hi])

    def _eps_for_tol(self, tol: float) -> float:
        """Largest threshold whose selection keeps the projection error <= tol."""
        cand = self._candidates()
        seq = np.asarray(self.sequence, dtype=float)

        def ok(eps: float) -> bool:
            idx = self._to_sequence_mask(self._mask_for(eps))
            return _projection_error(seq, seq[:, idx]) <= tol

        lo, hi = 0, len(cand) - 1
        if ok(cand[hi]):
            return float(cand[hi])
        if not ok(cand[lo]):
            return float(cand[lo])         # best we can do
        while hi - lo > 1:                 # invariant: ok(lo) and not ok(hi)
            mid = (lo + hi) // 2
            if ok(cand[mid]):
                lo = mid
            else:
                hi = mid
        return float(cand[lo])

    def eps_sweep(self, quantiles: np.ndarray | None = None) -> EpsSweep:
        """Number of selected snapshots as a function of ``eps``.

        Parameters
        ----------
        quantiles : array-like, optional
            Distance quantiles at which to evaluate. Defaults to 40
            log-spaced values in ``[1e-3, 0.5]``.

        Returns
        -------
        EpsSweep
            ``(eps, quantile, n_kept, knee)``; plot ``n_kept`` over ``eps``
            (log x-axis) to see the trade-off. ``knee`` is the index that
            ``sampling()`` uses in ``'auto'`` mode.
        """
        q = DEFAULT_QUANTILES if quantiles is None else np.sort(np.asarray(quantiles, float))
        eps = np.array([self.eps_from_quantile(qi) for qi in q])
        n_kept = np.array([self._n_kept(e) for e in eps])
        knee = _knee_index(np.log(q), n_kept)
        return EpsSweep(eps=eps, quantile=q, n_kept=n_kept, knee=knee)

    def resolve_eps(
        self,
        eps: float | str | None = None,
        *,
        quantile: float | None = None,
        n_samples: int | None = None,
        keep_ratio: float | None = None,
        tol: float | None = None,
    ) -> float:
        """Translate any of the supported specifications into an absolute
        threshold. Same arguments as :meth:`sampling`."""
        given = {
            name: val
            for name, val in dict(
                eps=None if eps == "auto" else eps,
                quantile=quantile, n_samples=n_samples, keep_ratio=keep_ratio, tol=tol,
            ).items()
            if val is not None
        }
        if len(given) > 1:
            raise ValueError(
                "Specify at most one of eps, quantile, n_samples, keep_ratio, tol; "
                f"got {sorted(given)}"
            )
        n_states = self.state_matrix.shape[0]

        if eps is not None and eps != "auto":
            if isinstance(eps, str):
                raise ValueError(f"eps must be a number or 'auto', got {eps!r}")
            return float(eps)
        if quantile is not None:
            return self.eps_from_quantile(quantile)
        if n_samples is not None:
            if n_samples < 1:
                raise ValueError(f"n_samples must be >= 1, got {n_samples}")
            return self._eps_for_count(int(n_samples))
        if keep_ratio is not None:
            if not 0.0 < keep_ratio <= 1.0:
                raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
            return self._eps_for_count(max(1, round(keep_ratio * n_states)))
        if tol is not None:
            if tol < 0:
                raise ValueError(f"tol must be >= 0, got {tol}")
            return self._eps_for_tol(float(tol))
        # auto
        sweep = self.eps_sweep()
        return float(sweep.eps[sweep.knee])

    # ------------------------------------------------------------ public
    def sampling(
        self,
        eps: float | str | None = None,
        *,
        quantile: float | None = None,
        n_samples: int | None = None,
        keep_ratio: float | None = None,
        tol: float | None = None,
    ) -> np.ndarray:
        """Run the maximum-entropy selection and return the kept snapshots.

        Specify **at most one** of the arguments below; with none, ``eps`` is
        chosen automatically at the knee of :meth:`eps_sweep`.

        Parameters
        ----------
        eps : float or 'auto', optional
            Absolute recurrence threshold on the normalized distances in
            ``[0, 1]``. Larger ``eps`` => more states count as redundant =>
            fewer snapshots are kept.
        quantile : float, optional
            Use the ``quantile``-quantile of all pairwise distances as ``eps``.
            Scale-free, so the same value behaves similarly on different data.
        n_samples : int, optional
            Choose ``eps`` such that (approximately) ``n_samples`` snapshots are
            kept. Exact whenever the target count is attainable.
        keep_ratio : float, optional
            Like ``n_samples`` but relative to the sequence length.
        tol : float, optional
            Choose the largest ``eps`` for which the selected snapshots still
            span the full sequence with relative Frobenius projection error
            ``<= tol``. Costs one SVD of the selection per bisection step.

        Returns
        -------
        np.ndarray
            The selected columns of ``sequence``. The threshold and indices
            are available as ``eps_`` and ``indices_``.
        """
        self.eps_ = self.resolve_eps(
            eps, quantile=quantile, n_samples=n_samples, keep_ratio=keep_ratio, tol=tol
        )
        self.mask_ = self._to_sequence_mask(self._mask_for(self.eps_))
        self.indices_ = np.flatnonzero(self.mask_)
        self.sampled_sequence_elements = self.sequence[:, self.mask_]
        return self.sampled_sequence_elements
