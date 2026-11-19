"""Low-level building blocks shared by :mod:`mess.mess_core` and :mod:`mess.adamess`."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def _gram_distances(columns: np.ndarray) -> np.ndarray:
    """Euclidean column distances via ``|a|^2 + |b|^2 - 2 a.b`` (one BLAS
    matrix product; much faster than ``pdist`` for many features). Columns are
    centred first, which leaves distances unchanged but reduces cancellation."""
    C = columns - columns.mean(axis=1, keepdims=True)
    sq = np.einsum("ij,ij->j", C, C)
    D2 = C.T @ C
    D2 *= -2.0
    D2 += sq[:, None]
    D2 += sq[None, :]
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    D2 = 0.5 * (D2 + D2.T)        # exact symmetry
    return np.sqrt(D2, out=D2)


def normalized_distance_matrix(columns: np.ndarray, method: str = "gram") -> np.ndarray:
    """Pairwise euclidean distances between the columns of ``columns``,
    scaled to ``[0, 1]`` by the largest distance.

    Parameters
    ----------
    columns : (n_features, n) array
    method : {'gram', 'pdist'}
        ``'gram'`` (default) uses a single matrix product and is typically
        10-100x faster; ``'pdist'`` is the exact pairwise computation of
        :func:`scipy.spatial.distance.pdist`. Both agree to ~1e-8 relative.

    If all columns are identical the (all-zero) matrix is returned unscaled
    instead of producing ``NaN``. Complex data is supported (distances use
    ``|a - b|`` with the complex modulus).
    """
    columns = np.asarray(columns)
    if np.iscomplexobj(columns):
        # |a-b|^2 = |Re(a-b)|^2 + |Im(a-b)|^2  -> stack real and imaginary parts
        columns = np.vstack([columns.real, columns.imag])
    columns = columns.astype(float, copy=False)
    if columns.shape[1] < 2:
        return np.zeros((columns.shape[1], columns.shape[1]))
    if method == "gram":
        D = _gram_distances(columns)
    elif method == "pdist":
        D = squareform(pdist(columns.T, metric="euclidean"))
    else:
        raise ValueError(f"method must be 'gram' or 'pdist', got {method!r}")
    max_val = D.max()
    if max_val > 0:
        D /= max_val
    return D


def greedy_mess(recurrence_matrix: np.ndarray) -> np.ndarray:
    """Greedy core of MESS on a boolean recurrence matrix.

    Walks the states in order and keeps a candidate only if it increases the
    running estimate ``v`` of the epsilon-Frobenius potential. ``v`` is updated
    Welford-style, ``v <- (k^2 v + d) / (k+1)^2``, where ``d`` is twice the
    number of kept states the candidate recurs with, plus one (self term).

    Instead of re-summing ``R[i, kept]`` for every candidate, a running count
    per row is updated only when a state is accepted. That makes each
    rejected candidate O(1) and each accepted one O(n).

    Parameters
    ----------
    recurrence_matrix : (n, n) array of bool

    Returns
    -------
    (n,) array of bool
        Keep-mask; the first state is always kept.
    """
    R = np.asarray(recurrence_matrix, dtype=bool)
    n = R.shape[1] if R.ndim == 2 else 0
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = True
    counts = R[:, 0].astype(np.int64)   # counts[i] = sum_j kept R[i, j]
    v, k = 1.0, 1
    for i in range(1, n):
        d = 2 * counts[i] + 1
        if (d - (2 * k + 1) * v) < 0:
            keep[i] = True
            counts += R[:, i]
            v = (k * k * v + d) / ((k + 1) ** 2)
            k += 1
    return keep


def greedy_mess_from_distance(D: np.ndarray, eps: float) -> np.ndarray:
    """Same as ``greedy_mess(D < eps)`` for a *symmetric* distance matrix,
    but without materializing the ``(n, n)`` boolean matrix. This is what
    makes scanning many ``eps`` values cheap.
    """
    n = D.shape[0]
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = True
    counts = (D[0] < eps).astype(np.int64)
    v, k = 1.0, 1
    for i in range(1, n):
        d = 2 * counts[i] + 1
        if (d - (2 * k + 1) * v) < 0:
            keep[i] = True
            counts += D[i] < eps
            v = (k * k * v + d) / ((k + 1) ** 2)
            k += 1
    return keep
