"""
Maximum Entropy Snapshot Sampling (MESS).

Selects a compact, information-rich subset of snapshots from a data sequence by
greedily keeping only those states that *increase* an estimate of the system's
epsilon-Frobenius entropy. Redundant / frequently recurring states are dropped.

Reference
---------
F. Kasolis and M. Clemens, "Maximum Entropy Snapshot Sampling for Reduced Basis
Generation", arXiv:2005.01280, 2020.  https://arxiv.org/abs/2005.01280
"""

import numpy as np

from scipy.spatial.distance import pdist, squareform


class MESS():
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
          (transition energy). Use this to prioritize high-change transitions.

    Attributes
    ----------
    state_matrix : np.ndarray
        Normalized pairwise-distance matrix (in [0, 1]) used as the basis for
        the recurrence test.
    recurrence_matrix : np.ndarray of bool
        ``state_matrix < eps`` — True where two states count as "recurrent"
        (close enough to be considered the same).
    sampled_sequence_elements : np.ndarray
        The selected subset of columns after :meth:`sampling`.
    """

    def __init__(self, sequence, analysis_type='state'):
        self.sequence = sequence
        self.analysis_type = analysis_type
        self.state_matrix = None
        self.recurrence_matrix = None
        self.sampled_sequence_elements = None

        if analysis_type == 'state':
            # --- State analysis ------------------------------------------------
            # Pairwise euclidean distance between every pair of snapshots
            # (columns). pdist works on rows, so we transpose: sequence.T has one
            # snapshot per row. squareform turns the condensed vector into the
            # full (n, n) symmetric distance matrix.
            self.state_matrix = pdist(self.sequence.T, metric='euclidean')
            self.state_matrix = squareform(self.state_matrix)
            # Normalize to [0, 1] so that the recurrence threshold `eps` has a
            # consistent, scale-independent meaning across different data sets.
            self.state_matrix /= self.state_matrix.max()

        elif analysis_type == 'energy':
            # --- Energy analysis ----------------------------------------------
            # Look at the transitions between consecutive snapshots instead of
            # the snapshots themselves.
            energy_states = np.diff(self.sequence, axis=1)        # column-to-column change
            energy_states = np.abs(energy_states)
            # Squared L2 norm of each transition = a scalar "energy" per step.
            energy_states = np.linalg.norm(energy_states, axis=0)**2

            # Distance matrix between these scalar energies (broadcasting trick:
            # |e_i - e_j| for all pairs).
            self.state_matrix = np.abs(energy_states[:, None] - energy_states[None, :])
            self.state_matrix /= self.state_matrix.max()

    def sampling(self, eps=0.01):
        """Run the maximum-entropy selection and return the kept snapshots.

        Parameters
        ----------
        eps : float, default 0.01
            Recurrence threshold in [0, 1]. Two states are "recurrent" (treated
            as redundant) when their normalized distance is below ``eps``.
            Larger ``eps`` ⇒ more states count as redundant ⇒ fewer snapshots
            are kept.

        Returns
        -------
        np.ndarray
            The selected columns of ``sequence``.
        """
        # Boolean recurrence matrix: True where two states are within eps.
        self.recurrence_matrix = self.state_matrix < eps

        # Greedy entropy-maximizing pass returns a boolean keep-mask.
        indices = self.__maximum_entropy_snapshot_sampling()

        if self.analysis_type == 'energy':
            # Energy analysis works on the (n-1) transitions, so the mask is one
            # element short of the snapshot count. Append one entry to realign
            # the mask with the original `sequence` columns.
            indices = np.append(indices, False)

        self.sampled_sequence_elements = self.sequence[:, indices]

        return self.sampled_sequence_elements

    def __maximum_entropy_snapshot_sampling(self):
        """Greedy core of MESS.

        Walks the snapshots in order, keeping a running estimate ``v`` of the
        epsilon-Frobenius potential. A candidate snapshot is *kept* only if
        adding it strictly increases the entropy estimate; otherwise it is
        considered redundant and skipped.

        The update ``v <- (k^2 v + d) / (k+1)^2`` is a Welford-style incremental
        mean of the potential over the ``k`` snapshots kept so far.

        Returns
        -------
        np.ndarray of bool
            Keep-mask over the columns (or transitions, for energy analysis).
        """
        n = self.recurrence_matrix.shape[1]

        # keep-mask; always anchor on the first snapshot.
        idx = np.zeros(n, dtype=bool)
        idx[0] = True

        v = 1   # running entropy-potential estimate
        k = 1   # number of snapshots kept so far

        for cnt in range(n - 1):
            # d = local recurrence count of the candidate against all kept
            # snapshots (x2 for symmetry, +1 for the self term). High d means
            # the candidate is "recurrent" with many already-kept states, i.e.
            # redundant.
            d = 2 * np.sum(self.recurrence_matrix[cnt + 1, idx]) + 1

            # Acceptance test: keep the candidate only if it increases the
            # entropy estimate, i.e. when d is small relative to the current
            # potential (d - (2k+1) v < 0).
            if (d - (2 * k + 1) * v) < 0:
                idx[cnt + 1] = True
                # incremental update of the potential and the kept-count
                v = (k * k * v + d) / ((k + 1)**2)
                k += 1

        return idx
