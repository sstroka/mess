import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from mess import greedy_mess, mess_local, normalized_distance_matrix
from mess._core import greedy_mess_from_distance


def reference_greedy(R):
    """The original (pre-0.2) implementation, kept verbatim as an oracle."""
    n = R.shape[1]
    idx = np.zeros(n, dtype=bool)
    idx[0] = True
    v, k = 1, 1
    for cnt in range(n - 1):
        d = 2 * np.sum(R[cnt + 1, idx]) + 1
        if (d - (2 * k + 1) * v) < 0:
            idx[cnt + 1] = True
            v = (k * k * v + d) / ((k + 1) ** 2)
            k += 1
    return idx


@pytest.mark.parametrize("eps", [0.0, 0.02, 0.1, 0.3, 0.7, 1.1])
def test_greedy_matches_reference(random_walk, eps):
    D = normalized_distance_matrix(random_walk)
    R = D < eps
    expected = reference_greedy(R)
    np.testing.assert_array_equal(greedy_mess(R), expected)
    np.testing.assert_array_equal(greedy_mess_from_distance(D, eps), expected)
    np.testing.assert_array_equal(mess_local(R), expected)


def test_greedy_threshold_is_strict(random_walk):
    """eps exactly equal to an occurring distance: that pair must NOT count."""
    D = normalized_distance_matrix(random_walk)
    tri = np.sort(D[np.triu_indices_from(D, 1)])
    for eps in tri[[10, 100, 1000, 5000]]:
        expected = reference_greedy(D < eps)
        np.testing.assert_array_equal(greedy_mess_from_distance(D, eps), expected)


def test_greedy_nonsymmetric_matches_reference(rng):
    R = rng.random((60, 60)) < 0.3
    np.testing.assert_array_equal(greedy_mess(R), reference_greedy(R))


def test_greedy_edge_cases():
    assert greedy_mess(np.zeros((0, 0), bool)).shape == (0,)
    np.testing.assert_array_equal(greedy_mess(np.ones((1, 1), bool)), [True])
    # nothing recurrent -> everything kept; everything recurrent -> only the first
    assert greedy_mess(np.zeros((20, 20), bool)).all()
    assert greedy_mess(np.ones((20, 20), bool)).sum() == 1


@pytest.mark.parametrize("method", ["gram", "pdist"])
def test_distance_matrix(random_walk, method):
    D = normalized_distance_matrix(random_walk, method=method)
    ref = squareform(pdist(random_walk.T))
    np.testing.assert_allclose(D, ref / ref.max(), atol=1e-10)
    assert D.max() == pytest.approx(1.0)
    np.testing.assert_array_equal(D, D.T)
    np.testing.assert_array_equal(np.diag(D), 0.0)


def test_distance_matrix_constant_columns():
    D = normalized_distance_matrix(np.ones((5, 8)))
    assert np.isfinite(D).all() and (D == 0).all()


def test_distance_matrix_bad_method(random_walk):
    with pytest.raises(ValueError):
        normalized_distance_matrix(random_walk, method="nope")
