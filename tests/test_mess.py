import numpy as np
import pytest

from mess import MESS
from mess.mess_core import _projection_error


# ------------------------------------------------------------------ basics
@pytest.mark.parametrize("analysis_type", ["state", "energy"])
def test_output_is_subset_of_columns(random_walk, analysis_type):
    m = MESS(random_walk, analysis_type=analysis_type)
    out = m.sampling(eps=0.1)
    assert out.shape[0] == random_walk.shape[0]
    assert 1 <= out.shape[1] <= random_walk.shape[1]
    np.testing.assert_array_equal(out, random_walk[:, m.indices_])
    assert m.mask_.shape == (random_walk.shape[1],)
    assert m.mask_[0], "first snapshot is always kept"
    assert np.all(np.diff(m.indices_) > 0), "order is preserved"


def test_energy_never_selects_last_snapshot(random_walk):
    m = MESS(random_walk, analysis_type="energy")
    m.sampling(eps=0.0)                      # nothing recurrent -> all transitions kept
    assert m.mask_[:-1].all() and not m.mask_[-1]


def test_duplicates_are_dropped(rng):
    base = rng.standard_normal((10, 5))
    data = np.repeat(base, 20, axis=1)       # each state repeated 20 times
    out = MESS(data).sampling(eps=1e-6)
    assert out.shape[1] <= 5


def test_more_eps_fewer_snapshots(random_walk):
    m = MESS(random_walk)
    counts = [m.sampling(eps=e).shape[1] for e in (0.01, 0.05, 0.2, 0.5)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_constant_data_does_not_crash():
    data = np.ones((4, 30))
    for kw in ({}, {"eps": 0.1}, {"quantile": 0.1}, {"n_samples": 3}, {"tol": 1e-6}):
        out = MESS(data).sampling(**kw)
        assert np.isfinite(out).all()
        assert out.shape[1] >= 1


def test_recurrence_matrix_property(random_walk):
    m = MESS(random_walk)
    assert m.recurrence_matrix is None
    m.sampling(eps=0.1)
    np.testing.assert_array_equal(m.recurrence_matrix, m.state_matrix < 0.1)


@pytest.mark.parametrize("bad", [np.ones(5), np.ones((2, 3, 4))])
def test_rejects_non_2d(bad):
    with pytest.raises(ValueError):
        MESS(bad)


def test_rejects_unknown_analysis_type(random_walk):
    with pytest.raises(ValueError):
        MESS(random_walk, analysis_type="entropy")


def test_gram_and_pdist_agree(random_walk):
    a = MESS(random_walk, distance_method="gram").sampling(quantile=0.05)
    b = MESS(random_walk, distance_method="pdist").sampling(quantile=0.05)
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------- eps modes
def test_quantile_is_scale_invariant(random_walk):
    a = MESS(random_walk).sampling(quantile=0.05)
    m = MESS(1000.0 * random_walk + 3.0)
    b = m.sampling(quantile=0.05)
    np.testing.assert_allclose(b, 1000.0 * a + 3.0)


def test_quantile_matches_distance_distribution(random_walk):
    m = MESS(random_walk)
    m.sampling(quantile=0.1)
    n = m.state_matrix.shape[0]
    tri = m.state_matrix[np.triu_indices(n, 1)]
    assert np.mean(tri < m.eps_) == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize("analysis_type", ["state", "energy"])
@pytest.mark.parametrize("target", [1, 5, 20, 60])
def test_n_samples_hits_target(random_walk, analysis_type, target):
    m = MESS(random_walk, analysis_type=analysis_type)
    out = m.sampling(n_samples=target)
    # the kept count is a step function of eps, so allow a small miss
    assert abs(out.shape[1] - target) <= max(2, 0.05 * target)


def test_keep_ratio(random_walk):
    out = MESS(random_walk).sampling(keep_ratio=0.1)
    assert abs(out.shape[1] - 25) <= 2


def test_n_samples_larger_than_sequence(random_walk):
    out = MESS(random_walk).sampling(n_samples=10_000)
    assert out.shape[1] == random_walk.shape[1]


@pytest.mark.parametrize("tol", [1e-1, 1e-2, 1e-3])
def test_tol_is_respected(oscillation, tol):
    m = MESS(oscillation)
    out = m.sampling(tol=tol)
    assert _projection_error(oscillation, out) <= tol


def test_tol_picks_fewer_snapshots_for_looser_tolerance(random_walk):
    m = MESS(random_walk)
    loose = m.sampling(tol=0.2).shape[1]
    tight = m.sampling(tol=1e-3).shape[1]
    assert loose <= tight


def test_auto_is_default_and_reasonable(oscillation):
    m = MESS(oscillation)
    out = m.sampling()
    assert m.eps_ is not None
    assert 1 < out.shape[1] < oscillation.shape[1]
    assert m.sampling(eps="auto").shape == out.shape
    # the oscillation has rank 20; the knee should keep enough to span it
    assert _projection_error(oscillation, out) < 1e-6


def test_eps_sweep(random_walk):
    sweep = MESS(random_walk).eps_sweep()
    assert len(sweep.eps) == len(sweep.n_kept) == len(sweep.quantile)
    assert np.all(np.diff(sweep.eps) >= 0)
    assert sweep.n_kept[0] >= sweep.n_kept[-1]
    assert 0 <= sweep.knee < len(sweep.eps)


def test_eps_sweep_custom_quantiles(random_walk):
    sweep = MESS(random_walk).eps_sweep(quantiles=[0.3, 0.01, 0.1])
    np.testing.assert_array_equal(sweep.quantile, [0.01, 0.1, 0.3])


@pytest.mark.parametrize("kwargs", [
    {"eps": 0.1, "quantile": 0.1},
    {"n_samples": 5, "tol": 0.1},
    {"eps": "nope"},
    {"quantile": 1.5},
    {"n_samples": 0},
    {"keep_ratio": 0.0},
    {"tol": -1.0},
])
def test_invalid_eps_arguments(random_walk, kwargs):
    with pytest.raises(ValueError):
        MESS(random_walk).sampling(**kwargs)


# ------------------------------------------------------------- dtypes
def test_uint8_energy_matches_float(rng):
    """np.diff on uint8 wraps around; the energy analysis must not."""
    img = rng.integers(0, 256, size=(40, 120), dtype=np.uint8)
    a = MESS(img, analysis_type="energy")
    b = MESS(img.astype(float), analysis_type="energy")
    np.testing.assert_allclose(a.state_matrix, b.state_matrix)


@pytest.mark.parametrize("analysis_type", ["state", "energy"])
def test_complex_data_uses_modulus(random_walk, rng, analysis_type):
    Z = random_walk + 1j * rng.standard_normal(random_walk.shape).cumsum(axis=1)
    stacked = np.vstack([Z.real, Z.imag])
    a = MESS(Z, analysis_type=analysis_type)
    b = MESS(stacked, analysis_type=analysis_type)
    np.testing.assert_allclose(a.state_matrix, b.state_matrix, atol=1e-12)
    out = a.sampling(tol=1e-2)
    assert np.iscomplexobj(out)
    assert _projection_error(Z, out) <= 1e-2
