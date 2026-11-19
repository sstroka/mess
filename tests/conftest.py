import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def oscillation():
    """Smooth, periodic sequence (20 features x 400 snapshots), rank 20."""
    t = np.linspace(0, 6 * np.pi, 400)
    return np.vstack([np.sin(k * t) * np.exp(-0.02 * t) for k in range(1, 11)]
                     + [np.cos(k * t) for k in range(1, 11)])


@pytest.fixture
def random_walk(rng):
    return rng.standard_normal((30, 250)).cumsum(axis=1)


def moving_pulse(n_dof=150, start=0.0, end=1000.0):
    """Synthetic solver: a Gaussian that moves fast in the middle of the range."""
    x = np.linspace(0, 1, n_dof)
    span = end - start

    def solve(s):
        sn = (s - start) / span
        centre = 0.5 * (1 + np.tanh(8 * (sn - 0.5)))
        return np.exp(-((x - centre) ** 2) / 0.004)

    return solve
