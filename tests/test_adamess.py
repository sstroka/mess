import numpy as np
import pytest
from conftest import moving_pulse

from mess import (
    AdaMESS,
    SamplingResult,
    build_pod_basis,
    gradient_brake_normalized,
    recurrence_matrix_from_columns,
    update_step_symmetric,
)

START, END = 0.0, 1000.0


@pytest.fixture
def result():
    ada = AdaMESS(moving_pulse(start=START, end=END), START, END,
                  target_keep=0.85, beta_grad=5.0)
    return ada.run()


# ------------------------------------------------------------ AdaMESS.run
def test_covers_full_range(result):
    assert result.positions[0] == START
    assert result.positions[-1] == END
    assert np.all(np.diff(result.positions) > 0)


def test_consistent_bookkeeping(result):
    n = result.n_fom_solves
    assert result.snapshots.shape == (150, n)
    assert len(result.positions) == len(result.step_history) == n
    np.testing.assert_allclose(np.diff(result.positions), result.step_history[:-1])


def test_steps_within_bounds():
    ada = AdaMESS(moving_pulse(start=START, end=END), START, END)
    res = ada.run()
    steps = np.diff(res.positions)
    assert np.all(steps[:-1] >= ada.step_min - 1e-9)   # last step may be shortened
    assert np.all(steps <= ada.step_max + 1e-9)


def test_denser_where_solution_changes_fast(result):
    pos = np.asarray(result.positions)
    centre = np.sum((pos > 400) & (pos < 600))
    edge = np.sum(pos > 800)
    assert centre > edge


def test_cheaper_than_uniform(result):
    n_uniform = int((END - START) / (0.002 * (END - START)))
    assert result.n_fom_solves < 0.5 * n_uniform


def test_snapshots_span_the_solution_manifold(result):
    solve = moving_pulse(start=START, end=END)
    test = np.column_stack([solve(s) for s in np.linspace(START, END, 97)])
    V, _ = build_pod_basis(result.snapshots)
    err = np.linalg.norm(test - V @ (V.T @ test)) / np.linalg.norm(test)
    assert err < 1e-2


def test_include_end_false_stays_inside():
    res = AdaMESS(moving_pulse(start=START, end=END), START, END,
                  include_end=False).run()
    assert res.positions[-1] < END


def test_max_solves():
    res = AdaMESS(moving_pulse(start=START, end=END), START, END, max_solves=7).run()
    assert res.n_fom_solves == 7


def test_diagnostics(result):
    assert len(result.diagnostics) == result.n_fom_solves - 9    # after warm-up
    keys = {"keep_ratio", "factor_keep", "factor_grad", "step_new", "rr", "epsilon", "pos"}
    assert keys <= set(result.diagnostics[0])


@pytest.mark.parametrize("kwargs", [
    {"start": 1.0, "end": 0.0},
    {"target_keep": 0.0},
    {"target_keep": 1.5},
    {"chunk_size": 2},
    {"step_min": 10.0, "step_max": 1.0},
])
def test_invalid_arguments(kwargs):
    args = {"start": START, "end": END, **kwargs}
    with pytest.raises(ValueError):
        AdaMESS(moving_pulse(), **args)


def test_options_run(result):
    for kw in ({"normalize_gradients": True}, {"abs_diffs": False}):
        res = AdaMESS(moving_pulse(start=START, end=END), START, END, **kw).run()
        assert res.positions[-1] == END


# ------------------------------------------------------------ primitives
def test_recurrence_matrix_from_columns(random_walk):
    R, eps, rr = recurrence_matrix_from_columns(random_walk, percentile=10.0)
    assert R.shape == (250, 250) and R.dtype == bool
    assert 0 < eps < 1
    assert rr == pytest.approx(0.10, abs=0.01)
    R0, _, _ = recurrence_matrix_from_columns(random_walk[:, :1])
    assert R0.shape == (0, 0)


def test_gradient_brake_never_expands():
    assert gradient_brake_normalized([1.0]) == 1.0
    assert gradient_brake_normalized([1, 1, 1, 0.5]) == 1.0
    assert gradient_brake_normalized([1, 1, 1, 4], beta_grad=1.0) == pytest.approx(0.3)
    assert gradient_brake_normalized([1, 1, 1, 2], beta_grad=1.0) == pytest.approx(0.5)
    assert gradient_brake_normalized([0, 0, 0, 0]) == 1.0


def test_step_update_symmetric():
    kw = dict(grad_norms=[1, 1, 1], target_keep=0.5, step_min=1, step_max=100,
              alpha_smooth=1.0, beta_keep=1.0)
    grow, _ = update_step_symmetric(10.0, [1, 0, 0, 0], **kw)      # redundant -> grow
    shrink, _ = update_step_symmetric(10.0, [1, 1, 1, 1], **kw)    # diverse -> shrink
    same, _ = update_step_symmetric(10.0, [1, 1, 0, 0], **kw)
    assert grow > 10.0 > shrink
    assert same == pytest.approx(10.0)
    clipped, _ = update_step_symmetric(10.0, [0, 0, 0, 0], **{**kw, "step_max": 15})
    assert clipped == 15


# ------------------------------------------------------------ POD / IO
def test_pod_basis_orthonormal(random_walk):
    V, sigma = build_pod_basis(random_walk)
    np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)
    assert np.all(np.diff(sigma) <= 0)
    V3, _ = build_pod_basis(random_walk, rank=3)
    assert V3.shape == (30, 3)
    Vbig, _ = build_pod_basis(random_walk, rank=10_000)
    assert Vbig.shape[1] == 30


def test_pod_basis_zero_data():
    V, sigma = build_pod_basis(np.zeros((5, 4)))
    assert V.shape == (5, 1) and np.all(sigma == 0)


def test_save_load_roundtrip(result, tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "run.h5"
    result.save(path)
    loaded = SamplingResult.load(path)
    assert loaded.variant == result.variant
    assert loaded.n_fom_solves == result.n_fom_solves
    np.testing.assert_allclose(loaded.positions, result.positions)
    np.testing.assert_allclose(loaded.step_history, result.step_history)
    np.testing.assert_array_equal(loaded.snapshots, result.snapshots)
    assert len(loaded.diagnostics) == len(result.diagnostics)
    assert loaded.diagnostics[3]["keep_ratio"] == pytest.approx(
        result.diagnostics[3]["keep_ratio"])
