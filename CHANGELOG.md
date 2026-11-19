# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — unreleased

### Added
- `MESS.sampling()` accepts one of `eps`, `quantile`, `n_samples`,
  `keep_ratio` or `tol`; with no argument `eps` is chosen automatically at the
  knee of the "kept snapshots vs. eps" curve.
- `MESS.eps_sweep()` returns that curve; `MESS.resolve_eps()` exposes the eps
  resolution on its own.
- Attributes `MESS.eps_`, `MESS.mask_`, `MESS.indices_`.
- `distance_method='gram'|'pdist'` for `MESS`; the Gram variant is the new
  default and ~15× faster on the tutorial image (same selections).
- Complex-valued data in `MESS` and `AdaMESS`.
- `AdaMESS` options `abs_diffs`, `include_end`, `max_solves`; input validation.
- `SamplingResult.save/load` persist the per-window diagnostics.
- pytest suite, GitHub Actions CI, ruff configuration, `CITATION.cff`.
- Examples: eps selection, Burgers trajectory, rare events in sensor data,
  AdaMESS for a moving coil.

### Changed
- **`MESS.sampling()` without arguments now uses the automatic eps** instead
  of `eps=0.01`. Pass `eps=0.01` explicitly for the old behaviour.
- `AdaMESS` now computes a snapshot exactly at `end` (previously the last
  sample could fall up to one step short of the range end). Use
  `include_end=False` for the old behaviour.
- MESS and AdaMESS share one greedy core (`mess.greedy_mess`).
- `matplotlib` and `h5py` are optional extras instead of hard dependencies.

### Fixed
- Energy analysis on unsigned integer data (e.g. `uint8` images, as in
  tutorial 2) overflowed in `np.diff` and produced wrong transition energies.
- Constant data no longer produces `NaN` distances.
- `build_pod_basis` handles all-zero input and caps an explicit `rank`.
- Tutorial 3 crashed with a `KeyError` (`runs[0.75]`); tutorials 1/2 plotted
  the distance matrix where the recurrence matrix was labelled.
- Documentation stated that a lower `target_keep` leads to fewer snapshots;
  it is the opposite.
- Packaging: removed empty `setup.py`, stray `src/__init__.py` and an invalid
  hatch option.

## [0.1.0]
- Initial Python implementation of MESS and AdaMESS with tutorials.
