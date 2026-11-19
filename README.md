# MESS & AdaMESS — Maximum Entropy Snapshot Sampling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17101989.svg)](https://doi.org/10.5281/zenodo.17101989)
[![tests](https://github.com/sstroka/mess/actions/workflows/tests.yml/badge.svg)](https://github.com/sstroka/mess/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Python implementation of **Maximum Entropy Snapshot Sampling (MESS)** and its
adaptive extension **AdaMESS** for memory-efficient reduced-basis generation.

- **MESS** picks a compact, information-rich subset of columns from a data
  sequence (time series, simulation trajectory, image, …) by greedily
  maximizing the ε-Frobenius entropy. Recurring states are dropped.
- **AdaMESS** turns MESS into an *adaptive generator*: it decides where along a
  parameter axis the next expensive full-order solve should happen, so the
  dense uniform sweep never has to be computed.

📖 **Documentation, tutorials with results and examples: [project wiki](https://github.com/sstroka/mess/wiki)**

---

## Installation

```bash
git clone https://github.com/sstroka/mess.git
cd mess
pip install -e .            # core: numpy + scipy
pip install -e ".[io]"      # + h5py for SamplingResult.save/load
pip install -e ".[dev]"     # + everything for tests, examples and notebooks
```

Requires Python ≥ 3.11.

---

## Quick start

### MESS — filter an existing sequence

```python
import numpy as np
from mess import MESS

data = np.random.rand(500, 2000)       # (n_features, n_snapshots)
mess = MESS(data)

reduced = mess.sampling()              # eps chosen automatically
mess.eps_, mess.indices_               # threshold used, selected column indices
```

### Choosing `eps`

The recurrence threshold `eps` is the only parameter of MESS. Distances are
normalized by the *largest* distance, so an absolute value does not carry over
between data sets, and `'state'` and `'energy'` analysis need thresholds that
differ by orders of magnitude. `sampling()` therefore accepts **one** of:

| Call                              | Meaning                                                              |
|-----------------------------------|----------------------------------------------------------------------|
| `mess.sampling()`                 | **automatic**: knee of the curve "kept snapshots vs. eps" (default)  |
| `mess.sampling(eps=0.05)`         | absolute threshold on the normalized distances (classic behaviour)   |
| `mess.sampling(quantile=0.02)`    | eps = 2 % quantile of all pairwise distances, scale-free             |
| `mess.sampling(n_samples=100)`    | eps chosen so that ~100 snapshots are kept (budget)                  |
| `mess.sampling(keep_ratio=0.1)`   | eps chosen so that ~10 % of the snapshots are kept                   |
| `mess.sampling(tol=1e-3)`         | largest eps whose selection reproduces the data to rel. error ≤ 1e-3 |

`mess.eps_sweep()` returns the whole curve (eps, quantile, number kept, knee)
for plotting. See [`examples/01_choosing_eps.py`](examples/01_choosing_eps.py).

> **Tip:** On noisy data the automatic mode can be generous, because noise
> makes many states look new. `tol=...` or `n_samples=...` give tighter control.

### AdaMESS — adaptive snapshot generation

```python
from mess import AdaMESS, build_pod_basis

def solve(s):
    # compute and return the full-order solution vector at parameter s
    return phi

ada = AdaMESS(solve, start=S0, end=S1, target_keep=0.85, beta_grad=1.0)
res = ada.run()                                 # adaptive snapshot generation
V, sigma = build_pod_basis(res.snapshots)       # POD basis from the samples
res.save("run.h5")                              # needs the [io] extra
```

**Key knobs**

| Parameter              | Symbol  | Role                         | Effect                                        |
|------------------------|---------|------------------------------|-----------------------------------------------|
| `target_keep`          | $\zeta$ | target MESS keep ratio       | higher ⇒ larger steps, fewer solves           |
| `beta_keep`            |         | keep-ratio exponent          | higher ⇒ stronger reaction to redundancy      |
| `beta_grad`            | $\beta$ | gradient-brake strength      | higher ⇒ stronger slow-down at fast changes   |
| `chunk_size`           |         | sliding-window length        | larger ⇒ smoother, laggier adaptation         |
| `step_min`/`step_max`  |         | step bounds                  | safety rails (default 0.2 % / 5 % of span)    |
| `epsilon_percentile`   |         | recurrence threshold         | percentile of window distances (scale-free)   |

Complex-valued solutions (e.g. time-harmonic fields) are supported.

---

## Examples and notebooks

| File | Topic |
|------|-------|
| [`examples/01_choosing_eps.py`](examples/01_choosing_eps.py) | all eps modes compared, `eps_sweep` curve |
| [`examples/02_burgers_snapshots.py`](examples/02_burgers_snapshots.py) | MESS before POD on a Burgers shock trajectory vs. uniform/random subsampling |
| [`examples/03_rare_events_time_series.py`](examples/03_rare_events_time_series.py) | sensor time series: MESS keeps a rare fault that uniform sampling misses |
| [`examples/04_adamess_moving_coil.py`](examples/04_adamess_moving_coil.py) | AdaMESS for a coil passing over an observation region, with diagnostics |
| [`notebooks/tutorial_1.ipynb`](notebooks/tutorial_1.ipynb) | MESS for image compression (state analysis) |
| [`notebooks/tutorial_2.ipynb`](notebooks/tutorial_2.ipynb) | MESS with energy analysis |
| [`notebooks/tutorial_3_adamess.ipynb`](notebooks/tutorial_3_adamess.ipynb) | AdaMESS end-to-end: adaptive sampling, POD basis, benchmark |

Results and figures of all notebooks and examples are discussed in the
[wiki](https://github.com/sstroka/mess/wiki).

The examples write their figures to `examples/output/`:

```bash
pip install -e ".[examples]"
python examples/02_burgers_snapshots.py
```

---

## Development

```bash
pip install -e ".[dev]"
pytest                 # unit tests + doctests
ruff check .           # lint
```

CI runs the test suite on Python 3.11–3.14 (see `.github/workflows/tests.yml`).
Changes are listed in [`CHANGELOG.md`](CHANGELOG.md).

---

## Typical use cases

- Parametric or time-dependent simulations (CFD, structural mechanics,
  electromagnetics)
- Low-frequency human-exposure dosimetry and reduced-order modelling
- Sensor data and time series
- Surrogate modelling, data compression, clustering / segmentation

---

## Code contributors

| Name | Role | Affiliation |
|------|------|-------------|
| **Steven Stroka** | Lead author, AdaMESS implementation, maintainer | Chair of Electromagnetic Theory (TET), University of Wuppertal |
| *Fotios Kasolis* | Original MESS method | — |
| *Markus Clemens* | Original MESS method, supervision | Chair of Electromagnetic Theory (TET), University of Wuppertal |

<!-- TODO: add further contributors before publishing -->

---

## Citation

If you use this software, please cite the original MESS paper, the AdaMESS
paper and the software archive (see also [`CITATION.cff`](CITATION.cff)).

```bibtex
@article{kasolis2020mess,
  title   = {Maximum Entropy Snapshot Sampling for Reduced Basis Generation},
  author  = {Kasolis, Fotios and Clemens, Markus},
  journal = {arXiv preprint arXiv:2005.01280},
  year    = {2020},
  url     = {https://arxiv.org/abs/2005.01280}
}

@inproceedings{stroka2026adamess,
  title     = {Adaptive Maximum Entropy Snapshot Sampling for Memory Efficient
               Reduced-Basis Generation},
  author    = {Stroka, Steven and <co-authors>},
  booktitle = {IEEE Conference on Electromagnetic Field Computation (CEFC)},
  year      = {2026},
  address   = {Thessaloniki, Greece}
}

@software{mess_software,
  author    = {Stroka, Steven},
  title     = {{MESS \& AdaMESS}: Maximum Entropy Snapshot Sampling for
               Memory Efficient Reduced-Basis Generation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17101989},
  url       = {https://doi.org/10.5281/zenodo.17101989}
}
```

### Related work

- M. W. F. M. Bannenberg, F. Kasolis, M. Günther, M. Clemens,
  *Maximum entropy snapshot sampling for reduced basis modelling*,
  COMPEL, vol. 41, no. 3, 2022.
- *Efficient Low-Frequency Human Exposure Assessment With the Maximum Entropy
  Snapshot Sampling*, IEEE, 2024.

---

## License

Released under the MIT License. Copyright (c) 2025 Steven Stroka. See
[`LICENSE`](LICENSE).
