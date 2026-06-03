# Maximum Entropy Snapshot Sampling (MESS & AdaMESS)

This repository provides a Python implementation of **Maximum Entropy Snapshot
Sampling (MESS)** and its adaptive extension **AdaMESS** for memory-efficient
reduced-basis generation.

MESS was originally proposed in:

> F. Kasolis and M. Clemens,
> *Maximum Entropy Snapshot Sampling for Reduced Basis Generation*,
> arXiv:2005.01280, 2020.
> [https://arxiv.org/abs/2005.01280](https://arxiv.org/abs/2005.01280)

The adaptive variant **AdaMESS** is introduced in:

> S. Stroka, *et al.*,
> *Adaptive Maximum Entropy Snapshot Sampling for Memory Efficient
> Reduced-Basis Generation*, IEEE CEFC 2026.

---

## Overview

**MESS** selects representative system states (snapshots) from structured data
sequences by greedily maximizing the $\varepsilon$-Frobenius entropy of their
distribution. While originally developed for reduced-basis model generation, the
method is **general-purpose** and works as a **preprocessing technique** for any
application involving data sequences — time-series, simulation trajectories, or
images.

**AdaMESS** turns MESS from a *filter on an existing sequence* into an *adaptive
generator*. Instead of sampling a pre-computed dense data matrix, it decides
**where along a parameter axis the next expensive full-order snapshot should be
computed**. This avoids ever materializing the dense uniform sequence — the key
to its memory savings.

The method walks a 1-D parameter range and keeps a sliding window of recent
snapshots. From that window it forms

- a **recurrence / MESS keep-ratio** (how redundant is the local data?), and
- a **gradient brake** $\|P_{i+1}-P_i\|$
  (how fast does the solution change?),

and combines them into a **symmetric step-size controller**: a single keep-ratio
exponent shrinks the step where the data is too diverse and grows it where the
data is redundant. A POD basis is then built directly from the sampled
snapshots.

---

## Installation

```bash
git clone https://github.com/<your-org>/mess.git
cd mess
pip install -e .
```

Requires Python >= 3.11 and `numpy`, `scipy`, `matplotlib` (plus `h5py` for the
`SamplingResult.save/load` helpers).

---

## Quick start

### MESS (filter an existing sequence)

```python
import numpy as np
from mess import MESS

data = np.random.rand(100, 500)          # (n_features, n_snapshots)
mess = MESS(data)
reduced = mess.sampling(eps=0.2)         # information-rich subset
```

### AdaMESS (adaptive snapshot generation)

```python
from mess import AdaMESS, build_pod_basis

def solve(s):
    # compute and return the full-order solution vector at parameter s
    return phi

ada  = AdaMESS(solve, start=S0, end=S1, target_keep=0.85, beta_grad=1.0)
res  = ada.run()                                # adaptive snapshot generation
V, sigma = build_pod_basis(res.snapshots)       # POD basis from the samples
```

**Key knobs**

| Parameter            | Symbol   | Role                       | Effect                                   |
|----------------------|----------|----------------------------|------------------------------------------|
| `target_keep`        | $\zeta$  | MESS keep ratio            | lower ⇒ fewer snapshots, more aggressive |
| `beta_keep`          |          | keep-ratio exponent        | higher ⇒ stronger reaction to redundancy |
| `beta_grad`          | $\beta$  | gradient-brake strength    | higher ⇒ stronger slow-down at fast changes |
| `chunk_size`         |          | sliding-window length      | larger ⇒ smoother, laggier adaptation    |
| `step_min`/`step_max`|          | step bounds                | safety rails (default 0.2 % / 5 % of span) |

---

## Notebooks

| Notebook | Topic |
|----------|-------|
| `notebooks/tutorial_1.ipynb`         | MESS for data compression (image example) |
| `notebooks/tutorial_2.ipynb`         | MESS state vs. energy analysis |
| `notebooks/tutorial_3_adamess.ipynb` | **AdaMESS** end-to-end: adaptive sampling, POD basis, benchmark |

The AdaMESS tutorial is fully self-contained — it uses a cheap synthetic field
solver so it runs anywhere. Replace `solve(s)` with your own full-order solver
to apply it to a real problem.

---

## Features

- **MESS** snapshot selection via state-recurrence analysis (identifies and
  filters redundant or frequently recurring states).
- Optional **energy-variation analysis** (prioritizes high-change transitions).
- **AdaMESS** adaptive snapshot generation with symmetric step control
  (keep-ratio + gradient brake); solver-agnostic via a user-supplied
  `solve(s)` callable.
- Adaptive snapshot generation followed by a POD basis built directly from the
  samples.
- `SamplingResult` container with HDF5 save/load.

## Typical use cases

- Parametric or time-dependent simulations (CFD, structural mechanics,
  electromagnetics)
- Low-frequency human-exposure dosimetry and reduced-order modelling
- Sensor data and time-series
- Surrogate modelling, data compression, clustering / segmentation

---

## Code contributors

| Name | Role | Affiliation |
|------|------|-------------|
| **Steven Stroka** | Lead author, AdaMESS implementation, maintainer | Chair of Electromagnetic Theory (TET), University of Wuppertal |
| *Fotios Kasolis* | Original MESS method | — |
| *Markus Clemens* | Original MESS method, supervision | Chair of Electromagnetic Theory (TET), University of Wuppertal |
| *<add contributor>* | *<role>* | *<affiliation>* |

> _Please update this table with the actual contributors and roles before
> publishing the repository._

---

## Citation

If you use this software, please cite both the original MESS paper and the
AdaMESS paper, and the software archive.

### Original MESS method

```bibtex
@article{kasolis2020mess,
  title   = {Maximum Entropy Snapshot Sampling for Reduced Basis Generation},
  author  = {Kasolis, Fotios and Clemens, Markus},
  journal = {arXiv preprint arXiv:2005.01280},
  year    = {2020},
  url     = {https://arxiv.org/abs/2005.01280}
}
```

### AdaMESS

```bibtex
@inproceedings{stroka2026adamess,
  title     = {Adaptive Maximum Entropy Snapshot Sampling for Memory Efficient
               Reduced-Basis Generation},
  author    = {Stroka, Steven and <co-authors>},
  booktitle = {IEEE Conference on Electromagnetic Field Computation (CEFC)},
  year      = {2026},
  address   = {Thessaloniki, Greece}
}
```

### Software (Zenodo)

```bibtex
@software{mess_software,
  author    = {Stroka, Steven and <contributors>},
  title     = {{MESS \& AdaMESS}: Maximum Entropy Snapshot Sampling for
               Memory Efficient Reduced-Basis Generation},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.1.0},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17101989.svg)](https://doi.org/10.5281/zenodo.17101989)



---

## Related work

- M. W. F. M. Bannenberg, F. Kasolis, M. Günther, M. Clemens,
  *Maximum entropy snapshot sampling for reduced basis modelling*,
  COMPEL, vol. 41, no. 3, 2022.
- *Efficient Low-Frequency Human Exposure Assessment With the Maximum Entropy
  Snapshot Sampling*, IEEE, 2024.

---

## License

Released under the MIT License. Copyright (c) 2025 Steven Stroka. See
[`LICENSE`](LICENSE) for details.
