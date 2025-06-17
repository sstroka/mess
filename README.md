# Maximum Entropy Snapshot Sampling

This repository provides a Python implementation of **Maximum Entropy Snapshot Sampling**,  
originally proposed in:

> F. Kasolis and M. Clemens,  
> *Maximum Entropy Snapshot Sampling for Reduced Basis Generation*,  
> arXiv:2005.01280, 2020. 
> [https://arxiv.org/abs/2005.01280](https://arxiv.org/abs/2005.01280)

## Overview

**Maximum Entropy Snapshot Sampling (MESS)** is a method for selecting representative system states (snapshots) from structured data sequences. While originally developed for reduced basis model generation, the method is **general-purpose** and can be applied as a **preprocessing technique** to any application involving data sequences — including time-series or simulation data.

The method works by projecting the data into a reduced space and selecting snapshots that **maximize the $$\varepsilon$$-Frobenius entropy** of their distribution. This leads to a compact, diverse, and information-rich subset of the original data.


## MESS is particularly useful when:

- The dataset is large and contains redundancy  
- Only a limited number of samples can be stored or processed  
- Maximum information must be preserved with minimal samples


## Features

- Snapshot selection based on **state-recurrence analysis**  
  (identifies and filters redundant or frequently recurring system states)

- Optional [**energy-variation analysis**](https://iopscience.iop.org/article/10.1088/1742-6596/2090/1/012086)  
  (prioritizes high-change or high-energy transitions for enhanced diversity)

- Modular design enables integration into:
  - Model order reduction (e.g., POD, reduced basis methods)  
  - Machine learning pipelines  
  - Surrogate modeling  
  - Data compression  
  - Clustering or segmentation

## Typical Use Cases

- Parametric or time-dependent simulations (e.g., CFD, structural mechanics)  
- Sensor data and time-series  





