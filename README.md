# Maximum Entropy Snapshot Sampling

This repository provides a Python implementation of the method described in:

Kasolis, F., & Clemens, M. (2020). *Maximum Entropy Snapshot Sampling for Reduced Basis Generation*.  
[arXiv:2005.01280  [Titel anhand dieser ArXiv-ID in Citavi-Projekt übernehmen] ](https://arxiv.org/abs/2005.01280)

## Overview

Maximum Entropy Snapshot Sampling (MESS) is a method for selecting representative system states (snapshots) from structured data sequences. Originally developed for reduced basis model generation, the method is **general-purpose** and can be applied as a **preprocessing technique** to any kind of data series.

It is particularly useful in scenarios where:

- The dataset is large and potentially redundant
- Only a limited number of samples can be stored, processed, or used for training
- The goal is to retain maximum information with minimal sampling effort

MESS selects snapshots by maximizing the Shannon entropy of the projected states in a reduced space. This results in a compact, diverse, and information-rich subset of the original data.

## Features

- Maximum entropy-based snapshot selection
- **State-Recurrence Analysis** (default):
  - Filters out redundant or frequently recurring states
- **Energy Variation Analysis** (optional):
  - Prioritizes high-change or high-energy transitions
- Modular design allows use as a preprocessing step before:
  - Model reduction (e.g., POD, reduced basis)
  - Machine learning model training
  - Surrogate modeling
  - Data compression
  - Clustering or segmentation

## Typical Use Cases

- Parametric or time-dependent simulations (e.g., CFD, structural mechanics)
- Sensor data and time-series
- High-dimensional simulation trajectories
- Video frame selection and motion summarization
- Latent state analysis in autoencoders or other ML models

## Repository Structure