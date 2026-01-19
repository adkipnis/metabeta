# metabeta
---
🚧 I'm currently refactoring this repo. For legacy code, switch to v1. 🚧
---
## A fast neural model for Bayesian mixed-effects regression
📈 approximates posterior estimation for hierarchical regression models\
💡 allows prior specification for each regression parameter\
👨🏻‍🏫 trained on simulated hierarchical datasets with realistic structure\
🚀 orders of magnitude faster than HCMC\
🔥 fully implemented in [PyTorch](https://pytorch.org/)

This repo contains the source code for [data simulation](metabeta/data), [model training](metabeta/models), and [evaluation](metabeta/evaluation).\
For details, please read our [preprint](https://doi.org/10.48550/arXiv.2510.07473).

## Overview
1. Hierarchical datasets are summarized locally (per group) and globally (across groups).
2. During training, a normalizing flow learns the forward mapping (regression parameters → base distribution) for both parameter types.
3. During inference, samples are drawn from the base distribution, and are mapped backward (regression parameters ← base distribution).

## Setup with [uv](https://docs.astral.sh/uv/)
- uv install
- source .venv/bin/activate
- uv develop

## Citing the project

```bibtex
@article{
  metabeta,
  author  = {Alex Kipnis and Marcel Binz and Eric Schulz},
  title   = {metabeta - A fast neural model for Bayesian mixed-effects regression},
  journal = {arXiv preprint arXiv:2510.07473},
  year    = {2025},
}
```
