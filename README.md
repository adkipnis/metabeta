# metabeta
## A fast neural model for Bayesian mixed-effects regression
📈 approximates Bayesian posterior computation for hierarchical regression models\
⛓️‍💥 allows prior specification for each regression parameter\
🧮 trained on simulated hierarchical datasets with realistic structure\
🚀 runs orders of magnitude faster than HCMC

This repo contains the source code for [data simulation](metabeta/data), [model training](metabeta/models), and [evaluation](metabeta/evaluation).\
For details, please read our [preprint](TODO). Our model is built in [PyTorch](https://pytorch.org/).

## Overview
<img src="https://github.com/adkipnis/metabeta/blob/main/figures/overview.png" width="750" />

- _Dataset Simulation_: Sample priors, regression parameters, dependent predictors, and noise, and perform a forward pass.
- _Neural Model_: Hierarchical datasets are summarized locally (per group) and globally (across groups). During training, a normalizing flow learns the forward mapping (true parameters -> base distribution).
During inference, samples are drawn from the base distribution, and are passed backwards through the normalizing flow.
- _Example Posteriors_: Posterior KDEs of metabeta (MB) and Hamiltonian Monte Carlo (HMC) on a toy dataset.
- _Compute Time_: MB is orders of magnitude faster than HMC (on a Macbook M2 with realistic data).

## Setup with anaconda
- conda env create --file=env.yml
- conda activate mb
- pip install -e .

## Citing the project

```bibtex
@article{metabench,
  author  = {Alex Kipnis and Marcel Binz and Eric Schulz},
  title   = {metabeta - A fast neural model for Bayesian mixed-effects regression},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2025},
}
```
