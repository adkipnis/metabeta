# metabeta
## A fast neural model for Bayesian mixed-effects regression
📈 approximates posterior estimation for hierarchical regression models\
💡 allows prior specification for each regression parameter\
👨🏻‍🏫 trained on simulated hierarchical datasets with realistic structure\
🚀 orders of magnitude faster than HCMC\
🔥 fully implemented in [PyTorch](https://pytorch.org/)

This repo contains the source code for [data simulation](metabeta/data), [model training](metabeta/models), and [evaluation](metabeta/evaluation).\

## Overview
<img src="https://github.com/adkipnis/metabeta/blob/v1/figures/overview.png" width="750" />

- (A) _Dataset Simulation_: Sample priors, regression parameters, dependent predictors, noise, and perform a forward pass.
- (B) _Neural Model_:
  - (1) Hierarchical datasets are summarized locally (per group) and globally (across groups).
  - (2) During training, a normalizing flow learns the forward mapping (regression parameters → base distribution) for both parameter types.
  - (3) During inference, samples are drawn from the base distribution, and are mapped backward (regression parameters ← base distribution).
- _Example Posteriors_: Posterior KDEs of metabeta (MB) and Hamiltonian Monte Carlo (HMC) on a toy dataset.
- _Compute Time_: MB is orders of magnitude faster than HMC (on a Macbook M2 with realistic data).

## Setup with anaconda
- conda env create --file=env.yml
- conda activate mb
- pip install -e .
  
