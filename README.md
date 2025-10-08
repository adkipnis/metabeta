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
A. _Dataset Simulation_: Sample priors, regression parameters, dependent predictors, and noise, and perform a forward pass.\
B. _Neural Model_: Hierarchical datasets are summarized locally (per group) and globally (across groups). During training, the posterior networks learn the forward mapping
from the true regression parameters to a multivariate base distribution. During inference, we draw k samples from the base distribution, and apply the implicitly
learned backward mapping to them.\
C. _Example Posteriors_: Kernel density estimates from the posterior samples of metabeta (MB) and Hamiltonian Monte Carlo (HMC) on a toy dataset.\
D. _Compute Time_: For the test set, our model takes several orders of magnitude less time to compute in comparison to HMC (on a Macbook M2).

## Setup with anaconda
- conda env create --file=env.yml
- conda activate mb
- pip install -e .

## Citing the Project
To cite metabench in publications:

```bibtex
@article{metabench,
  author  = {Alex Kipnis and Marcel Binz and Eric Schulz},
  title   = {metabeta - A fast neural model for Bayesian mixed-effects regression},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2025},
}
```
