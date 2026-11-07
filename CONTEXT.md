# MetaBeta

Amortized Bayesian inference for generalized linear mixed models. This glossary fixes the terms
used when MetaBeta analyzes many simulated datasets to plan a study.

## Language

### Priors

**Training prior**:
The distribution over model structures, prior hyperparameters and datasets that a checkpoint's
networks were trained on; it bounds which analysis priors and designs the checkpoint supports.
_Avoid_: hyperprior, simulation prior

**Analysis prior**:
The prior under which a dataset is analyzed, given to MetaBeta at inference time.
_Avoid_: fitting prior, user prior

**Design prior**:
The distribution from which the true parameters of simulated datasets are drawn when planning a
study.
_Avoid_: sampling prior, data-generating prior

**Skeptical prior**:
An analysis prior on an effect, centered at 0 and narrow enough that the effect a study is
planned to detect is improbable before seeing data.
_Avoid_: null prior, conservative prior

### Design analysis

**Design**:
The layout of a planned study: the outcome family, the number of groups, the observations per
group, and which predictors vary within or between groups.
_Avoid_: scenario, setting

**Detection**:
The outcome of the decision rule on one dataset's posterior; by default, the 95% credible
interval of the effect excludes 0.
_Avoid_: significance, hit

**Power**:
The probability of a detection under a design at a fixed true effect.
_Avoid_: Bayesian power, sensitivity

**Assurance**:
The probability of a detection under a design, averaged over a design prior.
_Avoid_: expected power, probability of success
