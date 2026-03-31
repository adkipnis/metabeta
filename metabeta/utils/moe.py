"""
Pseudo Mixture of Experts: permutation-based ensembling for amortized inference.

Creates k random permutations of a single dataset's feature columns, runs all
1+k views through the model in a single batched forward pass, then un-permutes
and joins the proposals.
"""

import numpy as np
import torch

from metabeta.models.approximator import Approximator
from metabeta.utils.evaluation import Proposal, joinProposals
from metabeta.utils.sampling import samplePermutation


def _invertPermutation(perm: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a permutation tensor."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), device=perm.device)
    return inv


def _permuteBatch(
    batch: dict[str, torch.Tensor],
    dperm: np.ndarray,
    qperm: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Clone a B=1 batch and apply feature permutations to model inputs."""
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

    # d-permute (fixed effects)
    dp = torch.as_tensor(dperm, dtype=torch.long)
    out['X'] = out['X'][..., dp]
    out['nu_ffx'] = out['nu_ffx'][..., dp]
    out['tau_ffx'] = out['tau_ffx'][..., dp]
    out['mask_d'] = out['mask_d'][..., dp]

    # q-permute (random effects)
    qp = torch.as_tensor(qperm, dtype=torch.long)
    out['Z'] = out['Z'][..., qp]
    out['tau_rfx'] = out['tau_rfx'][..., qp]
    out['mask_q'] = out['mask_q'][..., qp]
    out['mask_mq'] = out['mask_m'].unsqueeze(-1) & out['mask_q'].unsqueeze(-2)

    return out


def _unpermuteProposal(
    proposal: Proposal,
    dperm: torch.Tensor,
    qperm: torch.Tensor,
) -> None:
    """Un-permute proposal samples back to original feature ordering (in-place)."""
    inv_d = _invertPermutation(dperm)
    inv_q = _invertPermutation(qperm)

    d = proposal.d

    # global samples: (1, n_samples, D) where D = d + q [+ 1]
    g = proposal.data['global']['samples']
    ffx = g[..., :d][..., inv_d]
    rest = g[..., d:]
    if proposal.has_sigma_eps:
        sigma_rfx = rest[..., :-1][..., inv_q]
        sigma_eps = rest[..., -1:]
        rest = torch.cat([sigma_rfx, sigma_eps], dim=-1)
    else:
        rest = rest[..., inv_q]
    proposal.data['global']['samples'] = torch.cat([ffx, rest], dim=-1)

    # local samples: (1, m, n_samples, q)
    proposal.data['local']['samples'] = proposal.data['local']['samples'][..., inv_q]


def _stackBatches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack list of B=1 batch dicts into a single batch with B=len(batches)."""
    out = {}
    for key in batches[0]:
        values = [b[key] for b in batches]
        if torch.is_tensor(values[0]):
            out[key] = torch.cat(values, dim=0)
        else:
            out[key] = values[0]
    return out


def _splitProposal(proposal: Proposal, n: int) -> list[Proposal]:
    """Split a B=n proposal into n individual B=1 proposals."""
    proposals = []
    for i in range(n):
        proposed = {
            'global': {
                'samples': proposal.data['global']['samples'][i : i + 1],
                'log_prob': proposal.data['global']['log_prob'][i : i + 1],
            },
            'local': {
                'samples': proposal.data['local']['samples'][i : i + 1],
                'log_prob': proposal.data['local']['log_prob'][i : i + 1],
            },
        }
        proposals.append(Proposal(proposed, has_sigma_eps=proposal.has_sigma_eps))
    return proposals


def multiCheckpointEstimate(
    models: list[Approximator],
    batch: dict[str, torch.Tensor],
    n_samples: int,
    k: int = 0,
    rng: np.random.Generator | None = None,
) -> Proposal:
    """True MoE: run multiple checkpoints and join their proposals.

    Each model independently produces n_samples posterior samples for batch
    (optionally with k permuted views via pseudo-MoE). All proposals are
    concatenated, yielding len(models) * (1+k) * n_samples total samples.

    Args:
        models: Trained Approximator models in eval mode. Must share the same
            d_ffx and d_rfx dimensions (compatible with batch).
        batch: Batch of datasets. For k>0, must be B=1.
        n_samples: Posterior samples per model per view.
        k: Additional permuted views per model (0 = plain forward pass each).
        rng: Random number generator for permutations.

    Returns:
        Joined proposal with len(models) * (1+k) * n_samples total samples.
    """
    proposals = [moeEstimate(model, batch, n_samples, k, rng=rng) for model in models]
    return joinProposals(proposals)


def moeEstimate(
    model: Approximator,
    batch: dict[str, torch.Tensor],
    n_samples: int,
    k: int,
    rng: np.random.Generator | None = None,
) -> Proposal:
    """Pseudo-MoE inference: run 1+k permuted views and join proposals.

    Takes a single dataset (B=1), creates k random column permutations,
    stacks the original + k permuted copies into a (1+k)-sized batch,
    runs a single forward pass, then un-permutes and concatenates samples.

    Args:
        model: Trained Approximator in eval mode.
        batch: Single-dataset batch (B=1).
        n_samples: Number of posterior samples per view.
        k: Number of additional permuted views (0 = baseline, no permutation).
        rng: Random number generator for permutations.

    Returns:
        Joined proposal with (1+k) × n_samples total samples.
    """
    if k == 0:
        return model.estimate(batch, n_samples=n_samples)

    if rng is None:
        rng = np.random.default_rng(0)

    d = int(batch['mask_d'].shape[-1])
    q = int(batch['mask_q'].shape[-1])

    # generate k permutations
    dperms = [samplePermutation(rng, d) for _ in range(k)]
    qperms = [samplePermutation(rng, q) for _ in range(k)]

    # build 1+k batch: original + k permuted copies
    batches = [batch]
    for dp, qp in zip(dperms, qperms):
        batches.append(_permuteBatch(batch, dp, qp))

    stacked = _stackBatches(batches)

    # single forward pass
    proposal = model.estimate(stacked, n_samples=n_samples)

    # split and un-permute
    parts = _splitProposal(proposal, 1 + k)
    for i, (dp, qp) in enumerate(zip(dperms, qperms)):
        dp_t = torch.as_tensor(dp, dtype=torch.long)
        qp_t = torch.as_tensor(qp, dtype=torch.long)
        _unpermuteProposal(parts[i + 1], dp_t, qp_t)

    # join all proposals (concatenate along samples dimension)
    return joinProposals(parts)
