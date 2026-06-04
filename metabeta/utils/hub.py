"""Hugging Face Hub download helpers for pretrained checkpoints."""

from __future__ import annotations

from pathlib import Path

from metabeta.utils.constants import LIKELIHOOD_FAMILIES

HF_REPO_ID = 'adkipnis/metabeta'

# Architecture version tag on HF Hub. Bump only when model architecture changes
# make existing checkpoints incompatible. Weight-only hotfixes move this tag to a
# new commit without changing its name or requiring a package release.
CHECKPOINT_VERSION = 'v1'


def download_checkpoint(
    family: str,
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """Return the local path to the cached pretrained checkpoint for ``family``.

    Downloads from HF Hub on first call; subsequent calls return the cached path
    without a network round-trip (unless the tag has moved to a new commit).

    Parameters
    ----------
    family:
        Likelihood family — one of ``"normal"``, ``"bernoulli"``, ``"poisson"``.
    cache_dir:
        Override the default HF Hub cache (``~/.cache/huggingface/hub``).
    force_download:
        Re-download even if a cached copy exists.
    """
    if family not in LIKELIHOOD_FAMILIES:
        raise ValueError(f'family must be one of {LIKELIHOOD_FAMILIES}, got {family!r}')

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            'huggingface_hub is required for automatic checkpoint download. '
            'Install it with:  pip install huggingface_hub'
        ) from exc

    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f'metabeta-{family}.pt',
        revision=CHECKPOINT_VERSION,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        force_download=force_download,
    )
    return Path(local_path)
