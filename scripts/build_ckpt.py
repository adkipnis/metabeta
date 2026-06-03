"""Build and publish pretrained joint checkpoints to Hugging Face Hub.

Usage
-----
# Build metabeta-{family}.pt files into OUTPUT_DIR:
uv run python scripts/build_ckpt.py --build

# Upload built files to HF Hub:
uv run python scripts/build_ckpt.py --upload

# Move the version tag on HF Hub to the current HEAD commit:
uv run python scripts/build_ckpt.py --tag

# Full pipeline in one go:
uv run python scripts/build_ckpt.py --build --upload --tag

Configuration
-------------
Fill in BEST_SEEDS below before running.  Set a seed to None to exclude that
(family, size) combination from the checkpoint (e.g. if training is not done).
A family whose every entry is None is skipped entirely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

CKPT_BASE = Path(__file__).parent.parent / 'metabeta' / 'outputs' / 'checkpoints'
OUTPUT_DIR = Path(__file__).parent.parent / 'metabeta' / 'outputs' / 'joint'

HF_REPO_ID = 'adkipnis/metabeta'
CHECKPOINT_VERSION = 'v1'  # HF Hub git tag — bump only on architecture changes

# Map (family, size) → best seed.  Set to None to skip that combo.
BEST_SEEDS: dict[tuple[str, str], int | None] = {
    ('normal', 'small'): 13,
    ('normal', 'medium'): 11, # cont.
    ('normal', 'large'): None,
    ('normal', 'huge'): None,
    ('bernoulli', 'small'): 6,
    ('bernoulli', 'medium'): 15, # cont.
    ('bernoulli', 'large'): 12, # cont.
    ('bernoulli', 'huge'): None,
    ('poisson', 'small'): 4,
    ('poisson', 'medium'): None,
    ('poisson', 'large'): None,
    ('poisson', 'huge'): None,
}

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

FAMILY_INITIAL = {'normal': 'n', 'bernoulli': 'b', 'poisson': 'p'}
FAMILIES = ('normal', 'bernoulli', 'poisson')
SIZES = ('small', 'medium', 'large', 'huge')


def _ckpt_dir(family: str, size: str, seed: int) -> Path:
    initial = FAMILY_INITIAL[family]
    return CKPT_BASE / f'data={size}-{initial}-mixed_model=large_seed={seed}'


def build(dry_run: bool = False) -> list[Path]:
    from metabeta.utils.api import joinCheckpoints

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    built: list[Path] = []

    for family in FAMILIES:
        checkpoints: dict[str, Path] = {}
        for size in SIZES:
            seed = BEST_SEEDS.get((family, size))
            if seed is None:
                continue
            path = _ckpt_dir(family, size, seed)
            if not path.exists():
                print(f'[ERROR] checkpoint not found: {path}', file=sys.stderr)
                sys.exit(1)
            checkpoints[f'{family}-{size}'] = path

        if not checkpoints:
            print(f'[SKIP]  {family} — no seeds configured')
            continue

        output_path = OUTPUT_DIR / f'metabeta-{family}.pt'
        print(f'[BUILD] {family}: {list(checkpoints)} → {output_path}')
        if not dry_run:
            joinCheckpoints(checkpoints, output_path=output_path)
        built.append(output_path)

    return built


def upload(dry_run: bool = False) -> None:
    from huggingface_hub import HfApi

    api = HfApi()

    model_card = Path(__file__).parent / 'hf_model_card.md'
    if model_card.exists():
        print(f'[UPLOAD] {model_card.name}  → {HF_REPO_ID}')
        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(model_card),
                path_in_repo='README.md',
                repo_id=HF_REPO_ID,
                repo_type='model',
                commit_message='update model card',
            )

    for family in FAMILIES:
        path = OUTPUT_DIR / f'metabeta-{family}.pt'
        if not path.exists():
            print(f'[SKIP]  {path.name} not found — run --build first')
            continue
        size_mb = path.stat().st_size / 1e6
        print(f'[UPLOAD] {path.name}  ({size_mb:.0f} MB)  → {HF_REPO_ID}')
        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=HF_REPO_ID,
                repo_type='model',
                commit_message=f'update {path.name}',
            )


def tag(dry_run: bool = False) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    print(f'[TAG]   moving {CHECKPOINT_VERSION!r} → HEAD on {HF_REPO_ID}')
    if not dry_run:
        try:
            api.delete_tag(HF_REPO_ID, tag=CHECKPOINT_VERSION, repo_type='model')
        except HfHubHTTPError:
            pass  # tag did not exist yet
        api.create_tag(
            HF_REPO_ID,
            tag=CHECKPOINT_VERSION,
            revision='main',
            repo_type='model',
        )
    print(f'[TAG]   done — {CHECKPOINT_VERSION} points to HEAD')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def setup() -> argparse.Namespace:
    # fmt: off
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build",    action="store_true", help="build joint checkpoint files")
    p.add_argument("--upload",   action="store_true", help="upload checkpoint files to HF Hub")
    p.add_argument("--tag",      action="store_true", help="move version tag on HF Hub to current HEAD")
    p.add_argument("--dry-run",  action="store_true", help="print actions without executing them")
    # fmt: on
    return p.parse_args()


def main() -> None:
    args = setup()

    if not (args.build or args.upload or args.tag):
        print('nothing to do — pass --build, --upload, and/or --tag', file=sys.stderr)
        sys.exit(1)

    if args.build:
        build(dry_run=args.dry_run)
    if args.upload:
        upload(dry_run=args.dry_run)
    if args.tag:
        tag(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
