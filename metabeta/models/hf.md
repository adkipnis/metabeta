# Hugging Face Hub checkpoint distribution

## Checkpoint structure

Three files hosted on HF Hub, one per likelihood family:

```
metabeta-normal.pt
metabeta-bernoulli.pt
metabeta-poisson.pt
```

Each file is a `torch.save` dict containing all four size variants for that family,
plus format metadata:

```python
{
    "_version": 1,           # JOINT_CHECKPOINT_VERSION — architecture format version
    "submodels": [...],      # same structure as the local joint checkpoint produced by pack.py
}
```

This matches the payload format already expected by `Api.__init__`, so no separate
unpacking step is needed.

## Versioning

The HF repo (`your-org/metabeta`) uses **git tags** to version checkpoints.
The tag represents the **architecture version** — it changes only when model
architecture changes make old weights incompatible with the current code.

`CHECKPOINT_VERSION` in `metabeta/utils/hub.py` holds the current tag (e.g. `"v1"`).
`hf_hub_download(..., revision=CHECKPOINT_VERSION)` pins downloads to that tag.

### Weight hotfix (no package release needed)

1. Push updated weights to HF Hub under the same filenames (new commit on `main`).
2. Move the tag to the new commit:

```python
from huggingface_hub import HfApi
api = HfApi()
api.delete_tag("your-org/metabeta", tag="v1")
api.create_tag("your-org/metabeta", tag="v1", revision="main")
```

Users automatically get the new weights on their next run: `hf_hub_download` resolves
the tag to a commit hash on each call, so a moved tag triggers a cache miss and
re-download.

### Architecture change (requires package release)

1. Push new checkpoints (new `JOINT_CHECKPOINT_VERSION` value inside the files).
2. Create a new tag (`v2`) — do **not** move the old one, so old package versions
   still resolve correctly.
3. Bump `CHECKPOINT_VERSION = "v2"` in `metabeta/utils/hub.py`.
4. Release a new package version.

**Always tag before releasing the package** so the tag and package version are in sync.

## Integration

`Api.from_pretrained(family)` is the user-facing entry point. It calls
`download_checkpoint(family)` from `metabeta/utils/hub.py`, then forwards the
local path to `Api.__init__`.

`huggingface_hub` is an optional dependency — it is only imported inside
`download_checkpoint`, with a clear `ImportError` if missing.
