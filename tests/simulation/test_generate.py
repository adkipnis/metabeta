from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from metabeta.simulation import Generator
from metabeta.utils.padding import maxShapes, aggregate


def make_cfg(**overrides: Any) -> argparse.Namespace:
    """
    Create a minimal argparse.Namespace compatible with Generator.
    """
    defaults = dict(
        # batch dimensions
        bs_train=32,
        bs_val=8,
        bs_test=8,
        bs_load=4,
        # partitions / sources
        d_tag="toy",
        partition="valid",
        begin=1,
        epochs=3,
        source="all",
        sgld=False,
        loop=True,  # default to loop in tests for speed/determinism
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_outdir_created(tmp_path: Path):
    cfg = make_cfg(partition="valid")
    outdir = tmp_path / "outputs" / "data"
    assert not outdir.exists()
    g = Generator(cfg, outdir)
    assert outdir.exists()
    assert outdir.is_dir()


def test_max_shapes_simple(tmp_path: Path):
    cfg = make_cfg()
    g = Generator(cfg, tmp_path)

    batch = [
        {"a": np.zeros((3, 2)), "b": np.zeros((5,))},
        {"a": np.zeros((4, 1)), "b": np.zeros((2,))},
        {"a": np.zeros((1, 7)), "b": np.zeros((9,))},
    ]
    shapes = maxShapes(batch)
    assert shapes["a"] == (4, 7)
    assert shapes["b"] == (9,)


def test_max_shapes_ndim_mismatch_raises(tmp_path: Path):
    """
    Current maxShapes asserts ndim consistency per key.
    If you later adopt the "align dims with ones" robust version, remove this test.
    """
    cfg = make_cfg()
    g = Generator(cfg, tmp_path)

    batch = [
        {"a": np.zeros((3, 2))},
        {"a": np.zeros((3, 2, 1))},  # ndim mismatch
    ]
    with pytest.raises(AssertionError, match="ndim mismatch"):
        _ = maxShapes(batch)


def test_aggregate_zero_padding(tmp_path: Path):
    cfg = make_cfg()
    g = Generator(cfg, tmp_path)

    batch = [
        {"x": np.ones((2, 3), dtype=np.float32), "y": np.array([1, 2], dtype=np.int64)},
        {"x": np.ones((1, 5), dtype=np.float32) * 2.0, "y": np.array([9], dtype=np.int64)},
    ]
    out = aggregate(batch)

    # shapes: x -> (bs, 2, 5), y -> (bs, 2)
    assert out["x"].shape == (2, 2, 5)
    assert out["y"].shape == (2, 2)

    # dataset 0: x occupies [:2, :3], rest zero
    assert np.allclose(out["x"][0, :2, :3], 1.0)
    assert np.allclose(out["x"][0, :2, 3:], 0.0)

    # dataset 1: x occupies [:1, :5], remaining rows zero
    assert np.allclose(out["x"][1, 0:1, :5], 2.0)
    assert np.allclose(out["x"][1, 1:, :], 0.0)

    # y padding
    assert np.array_equal(out["y"][0], np.array([1, 2]))
    assert np.array_equal(out["y"][1], np.array([9, 0]))


def test_gen_sizes_shapes_and_constraints(tmp_path: Path):
    """
    Verify that _genSizes returns arrays of correct shape and respects
    q <= d and min/max bounds for m and n (assuming truncLogUni round=True returns ints).
    """
    cfg = make_cfg(partition="train", max_d=6, max_q=10, min_m=2, max_m=7, min_n=3, max_n=11)
    g = Generator(cfg, tmp_path)

    main_seed = 2
    rng = np.random.default_rng(main_seed)
    n_datasets = 16
    mini_bs = 4

    d, q, m, ns = g._genSizes(rng, n_datasets=n_datasets, mini_batch_size=mini_bs)

    assert d.shape == (n_datasets,)
    assert q.shape == (n_datasets,)
    assert m.shape == (n_datasets,)
    assert ns.shape == (n_datasets, cfg.max_m)

    assert np.all(d >= 2) and np.all(d <= cfg.max_d)
    assert np.all(q >= 1)
    assert np.all(q <= d)
    assert np.all(m >= cfg.min_m) and np.all(m <= cfg.max_m)
    assert np.all(ns >= cfg.min_n) and np.all(ns <= cfg.max_n)


def test_genbatch_calls_genDataset_with_correct_args(monkeypatch, tmp_path: Path):
    """
    Verify that _genBatch:
    - produces the requested number of datasets
    - slices ns to m groups
    - enforces q <= d (via the args passed to _genDataset)
    This uses the real _genSizes and monkeypatches only _genDataset.
    """
    cfg = make_cfg(partition="train", max_d=6, max_q=10, min_m=2, max_m=7, min_n=3, max_n=11, loop=True)
    g = Generator(cfg, tmp_path)

    seen: list[dict[str, Any]] = []

    def fake_gen_dataset(cfg_arg, seedseq, d, q, ns_i):
        # record inputs
        seen.append(
            dict(
                d=int(d),
                q=int(q),
                m=int(len(ns_i)),
                ns=ns_i.astype(int).copy(),
                seedseq=seedseq,
            )
        )
        # variable shaped return
        n_total = int(np.sum(ns_i))
        return {
            "X": np.zeros((n_total, int(d)), dtype=np.float32),
            "groups": np.repeat(np.arange(len(ns_i), dtype=np.int64), ns_i),
            "theta": np.zeros((int(d) + int(q) + 1,), dtype=np.float32),
        }

    monkeypatch.setattr(Generator, "_genDataset", staticmethod(fake_gen_dataset))

    n_datasets = 16
    mini_bs = 4
    epoch = 2
    batch = g._genBatch(n_datasets=n_datasets, mini_batch_size=mini_bs, epoch=epoch)

    assert len(batch) == n_datasets
    assert len(seen) == n_datasets

    for rec in seen:
        d = rec["d"]
        q = rec["q"]
        m = rec["m"]
        ns_i = rec["ns"]

        assert 2 <= d <= cfg.max_d
        assert 1 <= q <= d
        assert cfg.min_m <= m <= cfg.max_m
        assert ns_i.shape == (m,)
        assert np.all(ns_i >= cfg.min_n)
        assert np.all(ns_i <= cfg.max_n)


def test_genbatch_deterministic_given_partition_and_epoch(monkeypatch, tmp_path: Path):
    """
    With _genDataset patched to not depend on anything except its inputs,
    _genBatch should be deterministic for a given (partition, epoch).
    """
    cfg = make_cfg(partition="train", max_d=6, max_q=4, min_m=2, max_m=6, min_n=3, max_n=9, loop=True)
    g = Generator(cfg, tmp_path)

    def run_once() -> list[tuple[int, int, int, tuple[int, ...]]]:
        seen: list[tuple[int, int, int, tuple[int, ...]]] = []

        def fake_gen_dataset(cfg_arg, seedseq, d, q, ns_i):
            seen.append((int(d), int(q), int(len(ns_i)), tuple(int(x) for x in ns_i)))
            return {"dummy": np.zeros((1,), dtype=np.float32)}

        monkeypatch.setattr(Generator, "_genDataset", staticmethod(fake_gen_dataset))
        _ = g._genBatch(n_datasets=12, mini_batch_size=3, epoch=1)
        return seen

    seen1 = run_once()
    seen2 = run_once()
    assert seen1 == seen2


@pytest.mark.parametrize("partition", ["train", "valid", "test"])
def test_seed_mapping_deterministic_within_partition(monkeypatch, tmp_path: Path, partition: str):
    """
    Confirms determinism within a partition for a fixed epoch argument.
    """
    cfg = make_cfg(partition=partition, max_d=10, max_q=5, min_m=2, max_m=8, min_n=3, max_n=12, epochs=10, loop=True)
    g = Generator(cfg, tmp_path)

    def run(epoch: int) -> list[int]:
        seen: list[int] = []

        def fake_gen_dataset(cfg_arg, seedseq, d, q, ns_i):
            seen.append(int(d))
            return {"dummy": np.zeros((1,), dtype=np.float32)}

        monkeypatch.setattr(Generator, "_genDataset", staticmethod(fake_gen_dataset))
        _ = g._genBatch(n_datasets=9, mini_batch_size=3, epoch=epoch)
        return seen

    a = run(epoch=3)
    b = run(epoch=3)
    assert a == b


def test_parallel_and_loop_produce_same_inputs(monkeypatch, tmp_path: Path):
    """
    Ensures the presampling + seedseq plumbing is consistent between loop and joblib paths.

    We don't compare full sampled datasets (heavy + stochastic); instead we patch _genDataset
    to return a signature derived only from inputs. Then loop vs parallel must match exactly.
    """
    base_cfg = make_cfg(partition="train", max_d=7, max_q=5, min_m=2, max_m=7, min_n=3, max_n=10, epochs=10)
    n_datasets = 24
    mini_bs = 4
    epoch = 3

    def fake_gen_dataset(cfg_arg, seedseq, d, q, ns_i):
        # signature purely from inputs
        return {
            "sig": np.array(
                [int(d), int(q), int(len(ns_i)), int(np.sum(ns_i))],
                dtype=np.int64,
            )
        }

    monkeypatch.setattr(Generator, "_genDataset", staticmethod(fake_gen_dataset))

    # loop
    cfg_loop = argparse.Namespace(**{**vars(base_cfg), "loop": True})
    g_loop = Generator(cfg_loop, tmp_path / "loop")
    loop_batch = g_loop._genBatch(n_datasets=n_datasets, mini_batch_size=mini_bs, epoch=epoch)
    loop_sigs = np.stack([ds["sig"] for ds in loop_batch], axis=0)

    # parallel
    cfg_par = argparse.Namespace(**{**vars(base_cfg), "loop": False})
    g_par = Generator(cfg_par, tmp_path / "par")
    par_batch = g_par._genBatch(n_datasets=n_datasets, mini_batch_size=mini_bs, epoch=epoch)
    par_sigs = np.stack([ds["sig"] for ds in par_batch], axis=0)

    assert np.array_equal(loop_sigs, par_sigs)
