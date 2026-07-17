from metabeta.simulation.generate import _generationSeed


def test_generation_seed_preserves_legacy_mapping():
    assert _generationSeed('train', 1) == 1
    assert _generationSeed('train', 9999) == 9999
    assert _generationSeed('valid') == 10_000
    assert _generationSeed('test') == 20_000


def test_generation_seed_offsets_high_training_epochs():
    assert _generationSeed('train', 10_000) == 30_000
    assert _generationSeed('train', 10_001) == 30_001
