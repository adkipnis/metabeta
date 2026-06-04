__all__ = [
    'Prior',
    'hypersample',
    'bambiDefaultPriors',
    'Synthesizer',
    'Scammer',
    'Emulator',
    'Subsampler',
    'Simulator',
    'simulate',
    'Generator',
]

# submodule → names it exports; loaded lazily on first attribute access
_LAZY: dict[str, tuple[str, ...]] = {
    '.prior': ('Prior', 'hypersample', 'bambiDefaultPriors'),
    '.synthesizer': ('Synthesizer', 'Scammer'),
    '.emulator': ('Emulator', 'Subsampler'),
    '.simulator': ('Simulator', 'simulate'),
    '.generate': ('Generator',),
}

_NAME_TO_MOD = {name: mod for mod, names in _LAZY.items() for name in names}


def __getattr__(name: str):
    if name not in _NAME_TO_MOD:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    import importlib

    mod = importlib.import_module(_NAME_TO_MOD[name], package=__name__)
    val = getattr(mod, name)
    globals()[name] = val
    return val
