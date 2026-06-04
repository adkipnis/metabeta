"""Public package interface for metabeta."""

from typing import Any

__all__ = ['Api', 'RouterResult']


def __getattr__(name: str) -> Any:
    if name in __all__:
        from metabeta.models.api import Api, RouterResult

        return {'Api': Api, 'RouterResult': RouterResult}[name]
    raise AttributeError(f"module 'metabeta' has no attribute {name!r}")
