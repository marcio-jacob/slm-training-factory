"""
datasets/registry.py — Maps dataset names to loader functions.

Add new datasets here by registering them with @register_dataset.
The factory CLI uses the 'name' field from config YAML to look up loaders.
"""

from __future__ import annotations

from typing import Callable, Dict

_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str):
    """Decorator to register a dataset loader by name."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def load_dataset_by_name(name: str, **kwargs):
    """
    Load a dataset by its registered name.

    Args:
        name: Dataset name as defined in config YAML (e.g. 'wikipedia-pt')
        **kwargs: Forwarded to the loader (e.g. max_samples, split)

    Returns:
        HuggingFace DatasetDict or Dataset
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}\n"
            f"Register new datasets in datasets/registry.py"
        )
    return _REGISTRY[name](**kwargs)


def list_datasets() -> list[str]:
    return sorted(_REGISTRY.keys())


# Import modules so their @register_dataset decorators run
from data import portuguese  # noqa: E402, F401
from data import judicial    # noqa: E402, F401
