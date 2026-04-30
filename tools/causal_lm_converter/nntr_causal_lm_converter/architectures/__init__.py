"""Registry of supported causal-LM architectures.

To add a new architecture:
  1. Create a module under ``architectures/`` that subclasses
     :class:`base.ArchitectureBuilder`.
  2. Call ``register("model_type", YourBuilder)`` at import time.
  3. Import the module here so it self-registers.
"""

from __future__ import annotations

from .base import ArchitectureBuilder

_REGISTRY: dict[str, type[ArchitectureBuilder]] = {}


def register(name: str, cls: type[ArchitectureBuilder]) -> None:
    """Register an architecture builder under a HF ``model_type`` key."""
    _REGISTRY[name] = cls


def get_builder(model_type: str) -> type[ArchitectureBuilder]:
    """Return the builder class for a given HF ``model_type``.

    Raises ``KeyError`` with a list of supported architectures so the CLI can
    surface a helpful error.
    """
    if model_type not in _REGISTRY:
        raise KeyError(
            f"No builder registered for model_type={model_type!r}. "
            f"Supported: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model_type]


def list_architectures() -> list[str]:
    return sorted(_REGISTRY)


# Side-effect imports: each module registers itself on import.
from . import qwen3  # noqa: E402,F401
