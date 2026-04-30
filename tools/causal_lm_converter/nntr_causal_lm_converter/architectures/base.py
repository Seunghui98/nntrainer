"""Architecture builder base class.

A builder owns the per-architecture knowledge:
  * which INI sections to emit (graph topology)
  * which HF state_dict keys map to which NNTrainer-named tensors

By keeping the rest of the converter architecture-agnostic, adding a new model
family is mostly mechanical: subclass :class:`ArchitectureBuilder`, fill in
``build_sections`` and ``build_weight_bindings``, and register.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from ..ini import IniSection
from ..weights import WeightBinding


@dataclass
class BuildContext:
    """Inputs that all builders need.

    ``hf_config`` is a plain dict (from ``config.json``); we deliberately do not
    require a transformers ``PretrainedConfig`` instance so the converter can
    run on configs alone (useful for tests with synthetic models).
    """

    hf_config: Mapping[str, object]
    # Sequence length the runtime should advertise as the static input shape.
    init_seq_len: int = 8
    # Total tokens the model should be sized to handle (init + generation).
    max_seq_len: int = 8


class ArchitectureBuilder(ABC):
    """Per-architecture INI and weight-binding emitter."""

    #: HF ``model_type`` strings this builder claims (informational; the
    #: registry keys are the source of truth).
    model_type: str = ""

    def __init__(self, ctx: BuildContext) -> None:
        self.ctx = ctx
        self.cfg = ctx.hf_config

    # ----- public API ------------------------------------------------------
    @abstractmethod
    def build_sections(self) -> list[IniSection]:
        """Return the ordered list of INI sections (Model + layers)."""

    @abstractmethod
    def build_weight_bindings(self) -> list[WeightBinding]:
        """Return the HF -> NNTrainer name mapping for all parameters."""

    # ----- helpers used by subclasses --------------------------------------
    def _get(self, key: str, default=None):
        """Safe access to ``hf_config`` with a default."""
        return self.cfg.get(key, default) if hasattr(self.cfg, "get") else (
            getattr(self.cfg, key, default)
        )

    def _require(self, key: str):
        """Access required config keys with a clear error if absent."""
        v = self._get(key, None)
        if v is None:
            raise KeyError(f"hf_config missing required key {key!r}")
        return v
