"""Tiny INI emitter.

We avoid Python's `configparser` because NNTrainer's iniparser is sensitive to
ordering (sections appear in declaration order and so do properties). Using a
list-of-pairs representation keeps the emitted file stable and diff-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class IniSection:
    """A single INI section.

    NNTrainer treats the section name as the layer name and keys as layer
    properties (e.g. ``Type``, ``input_layers``, layer-specific params).
    """

    name: str
    # List preserves declaration order so ``input_layers = ...`` and other
    # ordering-sensitive keys end up where the runtime expects them.
    items: list[tuple[str, str]] = field(default_factory=list)
    comment: str | None = None

    def add(self, key: str, value: object) -> "IniSection":
        """Append ``key = value`` to the section, coercing ``value`` to str."""
        self.items.append((key, _format_value(value)))
        return self


def _format_value(v: object) -> str:
    """Render a Python value as an INI-friendly string.

    NNTrainer accepts plain ``true/false`` for bools and decimal numbers as-is.
    Lists become comma-separated strings (used by ``input_layers``).
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(_format_value(x) for x in v)
    if isinstance(v, float):
        # Avoid scientific notation that some parsers reject for small epsilons.
        return f"{v:.10g}"
    return str(v)


def render_ini(sections: Iterable[IniSection]) -> str:
    """Serialize a sequence of sections into the final INI text."""
    lines: list[str] = []
    for sec in sections:
        if sec.comment:
            for line in sec.comment.splitlines():
                lines.append(f"# {line}")
        lines.append(f"[{sec.name}]")
        for k, v in sec.items:
            lines.append(f"{k} = {v}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
