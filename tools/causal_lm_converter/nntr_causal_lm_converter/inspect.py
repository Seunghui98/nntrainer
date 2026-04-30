"""Inspection helpers for converter artifacts.

Two CLI subcommands:

  * ``inspect-safetensors <path>`` — dump tensor names, dtypes, shapes,
    sizes. Useful for diffing two builds, verifying weight coverage, or
    checking that NNTrainer-style ``"<layer>:<role>"`` names line up with the
    INI graph.

  * ``inspect-ini <path>`` — list every section, its declared layer ``Type``,
    and inbound connections. Works without loading any model.

Both subcommands print human-readable tables by default and machine-friendly
TSV when ``--tsv`` is passed. They have no torch / transformers dependency
(``inspect-ini`` is std-lib only; ``inspect-safetensors`` only needs the
``safetensors`` package).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from typing import Iterable


# ---- safetensors ---------------------------------------------------------
def _read_safetensors_header(path: str) -> dict:
    """Parse a safetensors file header without materializing tensor data.

    Format (per the official spec):
      [8 bytes little-endian uint64: header_size][header_size bytes JSON]
      [tensor data...]

    We deliberately avoid the ``safetensors`` package here so the inspector
    stays usable in lean environments and gives identical output regardless
    of dtype support.
    """
    with open(path, "rb") as f:
        sz_bytes = f.read(8)
        if len(sz_bytes) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        (header_size,) = struct.unpack("<Q", sz_bytes)
        header_json = f.read(header_size).decode("utf-8")
    return json.loads(header_json)


def _safetensors_entries(header: dict) -> list[dict]:
    """Flatten the safetensors header into a list of {name, dtype, shape, size}."""
    out = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        offsets = meta.get("data_offsets", [0, 0])
        size = int(offsets[1]) - int(offsets[0])
        out.append({
            "name": name,
            "dtype": meta.get("dtype", "?"),
            "shape": meta.get("shape", []),
            "size": size,
        })
    out.sort(key=lambda e: e["name"])
    return out


def _print_safetensors(entries: list[dict], *, tsv: bool) -> None:
    if tsv:
        print("name\tdtype\tshape\tsize_bytes")
        for e in entries:
            print(f"{e['name']}\t{e['dtype']}\t{','.join(map(str, e['shape']))}\t{e['size']}")
        return
    name_w = max((len(e["name"]) for e in entries), default=4)
    name_w = max(name_w, 4)
    total = sum(e["size"] for e in entries)
    print(f"{'NAME':<{name_w}}  DTYPE  SHAPE                     SIZE")
    for e in entries:
        shape = "x".join(str(s) for s in e["shape"]) or "scalar"
        print(f"{e['name']:<{name_w}}  {e['dtype']:<5}  {shape:<24}  {e['size']:>12}")
    print(f"\n{len(entries)} tensors, {total:,} bytes total "
          f"({total / 1024 / 1024:.2f} MiB)")


def cmd_inspect_safetensors(args: argparse.Namespace) -> int:
    header = _read_safetensors_header(args.path)
    entries = _safetensors_entries(header)
    if args.filter:
        entries = [e for e in entries if args.filter in e["name"]]
    _print_safetensors(entries, tsv=args.tsv)
    return 0


# ---- INI inspection ------------------------------------------------------
def _parse_ini_sections(path: str) -> list[tuple[str, dict[str, str]]]:
    """Tiny INI parser: returns ``[(section, {key: value})]`` in order.

    We keep our own parser instead of ``configparser`` for two reasons:
      1. NNTrainer's loader is order-sensitive; we want to display exactly
         the order the runtime will see.
      2. We avoid case-folding so layer names round-trip exactly.
    """
    sections: list[tuple[str, dict[str, str]]] = []
    current: tuple[str, dict[str, str]] | None = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                if current is not None:
                    sections.append(current)
                current = (line[1:-1], {})
                continue
            if "=" in line and current is not None:
                k, _, v = line.partition("=")
                current[1][k.strip()] = v.strip()
    if current is not None:
        sections.append(current)
    return sections


def cmd_inspect_ini(args: argparse.Namespace) -> int:
    sections = _parse_ini_sections(args.path)
    if args.filter:
        sections = [s for s in sections if args.filter in s[0]]

    if args.tsv:
        print("section\ttype\tinput_layers")
        for name, props in sections:
            print(f"{name}\t{props.get('Type', '')}\t{props.get('input_layers', '')}")
        return 0

    name_w = max((len(n) for n, _ in sections), default=8)
    type_w = max((len(p.get("Type", "")) for _, p in sections), default=4)
    print(f"{'SECTION':<{name_w}}  {'TYPE':<{type_w}}  INPUT_LAYERS")
    for name, props in sections:
        print(f"{name:<{name_w}}  {props.get('Type', ''):<{type_w}}  "
              f"{props.get('input_layers', '')}")
    print(f"\n{len(sections)} sections in {args.path}")
    return 0


# ---- entry point ---------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nntr_causal_lm_converter.inspect",
        description="Inspect converter artifacts (.ini / .safetensors).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_st = sub.add_parser("safetensors",
                          help="Dump tensors in a .safetensors file.")
    p_st.add_argument("path")
    p_st.add_argument("--filter", default=None,
                      help="Substring filter on tensor names.")
    p_st.add_argument("--tsv", action="store_true",
                      help="Tab-separated output (machine readable).")
    p_st.set_defaults(func=cmd_inspect_safetensors)

    p_ini = sub.add_parser("ini", help="Dump sections in an .ini file.")
    p_ini.add_argument("path")
    p_ini.add_argument("--filter", default=None,
                       help="Substring filter on section names.")
    p_ini.add_argument("--tsv", action="store_true")
    p_ini.set_defaults(func=cmd_inspect_ini)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
