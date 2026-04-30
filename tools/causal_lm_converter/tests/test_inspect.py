"""Unit tests for the inspector.

These tests build minimal artifacts on the fly so they have no torch /
transformers dependency.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

from nntr_causal_lm_converter.inspect import (
    _parse_ini_sections,
    _read_safetensors_header,
    _safetensors_entries,
    main as inspect_main,
)
from nntr_causal_lm_converter.weights import write_safetensors


def _write_tiny_safetensors(path: str) -> None:
    """Write a small valid safetensors file using the converter's writer."""
    write_safetensors({
        "embedding0:Embedding": np.zeros((4, 8), dtype=np.float32),
        "layer0_wq:weight":     np.zeros((8, 8), dtype=np.float32),
        "layer0_attention_norm:gamma": np.zeros((8,), dtype=np.float32),
    }, path)


def test_safetensors_header_extracts_offsets(tmp_path):
    p = tmp_path / "m.safetensors"
    _write_tiny_safetensors(str(p))
    header = _read_safetensors_header(str(p))
    # All three tensors plus possibly __metadata__.
    names = [k for k in header if k != "__metadata__"]
    assert set(names) == {
        "embedding0:Embedding",
        "layer0_wq:weight",
        "layer0_attention_norm:gamma",
    }
    for name in names:
        assert "data_offsets" in header[name]
        assert header[name]["dtype"] == "F32"


def test_safetensors_entries_sorted_with_sizes(tmp_path):
    p = tmp_path / "m.safetensors"
    _write_tiny_safetensors(str(p))
    entries = _safetensors_entries(_read_safetensors_header(str(p)))
    # Sorted by name.
    names = [e["name"] for e in entries]
    assert names == sorted(names)
    # Sizes match expected (float32 = 4 bytes).
    by_name = {e["name"]: e for e in entries}
    assert by_name["embedding0:Embedding"]["size"] == 4 * 8 * 4
    assert by_name["layer0_attention_norm:gamma"]["size"] == 8 * 4


def test_inspect_safetensors_cli(tmp_path, capsys):
    p = tmp_path / "m.safetensors"
    _write_tiny_safetensors(str(p))
    rc = inspect_main(["safetensors", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "embedding0:Embedding" in out
    assert "F32" in out
    assert "tensors," in out  # the summary line


def test_inspect_safetensors_filter(tmp_path, capsys):
    p = tmp_path / "m.safetensors"
    _write_tiny_safetensors(str(p))
    inspect_main(["safetensors", str(p), "--filter", "wq"])
    out = capsys.readouterr().out
    assert "layer0_wq:weight" in out
    assert "embedding0:Embedding" not in out


def test_inspect_safetensors_tsv(tmp_path, capsys):
    p = tmp_path / "m.safetensors"
    _write_tiny_safetensors(str(p))
    inspect_main(["safetensors", str(p), "--tsv"])
    out = capsys.readouterr().out
    # First line is the header.
    assert out.splitlines()[0].split("\t") == ["name", "dtype", "shape", "size_bytes"]


def test_parse_ini_preserves_section_order(tmp_path):
    p = tmp_path / "m.ini"
    p.write_text(
        "[Model]\nType = NeuralNetwork\n\n"
        "[input0]\nType = input\ninput_shape = 1:1:8\n\n"
        "[layer0_wq]\nType = fully_connected\ninput_layers = embedding0\n"
    )
    sections = _parse_ini_sections(str(p))
    assert [s[0] for s in sections] == ["Model", "input0", "layer0_wq"]
    assert sections[2][1]["Type"] == "fully_connected"
    assert sections[2][1]["input_layers"] == "embedding0"


def test_inspect_ini_cli(tmp_path, capsys):
    p = tmp_path / "m.ini"
    p.write_text("[Model]\nType = NeuralNetwork\n[input0]\nType = input\n")
    rc = inspect_main(["ini", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Model" in out
    assert "NeuralNetwork" in out
    assert "input0" in out
