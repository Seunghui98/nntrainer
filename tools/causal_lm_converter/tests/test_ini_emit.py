"""Unit tests for the INI emitter.

These tests stay pure-Python and do not require torch / transformers, so they
run quickly and catch graph-shape regressions before any C++ build.
"""

from __future__ import annotations

import re

from nntr_causal_lm_converter.architectures.base import BuildContext
from nntr_causal_lm_converter.architectures.qwen3 import Qwen3Builder
from nntr_causal_lm_converter.ini import IniSection, render_ini


def _section_names(sections):
    return [s.name for s in sections]


def _ini_dict(sections):
    """Return a {section_name: {key: value}} mapping for assertions."""
    return {s.name: dict(s.items) for s in sections}


# --- generic INI emitter -------------------------------------------------
def test_render_handles_lists_and_bools():
    s = IniSection("foo")
    s.add("Type", "addition")
    s.add("input_layers", ["a", "b"])
    s.add("disable_bias", True)
    txt = render_ini([s])
    assert "Type = addition" in txt
    assert "input_layers = a,b" in txt
    assert "disable_bias = true" in txt


def test_render_preserves_section_order():
    a = IniSection("first")
    a.add("Type", "input")
    b = IniSection("second")
    b.add("Type", "rms_norm")
    txt = render_ini([a, b])
    assert txt.index("[first]") < txt.index("[second]")


# --- Qwen3 graph emission ------------------------------------------------
def test_qwen3_emits_expected_block_layout(tiny_qwen3_cfg):
    ctx = BuildContext(hf_config=tiny_qwen3_cfg, init_seq_len=4, max_seq_len=8)
    sections = Qwen3Builder(ctx).build_sections()
    names = _section_names(sections)

    # Mandatory header sections.
    assert names[:3] == ["Model", "input0", "embedding0"]
    assert names[-2:] == ["output_norm", "lm_head"]

    # Each block must contain the canonical 14 layers in order.
    expected_per_block = [
        "attention_norm",
        "wv", "wk", "wq",
        "k_norm", "q_norm",
        "attention", "attention_out",
        "residual_attn",
        "ffn_norm",
        "ffn_up", "ffn_gate", "ffn_swiglu", "ffn_down",
        "residual_ffn",
    ]
    for i in range(tiny_qwen3_cfg["num_hidden_layers"]):
        block = [n for n in names if n.startswith(f"layer{i}_")]
        assert block == [f"layer{i}_{suffix}" for suffix in expected_per_block]


def test_qwen3_attention_inputs_in_qkv_order(tiny_qwen3_cfg):
    """mha_core's input_layers must be [q_normed, k_normed, v]."""
    ctx = BuildContext(hf_config=tiny_qwen3_cfg)
    d = _ini_dict(Qwen3Builder(ctx).build_sections())
    assert d["layer0_attention"]["input_layers"] == \
        "layer0_q_norm,layer0_k_norm,layer0_wv"


def test_qwen3_residual_paths_link_correctly(tiny_qwen3_cfg):
    ctx = BuildContext(hf_config=tiny_qwen3_cfg)
    d = _ini_dict(Qwen3Builder(ctx).build_sections())
    # First residual: block input (= embedding0) + attention_out.
    assert d["layer0_residual_attn"]["input_layers"] == \
        "embedding0,layer0_attention_out"
    # Second residual: post-attention residual + ffn_down.
    assert d["layer0_residual_ffn"]["input_layers"] == \
        "layer0_residual_attn,layer0_ffn_down"
    # Layer 1's attention_norm reads from layer 0's final residual.
    assert d["layer1_attention_norm"]["input_layers"] == "layer0_residual_ffn"


def test_qwen3_tied_embedding_marks_lm_head(tiny_qwen3_cfg):
    """When tie_word_embeddings is True, the lm_head must reuse the embed.

    Also validates the ``unit`` property is set on the lm_head section: this
    is what switches the ``tie_word_embeddings`` C++ layer into "lm_head"
    matrix-multiply mode rather than the embedding lookup mode.
    """
    ctx = BuildContext(hf_config=tiny_qwen3_cfg)
    d = _ini_dict(Qwen3Builder(ctx).build_sections())
    assert d["embedding0"]["Type"] == "tie_word_embeddings"
    assert "unit" not in d["embedding0"], (
        "embedding section must NOT set unit (would flip layer mode)")
    assert d["lm_head"]["Type"] == "tie_word_embeddings"
    assert d["lm_head"]["shared_from"] == "embedding0"
    assert d["lm_head"]["unit"] == str(tiny_qwen3_cfg["vocab_size"])
    assert d["lm_head"]["disable_bias"] == "true"


def test_qwen3_untied_embedding_uses_plain_fc(tiny_qwen3_cfg):
    tiny_qwen3_cfg["tie_word_embeddings"] = False
    ctx = BuildContext(hf_config=tiny_qwen3_cfg)
    d = _ini_dict(Qwen3Builder(ctx).build_sections())
    assert d["embedding0"]["Type"] == "embedding_layer"
    assert d["lm_head"]["Type"] == "fully_connected"
    assert d["lm_head"]["unit"] == str(tiny_qwen3_cfg["vocab_size"])


def test_qwen3_attention_dimensions_use_gqa(tiny_qwen3_cfg):
    """K/V projection units must follow num_key_value_heads * head_dim."""
    ctx = BuildContext(hf_config=tiny_qwen3_cfg)
    d = _ini_dict(Qwen3Builder(ctx).build_sections())
    expected_kv_unit = tiny_qwen3_cfg["num_key_value_heads"] * \
        tiny_qwen3_cfg["head_dim"]
    expected_q_unit = tiny_qwen3_cfg["num_attention_heads"] * \
        tiny_qwen3_cfg["head_dim"]
    assert d["layer0_wk"]["unit"] == str(expected_kv_unit)
    assert d["layer0_wv"]["unit"] == str(expected_kv_unit)
    assert d["layer0_wq"]["unit"] == str(expected_q_unit)


def test_render_ini_is_valid_text(tiny_qwen3_cfg):
    """Smoke check: every emitted line is a comment, blank, header, or k=v."""
    ctx = BuildContext(hf_config=tiny_qwen3_cfg)
    text = render_ini(Qwen3Builder(ctx).build_sections())
    pattern = re.compile(r"^(#.*|\[[^\]]+\]|\s*|[A-Za-z_][A-Za-z_0-9]*\s*=\s*.+)$")
    for line in text.splitlines():
        assert pattern.match(line), f"unexpected INI line: {line!r}"
