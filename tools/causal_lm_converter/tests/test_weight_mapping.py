"""Unit tests for HF -> NNTrainer weight name mapping and materialization."""

from __future__ import annotations

import numpy as np
import pytest

from nntr_causal_lm_converter.architectures.base import BuildContext
from nntr_causal_lm_converter.architectures.qwen3 import Qwen3Builder
from nntr_causal_lm_converter.weights import (
    WeightBinding,
    materialize,
)


def _bindings_dict(b):
    return {x.hf_key: x for x in b}


def test_qwen3_bindings_cover_one_block(tiny_qwen3_cfg):
    """Every HF parameter in a single decoder block must have a binding."""
    bindings = Qwen3Builder(BuildContext(hf_config=tiny_qwen3_cfg)) \
        .build_weight_bindings()
    by_hf = _bindings_dict(bindings)

    expected_hf_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
    ]
    for k in expected_hf_keys:
        assert k in by_hf, f"missing binding for {k!r}"


def test_qwen3_fc_weights_are_marked_for_transpose(tiny_qwen3_cfg):
    """All FC-style projections must request transposition."""
    bindings = Qwen3Builder(BuildContext(hf_config=tiny_qwen3_cfg)) \
        .build_weight_bindings()
    by_hf = _bindings_dict(bindings)
    for k in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]:
        assert by_hf[k].transpose, f"{k} should be transposed"
    # Norms and embedding are not transposed.
    assert not by_hf["model.norm.weight"].transpose
    assert not by_hf["model.embed_tokens.weight"].transpose


def test_qwen3_tied_omits_lm_head_binding(tiny_qwen3_cfg):
    bindings = Qwen3Builder(BuildContext(hf_config=tiny_qwen3_cfg)) \
        .build_weight_bindings()
    assert "lm_head.weight" not in _bindings_dict(bindings)


def test_qwen3_untied_includes_lm_head_binding(tiny_qwen3_cfg):
    tiny_qwen3_cfg["tie_word_embeddings"] = False
    bindings = Qwen3Builder(BuildContext(hf_config=tiny_qwen3_cfg)) \
        .build_weight_bindings()
    by_hf = _bindings_dict(bindings)
    assert "lm_head.weight" in by_hf
    assert by_hf["lm_head.weight"].nntr_key == "lm_head:weight"
    assert by_hf["lm_head.weight"].transpose


def test_materialize_transposes_2d_tensor():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = materialize(
        [WeightBinding("k", "x:weight", transpose=True)],
        {"k": src})
    assert out["x:weight"].shape == (4, 3)
    np.testing.assert_array_equal(out["x:weight"], src.T)


def test_materialize_dtype_cast():
    src = np.ones((2, 2), dtype=np.float64)
    out = materialize(
        [WeightBinding("k", "x:weight", transpose=False)],
        {"k": src}, dtype="float16")
    assert out["x:weight"].dtype == np.float16


def test_materialize_raises_on_missing_keys():
    with pytest.raises(KeyError, match="missing"):
        materialize([WeightBinding("absent", "x:weight")], {})


def test_materialize_rejects_transpose_of_non_2d():
    with pytest.raises(ValueError, match="non-2D"):
        materialize(
            [WeightBinding("k", "x:weight", transpose=True)],
            {"k": np.zeros((3,), dtype=np.float32)})
