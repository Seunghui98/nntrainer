"""Tests for the small runtime-config emitter."""

from __future__ import annotations

import json

from nntr_causal_lm_converter.runtime_config import RuntimeConfig


def test_to_json_includes_required_keys():
    rc = RuntimeConfig(model_name="qwen3_tiny", architecture="qwen3")
    obj = json.loads(rc.to_json())
    # Keys consumed by the C++ runner.
    for key in (
        "batch_size",
        "init_seq_len",
        "max_seq_len",
        "model_tensor_type",
        "model_file_name",
    ):
        assert key in obj, f"missing {key}"
    assert obj["model_name"] == "qwen3_tiny"


def test_extra_fields_merge_into_top_level():
    rc = RuntimeConfig(extra={"my_extra": "abc"})
    obj = json.loads(rc.to_json())
    assert obj["my_extra"] == "abc"
