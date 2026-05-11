"""Unit tests for the Python end-to-end driver (`nntr_causal_lm_converter.run`).

These tests intentionally avoid spawning the real runner: they substitute a
fake runner script that emits the same stdout shape so we can verify the
driver's tokenize -> spawn -> parse -> decode logic without a built binary
or a network-loaded HF tokenizer.

Heavier integration coverage that does exercise the real runner lives in
``test_build_and_run.py`` (which is automatically skipped when the runner
binary isn't built).
"""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest


def _make_fake_runner(tmp_path, output_lines: list[str]) -> str:
    """Write a shell stub that prints a canned stdout block.

    The driver only cares about the runner's stdout (it grep-parses for
    ``generated_tokens:``), so a script that prints the right lines is a
    perfect substitute for the real C++ binary.
    """
    path = tmp_path / "fake_runner.sh"
    body = "#!/bin/sh\n" + "".join(f'echo "{ln}"\n' for ln in output_lines)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def _make_fake_artifacts(tmp_path, init_seq_len: int = 8) -> tuple[str, str]:
    """Create a directory with the three artifact files the driver expects."""
    converted = tmp_path / "converted"
    converted.mkdir()
    (converted / "qwen3.ini").write_text("[Model]\nType = NeuralNetwork\n")
    (converted / "qwen3.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}")
    cfg = {
        "init_seq_len": init_seq_len,
        "max_seq_len": init_seq_len * 2,
        "batch_size": 1,
        "vocab_size": 100,
        "model_tensor_type": "FP32-FP32",
        "model_file_name": "qwen3.safetensors",
        "model_name": "qwen3",
    }
    (converted / "nntr_config.json").write_text(json.dumps(cfg))
    return str(converted), "qwen3"


def _make_fake_tokenizer(tmp_path):
    """Drop a tiny char-level HF tokenizer in a directory.

    Uses ``tokenizers.Tokenizer`` to save a real ``tokenizer.json``. This
    keeps the test independent of any model on the HF Hub.
    """
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[PAD]": 0, "hello": 1, "world": 2, "!": 3, "[UNK]": 4}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tok.save(str(tokenizer_dir / "tokenizer.json"))
    # transformers needs a tokenizer_config.json to load via AutoTokenizer.
    (tokenizer_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
    }))
    return str(tokenizer_dir)


def test_resolve_artifacts_autodetects_single_ini(tmp_path):
    from nntr_causal_lm_converter.run import _resolve_artifacts
    converted, name = _make_fake_artifacts(tmp_path)
    ini, st, cfg = _resolve_artifacts(converted, None)
    assert ini.endswith("qwen3.ini")
    assert st.endswith("qwen3.safetensors")
    assert cfg.endswith("nntr_config.json")


def test_resolve_artifacts_errors_on_multiple_inis(tmp_path):
    from nntr_causal_lm_converter.run import _resolve_artifacts
    converted, _ = _make_fake_artifacts(tmp_path)
    (tmp_path / "converted" / "other.ini").write_text("[Model]\nType=NeuralNetwork\n")
    with pytest.raises(RuntimeError, match="Multiple"):
        _resolve_artifacts(converted, None)


def test_run_driver_parses_generated_tokens(tmp_path):
    """Driver should pick up ``generated_tokens: ...`` from runner stdout
    and decode it back to text."""
    pytest.importorskip("transformers")
    from nntr_causal_lm_converter.run import run

    converted, _ = _make_fake_artifacts(tmp_path, init_seq_len=8)
    tokenizer_dir = _make_fake_tokenizer(tmp_path)
    # The fake tokenizer maps:  hello=1 world=2 != >  generated_tokens [2, 3].
    fake_runner = _make_fake_runner(tmp_path, [
        "[runner] ok",
        "generated_tokens: 2,3",
    ])

    res = run(
        converted_dir=converted,
        tokenizer_path=tokenizer_dir,
        prompt="hello",
        generate=2,
        runner_path=fake_runner,
        add_special_tokens=False,
    )
    assert res["prompt_tokens"] == [1]
    assert res["generated_tokens"] == [2, 3]
    # Decoded text passes through the tokenizer; the exact string depends
    # on its config, but it must be non-empty.
    assert isinstance(res["generated_text"], str)
    assert "world" in res["generated_text"] or "!" in res["generated_text"]


def test_run_driver_omits_generated_when_runner_skips_it(tmp_path):
    """If the runner runs without --generate, the driver still returns a
    result, just without the generation keys."""
    pytest.importorskip("transformers")
    from nntr_causal_lm_converter.run import run

    converted, _ = _make_fake_artifacts(tmp_path)
    tokenizer_dir = _make_fake_tokenizer(tmp_path)
    fake_runner = _make_fake_runner(tmp_path, [
        "[runner] ok (smoke only)",
    ])

    res = run(
        converted_dir=converted,
        tokenizer_path=tokenizer_dir,
        prompt="hello",
        generate=0,
        runner_path=fake_runner,
        add_special_tokens=False,
    )
    assert "generated_tokens" not in res
    assert res["prompt_tokens"] == [1]


def test_run_driver_truncates_long_prompts(tmp_path, capsys):
    """Prompts longer than init_seq_len get truncated with a warning."""
    pytest.importorskip("transformers")
    from nntr_causal_lm_converter.run import run

    # init_seq_len = 3 so a five-word prompt overflows.
    converted, _ = _make_fake_artifacts(tmp_path, init_seq_len=3)
    tokenizer_dir = _make_fake_tokenizer(tmp_path)
    fake_runner = _make_fake_runner(tmp_path, [
        "[runner] ok",
        "generated_tokens: 1",
    ])

    res = run(
        converted_dir=converted,
        tokenizer_path=tokenizer_dir,
        prompt="hello world ! hello world",  # 5 tokens; init_seq_len=3
        generate=1,
        runner_path=fake_runner,
        add_special_tokens=False,
    )
    captured = capsys.readouterr()
    assert "truncating" in captured.err
    assert len(res["prompt_tokens"]) == 3
