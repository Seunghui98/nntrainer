"""End-to-end integration test: convert a tiny synthetic Qwen3 -> run via the
C++ runner -> verify the forward pass produces finite logits.

Skipped automatically when:
  * torch / transformers are unavailable, or
  * the runner binary has not been built (build it with
    ``ninja -C <build_dir> tools/causal_lm_converter/runner/causal_lm_runner``).

The build directory is auto-discovered: we look for ``build/``, ``builddir/``
under the repo root, and respect the ``NNTRAINER_BUILD_DIR`` env var.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


# Discover the runner binary so the test does not require manual configuration.
def _find_binary(repo_root: str, name: str) -> str | None:
    env = os.environ.get("NNTRAINER_BUILD_DIR")
    candidates = []
    if env:
        candidates.append(env)
    candidates.extend([
        os.path.join(repo_root, "build"),
        os.path.join(repo_root, "builddir"),
    ])
    for build in candidates:
        path = os.path.join(
            build, "tools", "causal_lm_converter", "runner", name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


@pytest.fixture(scope="module")
def runner_path(repo_root):
    p = _find_binary(repo_root, "causal_lm_runner")
    if not p:
        pytest.skip(
            "causal_lm_runner binary not built; run "
            "`ninja -C build tools/causal_lm_converter/runner/causal_lm_runner`"
        )
    return p


@pytest.fixture(scope="module")
def example_embedded_path(repo_root):
    p = _find_binary(repo_root, "causal_lm_example_embedded")
    if not p:
        pytest.skip("causal_lm_example_embedded binary not built")
    return p


@pytest.fixture(scope="module")
def torch_available():
    try:
        import torch  # noqa: F401
        from transformers import Qwen3Config, AutoModelForCausalLM  # noqa: F401
    except ImportError:
        pytest.skip("torch/transformers required for end-to-end test")


def _build_tiny_qwen3_state_dict(cfg_dict: dict):
    """Instantiate a real HF Qwen3 model with random weights."""
    from transformers import Qwen3Config, AutoModelForCausalLM
    import torch  # noqa: F401

    # Strip keys we added that HF's config does not understand.
    keys = set(Qwen3Config().to_dict().keys())
    filtered = {k: v for k, v in cfg_dict.items() if k in keys}
    cfg = Qwen3Config(**filtered)
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return cfg.to_dict(), model.state_dict()


def test_convert_and_run_tiny_qwen3(
    tmp_path, tiny_qwen3_cfg, runner_path, torch_available,
):
    """Full pipeline: convert -> load INI + safetensors -> forward -> finite."""
    hf_cfg, sd = _build_tiny_qwen3_state_dict(tiny_qwen3_cfg)
    # The HF config dict from to_dict() is the source of truth from here on.
    from nntr_causal_lm_converter.cli import convert
    out = convert(
        hf_config=hf_cfg,
        output_dir=str(tmp_path),
        model_name="qwen3_tiny",
        init_seq_len=8,
        max_seq_len=16,
        state_dict=sd,
        dtype="float32",
    )
    for kind in ("ini", "safetensors", "runtime_config"):
        assert os.path.isfile(out[kind]), f"missing {kind}: {out[kind]}"
        assert os.path.getsize(out[kind]) > 0

    proc = subprocess.run(
        [runner_path, out["ini"], out["safetensors"], out["runtime_config"]],
        capture_output=True, text=True, timeout=120,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, (
        f"runner failed (exit {proc.returncode}):\n{combined}"
    )
    # Acceptance criteria from the runner itself.
    assert "[runner] OK" in proc.stdout, combined
    assert "first 8 logits" in proc.stdout, combined


def test_runner_no_forward_smoke(
    tmp_path, tiny_qwen3_cfg, runner_path, torch_available,
):
    """Loading-only path (validates INI + safetensors header alignment)."""
    hf_cfg, sd = _build_tiny_qwen3_state_dict(tiny_qwen3_cfg)
    from nntr_causal_lm_converter.cli import convert
    out = convert(
        hf_config=hf_cfg,
        output_dir=str(tmp_path),
        model_name="qwen3_load_only",
        init_seq_len=4,
        max_seq_len=8,
        state_dict=sd,
        dtype="float32",
    )
    proc = subprocess.run(
        [runner_path, out["ini"], out["safetensors"], out["runtime_config"],
         "--no-forward"],
        capture_output=True, text=True, timeout=60,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, combined
    assert "[runner] OK (no forward)" in proc.stdout


def test_runner_top_k_and_input_tokens(
    tmp_path, tiny_qwen3_cfg, runner_path, torch_available,
):
    """End-to-end: feed explicit token IDs, ask for top-K next-token logits.

    Validates that:
      * ``--input-tokens`` reaches the embedding layer (no shape errors)
      * ``--print-shape`` reports a finite, vocab-sized buffer
      * ``--top-k`` prints N rank lines that all decode as token IDs within
        the model's vocab.
    """
    hf_cfg, sd = _build_tiny_qwen3_state_dict(tiny_qwen3_cfg)
    from nntr_causal_lm_converter.cli import convert
    out = convert(
        hf_config=hf_cfg, output_dir=str(tmp_path), model_name="qwen3_topk",
        init_seq_len=8, max_seq_len=16, state_dict=sd, dtype="float32",
    )

    # Feed deterministic token IDs that stay within vocab.
    vocab = int(hf_cfg["vocab_size"])
    tokens = ",".join(str(i % vocab) for i in range(8))

    proc = subprocess.run(
        [runner_path, out["ini"], out["safetensors"], out["runtime_config"],
         "--input-tokens", tokens,
         "--print-shape",
         "--top-k", "3"],
        capture_output=True, text=True, timeout=120,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, combined
    assert "fed 8 tokens" in proc.stdout, combined
    assert f"vocab={vocab}" in proc.stdout, combined
    # Top-3 must produce three rank lines, each with a valid token_id.
    import re
    ranks = re.findall(r"rank \d+: token_id=(\d+)\s+logit=([-\d\.eE]+)",
                       proc.stdout)
    assert len(ranks) == 3, f"expected 3 top-K rows, got {ranks!r}"
    for tid, _ in ranks:
        assert 0 <= int(tid) < vocab, f"token_id {tid} out of vocab {vocab}"


def test_runner_generate_produces_n_tokens(
    tmp_path, tiny_qwen3_cfg, runner_path, torch_available,
):
    """End-to-end: ``--generate N`` should print exactly N token IDs and
    they must all be within vocab. This exercises the C++ generation loop's
    KV-cache continuity across steps.
    """
    hf_cfg, sd = _build_tiny_qwen3_state_dict(tiny_qwen3_cfg)
    from nntr_causal_lm_converter.cli import convert
    out = convert(
        hf_config=hf_cfg, output_dir=str(tmp_path), model_name="qwen3_gen",
        # init_seq_len bigger than prompt_len so there's room for generation
        # positions inside the compiled graph.
        init_seq_len=16, max_seq_len=32, state_dict=sd, dtype="float32",
    )

    vocab = int(hf_cfg["vocab_size"])
    prompt_len = 4
    tokens = ",".join(str(i % vocab) for i in range(prompt_len))
    n_generate = 6

    proc = subprocess.run(
        [runner_path, out["ini"], out["safetensors"], out["runtime_config"],
         "--input-tokens", tokens,
         "--prompt-len", str(prompt_len),
         "--generate", str(n_generate)],
        capture_output=True, text=True, timeout=120,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, combined

    # Parse `generated_tokens: 12,34,...` line.
    import re
    m = re.search(r"generated_tokens:\s*([\d,]+)", proc.stdout)
    assert m is not None, f"missing generated_tokens line:\n{combined}"
    gen = [int(x) for x in m.group(1).split(",") if x]
    assert len(gen) == n_generate, (
        f"expected {n_generate} tokens, got {len(gen)}: {gen}")
    for tid in gen:
        assert 0 <= tid < vocab, f"token_id {tid} out of vocab {vocab}"


def test_example_embedded_consumes_same_artifacts(
    tmp_path, tiny_qwen3_cfg, example_embedded_path, torch_available,
):
    """The embedded example program must accept the same INI + safetensors
    files as the runner. This is the contract we promise to downstream
    applications: link against ``causal_lm_importer_dep`` and you can load
    any converter output with three lines of code.
    """
    hf_cfg, sd = _build_tiny_qwen3_state_dict(tiny_qwen3_cfg)
    from nntr_causal_lm_converter.cli import convert
    out = convert(
        hf_config=hf_cfg,
        output_dir=str(tmp_path),
        model_name="qwen3_emb_example",
        init_seq_len=4,
        max_seq_len=8,
        state_dict=sd,
        dtype="float32",
    )
    proc = subprocess.run(
        [example_embedded_path, out["ini"], out["safetensors"],
         out["runtime_config"]],
        capture_output=True, text=True, timeout=60,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, combined
    assert "loaded ok" in proc.stdout, combined
