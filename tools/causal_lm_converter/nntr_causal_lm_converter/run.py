"""End-to-end driver: prompt -> tokenize -> runner -> decode -> text.

Stitches together the three pieces the converter ships:

  1. ``transformers.AutoTokenizer`` to encode/decode (HF tokenizer files
     ship next to the model, so the user already has them).
  2. ``causal_lm_runner`` (built from ``runner/runner.cpp``) to run the
     forward pass and -- when ``--generate`` is set -- the greedy
     generation loop on top of the cached KV state.
  3. ``transformers.AutoTokenizer.decode`` to turn the runner's emitted
     token IDs back into readable text.

Generation lives inside the C++ runner, not in this driver: each
``incremental_inference`` call shares the model's KV cache with the
previous one, so we cannot loop from Python without losing it. The driver
spawns the runner exactly once per ``run`` invocation.

Typical usage:

  python -m nntr_causal_lm_converter.run \\
      --converted ./out                     \\
      --model-name qwen3-0.6b               \\
      --tokenizer ~/models/Qwen3-0.6B       \\
      --prompt "The capital of France is"   \\
      --generate 12

Outputs:

  prompt:        The capital of France is
  prompt_tokens: 791,6864,304,9822,374
  generated:     " Paris. It is also the most populous city in France."
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Iterable


def _find_runner(build_hint: str | None) -> str:
    """Locate the ``causal_lm_runner`` binary.

    Lookup order:
      1. ``--runner PATH`` (explicit, wins).
      2. ``$NNTRAINER_RUNNER`` env var.
      3. ``$NNTRAINER_BUILD_DIR/tools/causal_lm_converter/runner/causal_lm_runner``.
      4. ``./build`` and ``./builddir`` next to the converter package's repo
         root (two levels up from this file).
    """
    if build_hint and os.path.isfile(build_hint) and os.access(build_hint, os.X_OK):
        return build_hint
    env = os.environ.get("NNTRAINER_RUNNER")
    if env and os.path.isfile(env):
        return env
    env_build = os.environ.get("NNTRAINER_BUILD_DIR")
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    candidates = []
    if env_build:
        candidates.append(env_build)
    candidates.extend([
        os.path.join(repo_root, "build"),
        os.path.join(repo_root, "builddir"),
    ])
    for build in candidates:
        path = os.path.join(
            build, "tools", "causal_lm_converter", "runner", "causal_lm_runner")
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "causal_lm_runner not found. Build it with "
        "`ninja -C build tools/causal_lm_converter/runner/causal_lm_runner` "
        "or pass --runner /path/to/binary.")


def _resolve_artifacts(converted_dir: str, model_name: str | None):
    """Resolve INI / safetensors / nntr_config.json paths.

    If ``model_name`` is omitted we look for a single ``*.ini`` in the
    directory and infer the stem from it.
    """
    if not os.path.isdir(converted_dir):
        raise FileNotFoundError(converted_dir)
    if model_name is None:
        inis = [f for f in os.listdir(converted_dir) if f.endswith(".ini")]
        if not inis:
            raise FileNotFoundError(
                f"No .ini under {converted_dir}; pass --model-name explicitly")
        if len(inis) > 1:
            raise RuntimeError(
                f"Multiple .ini files under {converted_dir} ({inis}); "
                f"specify --model-name")
        model_name = inis[0][:-4]
    ini = os.path.join(converted_dir, f"{model_name}.ini")
    st = os.path.join(converted_dir, f"{model_name}.safetensors")
    cfg = os.path.join(converted_dir, "nntr_config.json")
    for path in (ini, st, cfg):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    return ini, st, cfg


# Match the runner's machine-readable line.  The runner emits exactly:
#   ``generated_tokens: 12,34,56``
_RX_GEN = re.compile(r"^generated_tokens:\s*([\d,]+)\s*$", re.MULTILINE)


def run(
    *,
    converted_dir: str,
    tokenizer_path: str,
    prompt: str,
    generate: int = 0,
    top_k: int = 0,
    model_name: str | None = None,
    runner_path: str | None = None,
    pad: bool = True,
    add_special_tokens: bool = True,
    extra_runner_args: Iterable[str] | None = None,
) -> dict:
    """End-to-end inference. Returns a dict with prompt/tokens/output strings.

    Args:
      converted_dir: directory with ``<name>.ini``, ``<name>.safetensors``,
          ``nntr_config.json``.
      tokenizer_path: HF model directory containing ``tokenizer.json``.
      prompt: text to tokenize.
      generate: number of tokens to greedily generate after the prompt.
      top_k: also ask the runner for top-K next-token candidates (independent
          of ``generate``).
      model_name: stem of the artifacts; auto-detected if exactly one ``.ini``
          lives in ``converted_dir``.
      runner_path: explicit path to ``causal_lm_runner``.
      pad: if True, right-pad the encoded prompt to ``init_seq_len`` (read
          from ``nntr_config.json``). The runner accepts shorter inputs but
          warns; padding makes the log cleaner.
      add_special_tokens: pass-through to the HF tokenizer.
      extra_runner_args: forwarded verbatim, e.g. ``["--print-shape"]``.
    """
    from transformers import AutoTokenizer
    import json

    runner_bin = _find_runner(runner_path)
    ini, st, cfg = _resolve_artifacts(converted_dir, model_name)

    with open(cfg) as f:
        rc = json.load(f)
    init_seq_len = int(rc.get("init_seq_len", 0)) or 0
    if init_seq_len <= 0:
        raise ValueError(
            f"init_seq_len missing from {cfg}; cannot size prompt buffer")

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    ids = tok.encode(prompt, add_special_tokens=add_special_tokens)
    real_prompt_len = len(ids)
    if real_prompt_len == 0:
        raise ValueError(
            f"Tokenizer at {tokenizer_path} produced 0 tokens for prompt "
            f"{prompt!r}; check tokenizer files.")
    if real_prompt_len > init_seq_len:
        sys.stderr.write(
            f"[run] truncating prompt {real_prompt_len} -> {init_seq_len} "
            f"tokens to fit the compiled init_seq_len\n")
        ids = ids[:init_seq_len]
        real_prompt_len = init_seq_len
    if pad and len(ids) < init_seq_len:
        ids = ids + [0] * (init_seq_len - len(ids))

    cmd = [runner_bin, ini, st, cfg,
           "--input-tokens", ",".join(str(i) for i in ids),
           "--prompt-len", str(real_prompt_len)]
    if top_k > 0:
        cmd += ["--top-k", str(top_k)]
    if generate > 0:
        cmd += ["--generate", str(generate)]
    if extra_runner_args:
        cmd += list(extra_runner_args)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    out = {
        "prompt": prompt,
        "prompt_tokens": ids[:real_prompt_len],
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }

    m = _RX_GEN.search(proc.stdout)
    if m:
        gen_ids = [int(x) for x in m.group(1).split(",") if x]
        out["generated_tokens"] = gen_ids
        out["generated_text"] = tok.decode(
            gen_ids, skip_special_tokens=True)
        out["full_text"] = tok.decode(
            ids[:real_prompt_len] + gen_ids, skip_special_tokens=True)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nntr_causal_lm_converter.run",
        description="Tokenize a prompt, run the converted model via "
                    "causal_lm_runner, and print the generated text.")
    p.add_argument("--converted", required=True,
                   help="Directory containing <name>.ini, <name>.safetensors, "
                        "and nntr_config.json (the output of "
                        "`python -m nntr_causal_lm_converter`).")
    p.add_argument("--tokenizer", required=True,
                   help="HF model directory holding tokenizer.json. Usually "
                        "the same path you passed to --model when converting.")
    p.add_argument("--prompt", required=True, help="Text to feed in.")
    p.add_argument("--generate", type=int, default=16,
                   help="Number of tokens to greedy-decode after the prompt.")
    p.add_argument("--top-k", type=int, default=0,
                   help="Also print top-K next-token candidates after the "
                        "initial forward (purely diagnostic).")
    p.add_argument("--model-name", default=None,
                   help="Stem of the artifacts; auto-detected when only one "
                        ".ini lives under --converted.")
    p.add_argument("--runner", default=None,
                   help="Explicit path to the causal_lm_runner binary.")
    p.add_argument("--no-pad", action="store_true",
                   help="Do NOT pad the prompt to init_seq_len.")
    p.add_argument("--no-special-tokens", action="store_true",
                   help="Do NOT prepend BOS/system tokens.")
    p.add_argument("--show-runner-output", action="store_true",
                   help="Echo the raw runner stdout/stderr.")
    args = p.parse_args(list(argv) if argv is not None else None)

    res = run(
        converted_dir=args.converted,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        generate=args.generate,
        top_k=args.top_k,
        model_name=args.model_name,
        runner_path=args.runner,
        pad=not args.no_pad,
        add_special_tokens=not args.no_special_tokens,
    )

    if args.show_runner_output:
        sys.stderr.write("=== runner stdout ===\n")
        sys.stderr.write(res["stdout"])
        if res["stderr"]:
            sys.stderr.write("=== runner stderr ===\n")
            sys.stderr.write(res["stderr"])
        sys.stderr.write("=== /runner ===\n")

    print(f"prompt:           {res['prompt']!r}")
    print(f"prompt_token_ids: {res['prompt_tokens']}")
    if "generated_tokens" in res:
        print(f"generated_tokens: {res['generated_tokens']}")
        print(f"generated_text:   {res['generated_text']!r}")
        print(f"full_text:        {res['full_text']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
