"""Command-line entry: ``python -m nntr_causal_lm_converter ...``.

Glues together architecture builders, INI emission, weight materialization,
and safetensors writing. Designed for both library-style use (call
:func:`convert`) and command-line use.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Mapping, Optional

from . import get_builder, list_architectures
from .architectures.base import BuildContext
from .ini import render_ini
from .runtime_config import RuntimeConfig
from .weights import materialize, write_safetensors


def _load_hf_config(model_path: str) -> dict:
    """Load HF config.json from a local directory or repo path.

    We deliberately avoid importing ``transformers`` for the config-only path;
    a plain ``config.json`` is enough to emit the INI graph. Weight emission
    paths still pass an explicit ``state_dict`` through :func:`convert`.
    """
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No config.json under {model_path}")
    with open(cfg_path) as f:
        return json.load(f)


def convert(
    *,
    hf_config: Mapping[str, object],
    output_dir: str,
    model_name: str,
    init_seq_len: int = 8,
    max_seq_len: int = 8,
    state_dict: Optional[Mapping[str, object]] = None,
    dtype: str = "float32",
) -> dict[str, str]:
    """Run the conversion pipeline.

    Args:
      hf_config:    HuggingFace config dict (must contain ``model_type``).
      output_dir:   directory to write artifacts into (created if missing).
      model_name:   stem used for ``<model>.ini`` etc.
      init_seq_len: input sequence length compiled into the graph.
      max_seq_len:  maximum sequence length the runtime should expect.
      state_dict:   if provided, also writes ``<model>.safetensors``.
      dtype:        weight dtype (``float32`` or ``float16``).

    Returns:
      Mapping of artifact kind ('ini', 'safetensors', 'runtime_config') to
      absolute path. Keys are present only when the artifact was written.
    """
    model_type = hf_config.get("model_type")
    if not model_type:
        raise ValueError(
            f"hf_config missing 'model_type'; must be one of "
            f"{list_architectures()}"
        )

    builder_cls = get_builder(model_type)
    ctx = BuildContext(
        hf_config=hf_config,
        init_seq_len=init_seq_len,
        max_seq_len=max_seq_len,
    )
    builder = builder_cls(ctx)

    os.makedirs(output_dir, exist_ok=True)
    out: dict[str, str] = {}

    # 1. INI graph.
    ini_path = os.path.join(output_dir, f"{model_name}.ini")
    ini_text = render_ini(builder.build_sections())
    with open(ini_path, "w") as f:
        f.write(ini_text)
    out["ini"] = ini_path

    # 2. Weights (optional).
    if state_dict is not None:
        bindings = builder.build_weight_bindings()
        tensors = materialize(bindings, state_dict, dtype=dtype)
        st_path = os.path.join(output_dir, f"{model_name}.safetensors")
        write_safetensors(tensors, st_path)
        out["safetensors"] = st_path

    # 3. Runtime config.
    # We record ``vocab_size`` so the C++ runner can compute the last-token
    # logit slice for top-K decoding without having to re-parse the INI.
    vocab = int(hf_config.get("vocab_size", 0)) if hasattr(hf_config, "get") \
        else int(getattr(hf_config, "vocab_size", 0))
    rc = RuntimeConfig(
        batch_size=1,
        init_seq_len=init_seq_len,
        max_seq_len=max_seq_len,
        num_to_generate=max(0, max_seq_len - init_seq_len),
        vocab_size=vocab,
        embedding_dtype="FP32" if dtype == "float32" else "FP16",
        fc_layer_dtype="FP32" if dtype == "float32" else "FP16",
        lmhead_dtype="FP32" if dtype == "float32" else "FP16",
        model_file_name=f"{model_name}.safetensors",
        model_name=model_name,
        architecture=model_type,
    )
    rc_path = os.path.join(output_dir, "nntr_config.json")
    with open(rc_path, "w") as f:
        f.write(rc.to_json())
    out["runtime_config"] = rc_path

    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="nntr_causal_lm_converter",
        description="Convert HuggingFace causal-LM models to NNTrainer "
                    "INI + safetensors.",
    )
    p.add_argument("--model", required=True,
                   help="Path to a local model directory containing at least "
                        "config.json. If --weights is set, model.safetensors "
                        "or pytorch_model.bin is also required.")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--model-name", default=None,
                   help="Stem for output files (default: basename of --model).")
    p.add_argument("--init-seq-len", type=int, default=8,
                   help="Input sequence length compiled into the graph.")
    p.add_argument("--max-seq-len", type=int, default=8,
                   help="Maximum sequence length (init_seq_len + generation).")
    p.add_argument("--weights", action="store_true",
                   help="Also load and convert weights (requires torch + "
                        "transformers).")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="Weight dtype for the safetensors output.")
    args = p.parse_args(argv)

    hf_config = _load_hf_config(args.model)

    state_dict = None
    if args.weights:
        # Imported lazily so users that just want to dump INI need not install
        # the HF stack.
        from transformers import AutoModelForCausalLM
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32)
        model.eval()
        state_dict = model.state_dict()

    name = args.model_name or os.path.basename(os.path.normpath(args.model))
    written = convert(
        hf_config=hf_config,
        output_dir=args.output,
        model_name=name,
        init_seq_len=args.init_seq_len,
        max_seq_len=args.max_seq_len,
        state_dict=state_dict,
        dtype=args.dtype,
    )
    for k, v in written.items():
        print(f"  {k:<16} -> {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
