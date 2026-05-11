"""Tokenization helper: prompt -> token IDs in a form ``causal_lm_runner``
consumes.

The converter itself does not embed a tokenizer (causal-LM tokenizers are
model-family-specific and pull heavy dependencies). Instead, this small CLI
loads the HF tokenizer that lives next to the model, encodes a prompt, and
prints the resulting integer IDs.

Pipeline:

  python -m nntr_causal_lm_converter.tokenize \\
      --model ~/models/Qwen3-0.6B \\
      --prompt "Hello, world."          \\
      --pad-to 128
  # 9707,11,1879,13,0,0,...,0

Then feed it to the runner:

  ./causal_lm_runner model.ini model.safetensors nntr_config.json \\
      --input-tokens "9707,11,1879,..."   \\
      --top-k 5

Or pipe it via the ``--input-tokens-file`` flag:

  python -m nntr_causal_lm_converter.tokenize ... --pad-to 128 > /tmp/ids.txt
  ./causal_lm_runner ... --input-tokens-file /tmp/ids.txt --top-k 5

The same script can also decode token IDs back to text (handy for examining
the runner's top-K output).
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable


def _load_tokenizer(model_path: str):
    """Lazily load the HF tokenizer.

    We import inside the function so that the rest of the converter package
    does not pay the ``transformers`` import cost when tokenization is not
    requested.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def cmd_encode(args: argparse.Namespace) -> int:
    """``encode``: prompt -> token IDs."""
    tok = _load_tokenizer(args.model)
    ids = tok.encode(args.prompt, add_special_tokens=not args.no_special_tokens)
    if args.pad_to > 0 and len(ids) < args.pad_to:
        # Most causal LM inference frontends right-pad with 0 (the runner does
        # this too if a shorter token list is provided). Doing it here makes
        # the runner log cleaner.
        ids = ids + [0] * (args.pad_to - len(ids))
    elif args.pad_to > 0 and len(ids) > args.pad_to:
        sys.stderr.write(
            f"[tokenize] truncating {len(ids)} -> {args.pad_to} tokens\n")
        ids = ids[: args.pad_to]
    print(",".join(str(i) for i in ids))
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """``decode``: token IDs -> text (useful to read top-K results)."""
    tok = _load_tokenizer(args.model)
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        ids = [int(x) for x in sys.stdin.read().split() if x.strip()]
    text = tok.decode(ids, skip_special_tokens=args.skip_special_tokens)
    print(text)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nntr_causal_lm_converter.tokenize",
        description="Encode/decode prompts using the HF tokenizer next to a "
                    "converted model. Output format matches what "
                    "causal_lm_runner --input-tokens expects.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode",
                         help="Prompt -> comma-separated token IDs.")
    enc.add_argument("--model", required=True,
                     help="Path to the HF model directory containing "
                          "tokenizer.json / tokenizer_config.json.")
    enc.add_argument("--prompt", required=True,
                     help="Text to tokenize.")
    enc.add_argument("--pad-to", type=int, default=0,
                     help="Right-pad with 0s up to this length so the output "
                          "matches a fixed init_seq_len. 0 = no padding.")
    enc.add_argument("--no-special-tokens", action="store_true",
                     help="Do not prepend BOS/system tokens.")
    enc.set_defaults(func=cmd_encode)

    dec = sub.add_parser("decode",
                         help="Token IDs -> text. Reads from --ids or stdin.")
    dec.add_argument("--model", required=True)
    dec.add_argument("--ids", default=None,
                     help="Comma-separated token IDs. If omitted, reads "
                          "whitespace-separated IDs from stdin.")
    dec.add_argument("--skip-special-tokens", action="store_true")
    dec.set_defaults(func=cmd_decode)

    args = p.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
