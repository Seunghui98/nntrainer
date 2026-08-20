# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
## @file verify_decoder.py
## @brief PyTorch reference + nntrainer parity check for one BertDecoder
##        cross-attention decode step (the --decoder-init-parity path:
##        token=101 [CLS], position=0, given a golden (post-connector)
##        encoder hidden state).
##
## NOTE: written against the real checkpoint's documented layout
## (MIGRATION_384.md Findings §2-4: best/decoder/model.safetensors is a
## BertLMHeadModel with add_cross_attention=True, "bert."/"cls." key
## prefixes, hidden_size=512) but has not been run end to end in this
## environment — no torch/transformers install or checkpoint was reachable
## here. Treat the transformers API usage below (BertLMHeadModel.forward
## signature, output.logits shape) as best-effort until confirmed on a
## machine with both.
import argparse
import numpy as np
import torch
from transformers import BertLMHeadModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder_path", required=True,
                    help="HF decoder checkpoint dir, e.g. v4.0.0-S1/best/decoder "
                    "(config.json + model.safetensors, BertLMHeadModel with "
                    "add_cross_attention=True)")
    ap.add_argument("--golden", default="golden.encoder_hidden.npy",
                    help="post-connector encoder hidden state "
                    "[1,enc_len,decoder_hidden_size], e.g. from "
                    "verify_encoder.py's --out golden or "
                    "res/bert-decoder/golden.encoder_hidden.npy")
    ap.add_argument("--nntr-npy", default="nntr_decoder_init_logits.npy",
                    help="nntr_causallm --decoder-init-parity output to compare")
    ap.add_argument("--token-id", type=int, default=101,
                    help="input token id (default 101 = [CLS], matching "
                    "BertDecoder::decodeStep's default)")
    ap.add_argument("--out", default="pt_decoder_init_logits.npy")
    ap.add_argument("--min-cos", type=float, default=0.9999)
    args = ap.parse_args()

    model = BertLMHeadModel.from_pretrained(
        args.decoder_path, attn_implementation="eager"
    ).eval()

    enc_hidden = torch.from_numpy(np.load(args.golden)).to(torch.float32)
    if enc_hidden.dim() == 2:
        enc_hidden = enc_hidden.unsqueeze(0)

    input_ids = torch.tensor([[args.token_id]], dtype=torch.long)
    position_ids = torch.tensor([[0]], dtype=torch.long)
    token_type_ids = torch.tensor([[0]], dtype=torch.long)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            encoder_hidden_states=enc_hidden,
            use_cache=False,
            return_dict=True,
        )
    logits = out.logits.cpu().numpy().astype(np.float32)  # [1,1,vocab_size]
    pt_argmax = int(logits[0, -1].argmax())

    np.save(args.out, logits)
    print("pt logits shape:", logits.shape, "argmax:", pt_argmax)
    print("saved:", args.out)

    if args.nntr_npy:
        nntr = np.load(args.nntr_npy).astype(np.float64).reshape(-1)
        ref = logits.astype(np.float64).reshape(-1)
        if nntr.shape != ref.shape:
            raise SystemExit(
                f"SHAPE MISMATCH: nntr has {nntr.shape[0]} floats, PyTorch "
                f"has {ref.shape[0]} — check vocab_size/decoder_hidden_size "
                "agree between the PyTorch and nntrainer sides."
            )
        nntr_argmax = int(np.argmax(nntr))
        cos = float(ref @ nntr / (np.linalg.norm(ref) * np.linalg.norm(nntr)))
        print(f"pt argmax={pt_argmax}  nntr argmax={nntr_argmax}  "
              f"{'MATCH' if pt_argmax == nntr_argmax else 'MISMATCH'}")
        print(f"logits cosine = {cos:.8f} (min {args.min_cos})")
        ok = (pt_argmax == nntr_argmax) and (cos >= args.min_cos)
        print("DECODER PARITY:", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
