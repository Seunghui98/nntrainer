#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   mc_ids.py
@date   30 Sep 2026
@brief  [#194 P3] Render and tokenize the 40-question multiple choice
        (docs/measurements/prompts/mc-40.tsv) for NNTR_PPL_DECODE runs
@author dlwlzzero <dlwlzzero@gmail.com>

Each question is rendered as
    <question>\\nA. <a>\\nB. <b>\\nC. <c>\\nD. <d>\\nAnswer:
and tokenized the app's way (the model's tokenizer.json, no special
tokens). The app scores decode steps only, so the prompt it is given
(qNN.txt) is the rendering without its last token, and qNN.ids holds that
last token (forced as the first decode input) then the right letter's
token: the one scored step is P(letter | the whole rendering). alts.txt is
the four letter ids for NNTR_PPL_DECODE_ALTS. Refuses a rendering whose
truncated text does not tokenize back to the first n - 1 ids, or a letter
that is not one token.

Usage: mc_ids.py [--tsv mc-40.tsv] [--tokenizer tokenizer.json] [--out dir]
"""

import argparse
import os

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROMPTS = os.path.join(ROOT, "docs", "measurements", "prompts")


def render(q):
    """The question and its four options, ending in 'Answer:'."""
    return "%s\nA. %s\nB. %s\nC. %s\nD. %s\nAnswer:" % tuple(q[1:6])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tsv", default=os.path.join(PROMPTS, "mc-40.tsv"))
    ap.add_argument(
        "--tokenizer",
        default=os.path.join(
            os.environ.get("NNTR_MODEL_DIR", ""), "hf", "tokenizer.json"
        ),
    )
    ap.add_argument("--out", default=os.path.join(PROMPTS, "mc-40"))
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)

    def enc(s):
        return tok.encode(s, add_special_tokens=False).ids

    letters = []
    for c in "ABCD":
        ids = enc(" " + c)
        if len(ids) != 1:
            raise SystemExit("letter ' %s' is %d tokens" % (c, len(ids)))
        letters.append(ids[0])
    rows = [l.rstrip("\n").split("\t") for l in open(a.tsv, encoding="utf-8")]
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for q in rows[1:]:
        if len(q) != 7 or q[6] not in "ABCD" or len(q[6]) != 1:
            raise SystemExit("bad row: %r" % (q,))
        ids = enc(render(q))
        head = tok.decode(ids[:-1], skip_special_tokens=False)
        if enc(head) != ids[:-1] or not render(q).startswith(head):
            raise SystemExit("%s: the truncated text does not tokenize back" % q[0])
        with open(os.path.join(a.out, q[0] + ".txt"), "w", encoding="utf-8") as f:
            f.write(head)
        with open(os.path.join(a.out, q[0] + ".ids"), "w") as f:
            f.write("%d\n%d\n" % (ids[-1], letters["ABCD".index(q[6])]))
        n += 1
    with open(os.path.join(a.out, "alts.txt"), "w") as f:
        f.write(",".join(str(i) for i in letters) + "\n")
    print("MC IDS questions=%d letters=%s out=%s" % (n, letters, a.out))


if __name__ == "__main__":
    main()
