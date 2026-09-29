#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   loop_check.py
@date   29 Sep 2026
@brief  Flags repetition loops in nntrainer_causallm generations: the
        longest run of identical sentences (L1) and the share of repeated
        word 12-grams in the second half (L2)
@see    https://github.com/nntrainer/nntrainer
@author dlwlzzero <dlwlzzero@gmail.com>
@bug    No known bugs except for NYI items

Plan 152 section 4 step 4 (the re-judge's text column). Host only.

    python3 tools/htp/loop_check.py --prompt prompt512.txt A_p01.log D1_p01.log

For each log: the generated text is what follows the prompt echo and
precedes the "=================[ LLM" banner. Then
  L1  the longest run of consecutive identical sentences of >= 3 words
      (split after . ! ? 。 ？ ！ followed by whitespace; lowercased,
      whitespace collapsed, trailing punctuation stripped);
  L2  of the word 12-grams in the second half of the text's 12-gram
      sequence, the fraction that already occurred earlier in it;
  loop = L1 >= 3 or L2 >= 0.5.
One line per log: LOOP <label> L1run=<n> L2=<x> loop=0|1 (label = the
log's file name without .log). A variant fails a prompt only where it loops
and A does not: A itself loops on some prompts.

--words N cuts the generated text to its first N words. Self-test (plan
152), prompt512 and the #134 G=512 run-1 logs: at 200 words A 1 / 0.00 ->
0, B 7 / 1.00 -> 1, D 8 / 1.00 -> 1; D at 100 words 3 / 0.98; A at 300
words L2 0.60, at all 431 words 0.95.
"""

import argparse
import os
import re
import sys

BANNER = "=================[ LLM"
SENT_END = re.compile(r"(?<=[.!?。？！])\s+")
TRAIL_PUNCT = ".!?。？！,;:\"'"


def generated_text(log, prompt):
    """The text between the prompt echo and the banner."""
    end = log.find(BANNER)
    if end < 0:
        end = len(log)
    head = prompt.strip()
    start = log.find(head)
    if start < 0:
        # The echo may differ in whitespace: match on the last 80 chars.
        tail = " ".join(head.split())[-80:]
        flat = " ".join(log[:end].split())
        pos = flat.find(tail)
        if pos < 0:
            raise ValueError("prompt echo not found")
        return flat[pos + len(tail):]
    return log[start + len(head):end]


def l1_run(text):
    """Longest run of consecutive identical sentences of >= 3 words."""
    sents = []
    for s in SENT_END.split(text):
        s = " ".join(s.lower().split()).rstrip(TRAIL_PUNCT).strip()
        if s:
            sents.append(s)
    best = run = 0
    prev = None
    for s in sents:
        if len(s.split()) < 3:
            prev, run = None, 0
            continue
        run = run + 1 if s == prev else 1
        prev = s
        best = max(best, run)
    return best


def l2_share(words, n=12):
    """Share of the second half of the n-gram sequence already seen before
    (halving the grams, not the words, is what plan 152's numbers used)."""
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    half = len(grams) // 2
    seen = set()
    rep = tot = 0
    for i, g in enumerate(grams):
        if i >= half:
            tot += 1
            rep += g in seen
        seen.add(g)
    return rep / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompt", required=True, help="the prompt file")
    ap.add_argument("--words", type=int, default=0,
                    help="cut the generated text to its first N words")
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()
    with open(args.prompt, encoding="utf-8", errors="replace") as f:
        prompt = f.read()
    rc = 0
    for path in args.logs:
        label = os.path.basename(path)
        if label.endswith(".log"):
            label = label[:-4]
        with open(path, encoding="utf-8", errors="replace") as f:
            log = f.read()
        try:
            text = generated_text(log, prompt)
        except ValueError as e:
            print(f"LOOP {label} error={e}")
            rc = 1
            continue
        words = text.split()
        if args.words:
            words = words[:args.words]
            text = " ".join(words)
        l1 = l1_run(text)
        l2 = l2_share([w.lower() for w in words])
        loop = int(l1 >= 3 or l2 >= 0.5)
        print(f"LOOP {label} L1run={l1} L2={l2:.2f} loop={loop}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
