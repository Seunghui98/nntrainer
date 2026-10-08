"""Token index where a generated text turns into a loop.

usage: python3 -I loopstart.py TOKENIZER LOG...
Generated text = after the last '<|turn>model' up to the summary banner, with
[HTP]/[PPL] lines removed. Loop start = the smallest i such that tokens[i:]
is periodic with some period p <= 64 for at least 3 full periods to the end
(a trailing partial period allowed). Also the first index from which every
16-token window repeats an earlier one in the generated text (a loop whose
content drifts, e.g. a growing tag). Prints both, or "none".
"""
import re
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
for f in sys.argv[2:]:
    s = open(f, errors="replace").read()
    i = s.rfind("<|turn>model")
    j = s.find("=================[ LLM", i)
    t = re.sub(r"\[(HTP|PPL)\][^\n]*\n", "", s[i + len("<|turn>model"):j])
    ids = tok.encode(t.strip(), add_special_tokens=False).ids
    n = len(ids)
    best = None
    for p in range(1, 65):
        k = n - p  # walk back while ids[k] == ids[k + p]
        while k > 0 and ids[k - 1] == ids[k - 1 + p]:
            k -= 1
        if n - k >= 3 * p and (best is None or k < best[0]):
            best = (k, p)
    # softer: the first index after which every 16-token window repeats one
    # that already occurred earlier in the generated text
    W = 16
    seen, rep = {}, [False] * n
    for a in range(n - W + 1):
        key = tuple(ids[a:a + W])
        rep[a] = key in seen
        seen.setdefault(key, a)
    onset = None
    for a in range(n - W + 1):
        if all(rep[a:n - W + 1]):
            onset = a
            break
    # loosest: where some 10-token window occurs for the third time (a
    # repeated sentence, even with drifting text between its copies)
    cnt, third = {}, None
    for a in range(n - 9):
        key = tuple(ids[a:a + 10])
        cnt[key] = cnt.get(key, 0) + 1
        if cnt[key] == 3:
            third = a
            break
    print(f.split("/")[-1], "tokens=%d" % n, "third10_at=%s" % third,
          ("periodic_from=%d period=%d" % best) if best else "periodic_from=none",
          "repeat16_from=%s" % (onset if onset is not None else "none"))
