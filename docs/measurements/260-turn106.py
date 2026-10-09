"""@file    260-turn106.py
@brief   #260: generated text up to the first <turn|> (id 106) and its index.

usage: python3 260-turn106.py TOKENIZER LOG...
The generated region starts after the prompt's closing
'</Text><turn|>\n<|turn>model' and ends at the app's summary banner, with
[HTP]/[PPL] lines removed. Prints the token count, the token index of the
first <turn|> (or "none"), and the text before it.
"""
import re
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
for f in sys.argv[2:]:
    s = open(f, errors="replace").read()
    m = "</Text><turn|>\n<|turn>model"
    i = s.find(m)
    j = s.find("=================[ LLM", i)
    g = re.sub(r"\[(HTP|PPL)\][^\n]*\n", "", s[i + len(m):j]).lstrip("\n")
    n = len(tok.encode(g.strip(), add_special_tokens=False).ids)
    k = g.find("<turn|>")
    idx = len(tok.encode(g[:k], add_special_tokens=False).ids) if k >= 0 else None
    before = g[:k] if k >= 0 else g
    print("=== %s tokens=%d first_turn_end_at=%s" % (f.split("/")[-1], n, idx))
    print(" ".join(before.split()))
