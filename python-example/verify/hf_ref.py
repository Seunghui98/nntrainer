# HF reference: greedy-decode Qwen3-0.6B on the EXACT same raw prompt string
# that nntr_config.json's sample_input uses, then print token ids + text so we
# can compare against nntr_causallm output.
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/workspace/qwen3run/hf/Qwen3-0.6B"
NCFG = "/workspace/nntrainer/Applications/CausalLM/res/qwen3/qwen3-0.6b/nntr_config.json"

with open(NCFG) as f:
    ncfg = json.load(f)
prompt = ncfg["sample_input"]
n_gen = int(ncfg.get("num_to_generate", 64))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

# Encode the raw string exactly (no extra chat-template re-application).
enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
print("PROMPT_TOKEN_IDS:", enc.input_ids[0].tolist())

with torch.no_grad():
    out = model.generate(
        **enc,
        max_new_tokens=n_gen,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
    )
gen_ids = out[0][enc.input_ids.shape[1]:].tolist()
print("GEN_TOKEN_IDS:", gen_ids)
print("GEN_TEXT:", json.dumps(tok.decode(gen_ids, skip_special_tokens=False)))
