"""
Verify NNTrainer's GatedDeltaNet by comparing with HuggingFace reference.
Run this script, then compare the printed values with NNTrainer's debug output.
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_path = "Qwen/Qwen3.5-2B"
prompt = "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
model.eval()

# Get text model
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    text_model = model.model
elif hasattr(model, 'language_model'):
    text_model = model.language_model.model
else:
    text_model = model.model

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
print(f"Input tokens: {input_ids.shape}, ids: {input_ids[0].tolist()}")

# Hook to capture layer 0 outputs
layer0_output = {}

def hook_fn(module, input, output):
    if isinstance(output, tuple):
        layer0_output['hidden'] = output[0].detach()
    else:
        layer0_output['hidden'] = output.detach()

def hook_input_fn(module, input, output):
    layer0_output['input'] = input[0].detach()

# Register hook on layer 0
layer0 = text_model.layers[0]
hook1 = layer0.register_forward_hook(hook_fn)
hook2 = layer0.register_forward_hook(hook_input_fn)

# Also hook the linear_attn module directly
if hasattr(layer0, 'linear_attn'):
    linear_attn = layer0.linear_attn
    la_output = {}
    def la_hook(module, input, output):
        if isinstance(output, tuple):
            la_output['output'] = output[0].detach()
        else:
            la_output['output'] = output.detach()
        la_output['input'] = input[0].detach()
    hook3 = linear_attn.register_forward_hook(la_hook)

with torch.no_grad():
    outputs = model(input_ids)

hook1.remove()
hook2.remove()

# Print results
if 'input' in la_output:
    la_in = la_output['input']
    la_out = la_output['output']
    print(f"\n=== Layer 0 Linear Attention ===")
    print(f"Input norm (first token): {la_in[0, 0].norm().item():.6f}")
    print(f"Input norm (last token):  {la_in[0, -1].norm().item():.6f}")
    print(f"Output norm (first token): {la_out[0, 0].norm().item():.6f}")
    print(f"Output norm (last token):  {la_out[0, -1].norm().item():.6f}")
    print(f"Output first 5 values (token 0): {la_out[0, 0, :5].tolist()}")
    print(f"Output first 5 values (last token): {la_out[0, -1, :5].tolist()}")

# Print logits
logits = outputs.logits[0, -1]  # last position
top5_vals, top5_idx = logits.topk(5)
print(f"\n=== Final logits (last position) ===")
print(f"Max logit: {logits.max().item():.4f}, Min: {logits.min().item():.4f}")
print(f"Top 5 tokens: {top5_idx.tolist()}")
print(f"Top 5 logits: {top5_vals.tolist()}")
print(f"Top 5 decoded: {[tokenizer.decode([t]) for t in top5_idx.tolist()]}")
print(f"Logit[198]: {logits[198].item():.4f}")

# Check what the model would generate
gen = model.generate(input_ids, max_new_tokens=20, do_sample=False)
print(f"\n=== Generated text (greedy, 20 tokens) ===")
print(tokenizer.decode(gen[0][input_ids.shape[1]:]))
