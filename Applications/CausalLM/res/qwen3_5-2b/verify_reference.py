"""
Verify NNTrainer's GatedDeltaNet by comparing with HuggingFace reference.
"""
import torch
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

# Hooks to capture intermediate values
captured = {}

layer0 = text_model.layers[0]
la = layer0.linear_attn

# Hook on the full linear_attn output
def la_hook(module, args, kwargs, output):
    if isinstance(output, tuple):
        captured['la_out'] = output[0].detach()
    else:
        captured['la_out'] = output.detach()

# Hook on decoder layer output (after residual)
def decoder_hook(module, args, kwargs, output):
    if isinstance(output, tuple):
        captured['decoder_out'] = output[0].detach()
    else:
        captured['decoder_out'] = output.detach()

h1 = la.register_forward_hook(la_hook, with_kwargs=True)
h2 = layer0.register_forward_hook(decoder_hook, with_kwargs=True)

with torch.no_grad():
    outputs = model(input_ids)

h1.remove()
h2.remove()

# Print linear_attn output
if 'la_out' in captured:
    la_out = captured['la_out']
    print(f"\n=== Layer 0 linear_attn OUTPUT ===")
    print(f"Shape: {la_out.shape}")
    print(f"Token 0 first 10: {la_out[0, 0, :10].tolist()}")
    print(f"Token 0 norm: {la_out[0, 0].norm().item():.6f}")
    print(f"Token -1 first 10: {la_out[0, -1, :10].tolist()}")
    print(f"Token -1 norm: {la_out[0, -1].norm().item():.6f}")

if 'decoder_out' in captured:
    dec = captured['decoder_out']
    print(f"\n=== Layer 0 decoder OUTPUT (after residual) ===")
    print(f"Token 0 first 10: {dec[0, 0, :10].tolist()}")

# Final logits
logits = outputs.logits[0, -1]
top5_vals, top5_idx = logits.topk(5)
print(f"\n=== Final logits (last position) ===")
print(f"Max: {logits.max().item():.4f}, Min: {logits.min().item():.4f}")
print(f"Top 5 tokens: {top5_idx.tolist()}")
print(f"Top 5 logits: {[f'{v:.4f}' for v in top5_vals.tolist()]}")
print(f"Top 5 decoded: {[tokenizer.decode([t]) for t in top5_idx.tolist()]}")
print(f"Logit[198]: {logits[198].item():.4f}")

# Generate
gen = model.generate(input_ids, max_new_tokens=20, do_sample=False)
print(f"\n=== Generated (greedy, 20 tokens) ===")
print(tokenizer.decode(gen[0][input_ids.shape[1]:]))
