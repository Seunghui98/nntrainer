"""Verify Q_gate values for layer 3 self-attention."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Qwen/Qwen3.5-2B"
prompt = "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
model.eval()

text_model = model.model if hasattr(model.model, 'layers') else model.model
inputs = tokenizer(prompt, return_tensors="pt")

# Get layer 3 self_attn module
sa = text_model.layers[3].self_attn
ln = text_model.layers[3].input_layernorm

# Run model to get embeddings through layers 0-2
with torch.no_grad():
    # Get hidden states after embedding + layers 0-2
    hidden = text_model.embed_tokens(inputs["input_ids"])
    # We need to pass through layers 0-2 to get the correct input for layer 3
    # Instead, just run the full model and hook layer 3

    ln_output = [None]
    def ln_hook(m, inp, out):
        ln_output[0] = out.detach()

    h = ln.register_forward_hook(ln_hook)
    outputs = model(inputs["input_ids"])
    h.remove()

    # Now compute q_proj manually on the captured layernorm output
    x = ln_output[0]  # (1, 18, 2048)
    q_full = sa.q_proj(x)  # (1, 18, 4096)

    num_heads = getattr(sa, 'num_heads', 8)
    head_dim = getattr(sa, 'head_dim', 256)
    q_dim = q_full.shape[-1] // 2  # half is query, half is gate

    query = q_full[:, :, :q_dim]   # (1, 18, 2048)
    gate = q_full[:, :, q_dim:]    # (1, 18, 2048)

print(f"=== Layer 3 Q_proj ===")
print(f"num_heads={num_heads}, head_dim={head_dim}, q_dim={q_dim}")
print(f"Query (pos 0)[0:10]: {query[0, 0, :10].tolist()}")
print(f"Gate  (pos 0)[0:10]: {gate[0, 0, :10].tolist()}")
print(f"sigmoid(Gate)[0:10]: {torch.sigmoid(gate[0, 0, :10]).tolist()}")

# Compare with V to compute expected o_proj input
v = sa.v_proj(x)  # (1, 18, 512)
print(f"\nV (pos 0)[0:10]: {v[0, 0, :10].tolist()}")

# For pos 0, attn output = V[0] (causal, self-attend only)
# Gated output = V * sigmoid(gate) (reshaped per head)
# Element 0 = V[head0, dim0] * sigmoid(gate[head0, dim0])
print(f"\nExpected gated[0] = V[0]*sigmoid(gate[0]) = {v[0,0,0].item():.6f} * {torch.sigmoid(gate[0,0,0]).item():.6f} = {(v[0,0,0] * torch.sigmoid(gate[0,0,0])).item():.6f}")

# q_proj weight info
print(f"\nq_proj weight shape: {sa.q_proj.weight.shape}")
print(f"q_proj.weight[0, 0:5]:    {sa.q_proj.weight[0, :5].tolist()}")
print(f"q_proj.weight[2048, 0:5]: {sa.q_proj.weight[q_dim, :5].tolist()}")
