"""
Verify Q_gate values for layer 3 self-attention.
Compare NNTrainer's Q_gate output with HuggingFace reference.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Qwen/Qwen3.5-2B"
prompt = "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
model.eval()

if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    text_model = model.model
else:
    text_model = model.model

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

# Hook layer 3 self_attn to capture gate values
layer3 = text_model.layers[3]
sa = layer3.self_attn
captured = {}

def sa_hook(module, args, kwargs, output):
    # Capture internal values from the self_attn forward
    captured['sa_output'] = output[0].detach() if isinstance(output, tuple) else output.detach()

h1 = sa.register_forward_hook(sa_hook, with_kwargs=True)

# Also manually compute q_proj to get gate values
with torch.no_grad():
    outputs = model(input_ids)

h1.remove()

# Manually get gate values for layer 3
with torch.no_grad():
    # Get the input to layer 3 self_attn (= input_layernorm output)
    layer3_input = None
    def capture_ln(module, args, kwargs, output):
        nonlocal layer3_input
        layer3_input = output.detach()
    h = layer3.input_layernorm.register_forward_hook(capture_ln, with_kwargs=True)
    outputs2 = model(input_ids)
    h.remove()

    # Compute q_proj manually
    q_proj_out = sa.q_proj(layer3_input)  # (1, 18, 4096)
    num_heads = sa.num_heads
    head_dim = sa.head_dim
    q_dim = num_heads * head_dim

    query = q_proj_out[:, :, :q_dim]      # (1, 18, 2048) - query part
    gate = q_proj_out[:, :, q_dim:]        # (1, 18, 2048) - gate part

    print(f"=== Layer 3 Q_proj analysis ===")
    print(f"q_proj output shape: {q_proj_out.shape}")
    print(f"num_heads={num_heads}, head_dim={head_dim}, q_dim={q_dim}")
    print(f"query shape: {query.shape}, gate shape: {gate.shape}")
    print(f"\nQuery (pos 0, first 10): {query[0, 0, :10].tolist()}")
    print(f"Gate  (pos 0, first 10): {gate[0, 0, :10].tolist()}")
    print(f"sigmoid(Gate) (pos 0, first 10): {torch.sigmoid(gate[0, 0, :10]).tolist()}")

    # Also print q_proj weight structure
    print(f"\nq_proj weight shape: {sa.q_proj.weight.shape}")
    print(f"q_proj weight[0, :5]: {sa.q_proj.weight[0, :5].tolist()}")
    print(f"q_proj weight[{q_dim}, :5]: {sa.q_proj.weight[q_dim, :5].tolist()}")

    # Print o_proj input (= gated attention output)
    v_proj_out = sa.v_proj(layer3_input)
    print(f"\nV_proj output (pos 0, first 10): {v_proj_out[0, 0, :10].tolist()}")

# Print final logits
logits = outputs.logits[0, -1]
top5_vals, top5_idx = logits.topk(5)
print(f"\n=== Final logits ===")
print(f"Top 5: {[(tokenizer.decode([t]), f'{v:.2f}') for t, v in zip(top5_idx.tolist(), top5_vals.tolist())]}")
