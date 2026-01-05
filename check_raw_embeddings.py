
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

def debug_full_model():
    print("Loading model...")
    model = SentenceTransformer(
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float32},
    )
    
    # Access the underlying Qwen2Model
    # SentenceTransformer -> Transformer -> auto_model
    if hasattr(model, 'modules') and hasattr(model[0], 'auto_model'):
        transformer_model = model[0].auto_model
    else:
        transformer_model = model.model

    print("\n[DEBUG] Registering Hooks for Layer-wise Comparison...")

    def print_tensor(name, tensor):
        # Print first 5 elements of the first token (batch 0, sequence 0)
        # Tensor is likely [Batch, Seq, Dim]
        if tensor.dim() >= 2:
            data = tensor[0, 0, :5].detach().cpu().numpy()
        else:
            data = tensor[:5].detach().cpu().numpy() # Fallback
            
        print(f"DEBUG: {name} [:5]:")
        # Format matching C++ output style: space separated
        formatted = " ".join([f"{x:.6g}" for x in data])
        print(f"  Values: {formatted}")

    # Hook for RMSNorm
    def hook_rmsnorm(name):
        def fn(module, inputs, output):
            # inputs is a tuple, input[0] is the tensor
            print_tensor(f"RMSNorm (Layer: {name}) In", inputs[0])
            print_tensor(f"RMSNorm (Layer: {name}) Out", output)
        return fn

    # Hook for MLP (SwiGLU)
    # Qwen2MLP has gate_proj, up_proj, down_proj
    # We want to emulate C++ "SwiGLU" debugging which shows Gate, Val, and Out (Silu(Gate)*Val)
    # Since we can't easily hook *inside* the forward method of MLP without replacing it,
    # we will hook the Linear layers inside MLP to capture their outputs.
    
    # Storage for MLP intermediates per layer
    mlp_cache = {} 

    def hook_mlp_gate(layer_idx):
        def fn(module, inputs, output):
            mlp_cache[f"{layer_idx}_gate"] = output
        return fn

    def hook_mlp_up(layer_idx):
        def fn(module, inputs, output):
            mlp_cache[f"{layer_idx}_up"] = output
            
            # Since up_proj and gate_proj run in parallel, we can try to print here if both are ready,
            # but order isn't guaranteed. Better to hook the MLP module output? 
            # No, MLP module output is after down_proj.
            # Let's print Gate and Val (Up) when both are available, or just print individually.
            # C++ prints them together.
            
            gate = mlp_cache.get(f"{layer_idx}_gate")
            val = output
            
            if gate is not None:
                print_tensor(f"SwiGLU (Layer: layer{layer_idx}_ffn_swiglu) In1(Gate)", gate)
                print_tensor(f"SwiGLU (Layer: layer{layer_idx}_ffn_swiglu) In2(Val)", val)
                
                # Calculate SwiGLU Output: SiLU(Gate) * Val
                swiglu_out = F.silu(gate) * val
                print_tensor(f"SwiGLU (Layer: layer{layer_idx}_ffn_swiglu) Out", swiglu_out)
                
                # Clean up cache
                del mlp_cache[f"{layer_idx}_gate"]
                
        return fn

    # Determine where 'layers' are stored
    layers = None
    if hasattr(transformer_model, 'layers'):
        layers = transformer_model.layers
    elif hasattr(transformer_model, 'model') and hasattr(transformer_model.model, 'layers'):
        layers = transformer_model.model.layers
    else:
        print("[ERROR] Could not find 'layers' attribute in model.")
        return

    # Iterate over layers and register hooks
    for i, layer in enumerate(layers):
        # layer is Qwen2DecoderLayer
        
        # 1. Input LayerNorm (Attention Norm)
        layer.input_layernorm.register_forward_hook(hook_rmsnorm(f"layer{i}_attention_norm"))
        
        # 2. Post Attention LayerNorm (FFN Norm)
        layer.post_attention_layernorm.register_forward_hook(hook_rmsnorm(f"layer{i}_ffn_norm"))
        
        # 3. MLP Internals
        # Note: gate_proj and up_proj are Linear layers
        layer.mlp.gate_proj.register_forward_hook(hook_mlp_gate(i))
        layer.mlp.up_proj.register_forward_hook(hook_mlp_up(i))
    
    # 4. MLP
    # Hook for Final RMSNorm
    if hasattr(transformer_model, 'norm'):
        transformer_model.norm.register_forward_hook(hook_rmsnorm("output_norm"))
    elif hasattr(transformer_model, 'model') and hasattr(transformer_model.model, 'norm'):
        transformer_model.model.norm.register_forward_hook(hook_rmsnorm("output_norm"))


    # Input Data
    ids = [1986, 374, 458, 3110, 11652, 151643]
    input_tensor = torch.tensor([ids])
    
    attention_mask = torch.ones_like(input_tensor)
    print(f"\n--- Running Inference for IDs: {ids} ---")
    with torch.no_grad():
        transformer_model(input_tensor, attention_mask=attention_mask)



if __name__ == "__main__":
    debug_full_model()
