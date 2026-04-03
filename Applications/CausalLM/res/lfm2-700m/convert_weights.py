"""Auto-generated weight conversion script."""
import torch
import sys

def convert(model_path, output_path, dtype='float32'):
    from transformers import AutoModel
    target = torch.float32 if dtype == 'float32' else torch.float16
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    sd = model.state_dict()

    # Weight mapping (HF key, transform)
    WEIGHT_MAP = [
        ("embed_tokens.weight", "none"),

        ("layers.0.operator_norm.weight", "none"),
        ("layers.0.conv.in_proj.weight", "transpose"),
        ("layers.0.conv.conv.weight", "transpose"),
        ("layers.0.conv.out_proj.weight", "transpose"),
        ("layers.0.ffn_norm.weight", "none"),
        ("layers.0.feed_forward.w1.weight", "transpose"),
        ("layers.0.feed_forward.w3.weight", "transpose"),
        ("layers.0.feed_forward.w2.weight", "transpose"),

        ("layers.1.operator_norm.weight", "none"),
        ("layers.1.conv.in_proj.weight", "transpose"),
        ("layers.1.conv.conv.weight", "transpose"),
        ("layers.1.conv.out_proj.weight", "transpose"),
        ("layers.1.ffn_norm.weight", "none"),
        ("layers.1.feed_forward.w1.weight", "transpose"),
        ("layers.1.feed_forward.w3.weight", "transpose"),
        ("layers.1.feed_forward.w2.weight", "transpose"),

        ("layers.2.operator_norm.weight", "none"),
        ("layers.2.self_attn.q_proj.weight", "transpose"),
        ("layers.2.self_attn.q_layernorm.weight", "none"),
        ("layers.2.self_attn.k_proj.weight", "transpose"),
        ("layers.2.self_attn.k_layernorm.weight", "none"),
        ("layers.2.self_attn.v_proj.weight", "transpose"),
        ("layers.2.self_attn.out_proj.weight", "transpose"),
        ("layers.2.ffn_norm.weight", "none"),
        ("layers.2.feed_forward.w1.weight", "transpose"),
        ("layers.2.feed_forward.w3.weight", "transpose"),
        ("layers.2.feed_forward.w2.weight", "transpose"),

        ("layers.3.operator_norm.weight", "none"),
        ("layers.3.conv.in_proj.weight", "transpose"),
        ("layers.3.conv.conv.weight", "transpose"),
        ("layers.3.conv.out_proj.weight", "transpose"),
        ("layers.3.ffn_norm.weight", "none"),
        ("layers.3.feed_forward.w1.weight", "transpose"),
        ("layers.3.feed_forward.w3.weight", "transpose"),
        ("layers.3.feed_forward.w2.weight", "transpose"),

        ("layers.4.operator_norm.weight", "none"),
        ("layers.4.conv.in_proj.weight", "transpose"),
        ("layers.4.conv.conv.weight", "transpose"),
        ("layers.4.conv.out_proj.weight", "transpose"),
        ("layers.4.ffn_norm.weight", "none"),
        ("layers.4.feed_forward.w1.weight", "transpose"),
        ("layers.4.feed_forward.w3.weight", "transpose"),
        ("layers.4.feed_forward.w2.weight", "transpose"),

        ("layers.5.operator_norm.weight", "none"),
        ("layers.5.self_attn.q_proj.weight", "transpose"),
        ("layers.5.self_attn.q_layernorm.weight", "none"),
        ("layers.5.self_attn.k_proj.weight", "transpose"),
        ("layers.5.self_attn.k_layernorm.weight", "none"),
        ("layers.5.self_attn.v_proj.weight", "transpose"),
        ("layers.5.self_attn.out_proj.weight", "transpose"),
        ("layers.5.ffn_norm.weight", "none"),
        ("layers.5.feed_forward.w1.weight", "transpose"),
        ("layers.5.feed_forward.w3.weight", "transpose"),
        ("layers.5.feed_forward.w2.weight", "transpose"),

        ("layers.6.operator_norm.weight", "none"),
        ("layers.6.conv.in_proj.weight", "transpose"),
        ("layers.6.conv.conv.weight", "transpose"),
        ("layers.6.conv.out_proj.weight", "transpose"),
        ("layers.6.ffn_norm.weight", "none"),
        ("layers.6.feed_forward.w1.weight", "transpose"),
        ("layers.6.feed_forward.w3.weight", "transpose"),
        ("layers.6.feed_forward.w2.weight", "transpose"),

        ("layers.7.operator_norm.weight", "none"),
        ("layers.7.conv.in_proj.weight", "transpose"),
        ("layers.7.conv.conv.weight", "transpose"),
        ("layers.7.conv.out_proj.weight", "transpose"),
        ("layers.7.ffn_norm.weight", "none"),
        ("layers.7.feed_forward.w1.weight", "transpose"),
        ("layers.7.feed_forward.w3.weight", "transpose"),
        ("layers.7.feed_forward.w2.weight", "transpose"),

        ("layers.8.operator_norm.weight", "none"),
        ("layers.8.self_attn.q_proj.weight", "transpose"),
        ("layers.8.self_attn.q_layernorm.weight", "none"),
        ("layers.8.self_attn.k_proj.weight", "transpose"),
        ("layers.8.self_attn.k_layernorm.weight", "none"),
        ("layers.8.self_attn.v_proj.weight", "transpose"),
        ("layers.8.self_attn.out_proj.weight", "transpose"),
        ("layers.8.ffn_norm.weight", "none"),
        ("layers.8.feed_forward.w1.weight", "transpose"),
        ("layers.8.feed_forward.w3.weight", "transpose"),
        ("layers.8.feed_forward.w2.weight", "transpose"),

        ("layers.9.operator_norm.weight", "none"),
        ("layers.9.conv.in_proj.weight", "transpose"),
        ("layers.9.conv.conv.weight", "transpose"),
        ("layers.9.conv.out_proj.weight", "transpose"),
        ("layers.9.ffn_norm.weight", "none"),
        ("layers.9.feed_forward.w1.weight", "transpose"),
        ("layers.9.feed_forward.w3.weight", "transpose"),
        ("layers.9.feed_forward.w2.weight", "transpose"),

        ("layers.10.operator_norm.weight", "none"),
        ("layers.10.self_attn.q_proj.weight", "transpose"),
        ("layers.10.self_attn.q_layernorm.weight", "none"),
        ("layers.10.self_attn.k_proj.weight", "transpose"),
        ("layers.10.self_attn.k_layernorm.weight", "none"),
        ("layers.10.self_attn.v_proj.weight", "transpose"),
        ("layers.10.self_attn.out_proj.weight", "transpose"),
        ("layers.10.ffn_norm.weight", "none"),
        ("layers.10.feed_forward.w1.weight", "transpose"),
        ("layers.10.feed_forward.w3.weight", "transpose"),
        ("layers.10.feed_forward.w2.weight", "transpose"),

        ("layers.11.operator_norm.weight", "none"),
        ("layers.11.conv.in_proj.weight", "transpose"),
        ("layers.11.conv.conv.weight", "transpose"),
        ("layers.11.conv.out_proj.weight", "transpose"),
        ("layers.11.ffn_norm.weight", "none"),
        ("layers.11.feed_forward.w1.weight", "transpose"),
        ("layers.11.feed_forward.w3.weight", "transpose"),
        ("layers.11.feed_forward.w2.weight", "transpose"),

        ("layers.12.operator_norm.weight", "none"),
        ("layers.12.self_attn.q_proj.weight", "transpose"),
        ("layers.12.self_attn.q_layernorm.weight", "none"),
        ("layers.12.self_attn.k_proj.weight", "transpose"),
        ("layers.12.self_attn.k_layernorm.weight", "none"),
        ("layers.12.self_attn.v_proj.weight", "transpose"),
        ("layers.12.self_attn.out_proj.weight", "transpose"),
        ("layers.12.ffn_norm.weight", "none"),
        ("layers.12.feed_forward.w1.weight", "transpose"),
        ("layers.12.feed_forward.w3.weight", "transpose"),
        ("layers.12.feed_forward.w2.weight", "transpose"),

        ("layers.13.operator_norm.weight", "none"),
        ("layers.13.conv.in_proj.weight", "transpose"),
        ("layers.13.conv.conv.weight", "transpose"),
        ("layers.13.conv.out_proj.weight", "transpose"),
        ("layers.13.ffn_norm.weight", "none"),
        ("layers.13.feed_forward.w1.weight", "transpose"),
        ("layers.13.feed_forward.w3.weight", "transpose"),
        ("layers.13.feed_forward.w2.weight", "transpose"),

        ("layers.14.operator_norm.weight", "none"),
        ("layers.14.self_attn.q_proj.weight", "transpose"),
        ("layers.14.self_attn.q_layernorm.weight", "none"),
        ("layers.14.self_attn.k_proj.weight", "transpose"),
        ("layers.14.self_attn.k_layernorm.weight", "none"),
        ("layers.14.self_attn.v_proj.weight", "transpose"),
        ("layers.14.self_attn.out_proj.weight", "transpose"),
        ("layers.14.ffn_norm.weight", "none"),
        ("layers.14.feed_forward.w1.weight", "transpose"),
        ("layers.14.feed_forward.w3.weight", "transpose"),
        ("layers.14.feed_forward.w2.weight", "transpose"),

        ("layers.15.operator_norm.weight", "none"),
        ("layers.15.conv.in_proj.weight", "transpose"),
        ("layers.15.conv.conv.weight", "transpose"),
        ("layers.15.conv.out_proj.weight", "transpose"),
        ("layers.15.ffn_norm.weight", "none"),
        ("layers.15.feed_forward.w1.weight", "transpose"),
        ("layers.15.feed_forward.w3.weight", "transpose"),
        ("layers.15.feed_forward.w2.weight", "transpose"),

        ("embedding_norm.weight", "none"),
    ]

    with open(output_path, 'wb') as f:
        for hf_key, transform in WEIGHT_MAP:
            t = sd[hf_key].to(target)
            if transform == 'transpose' and t.dim() == 2:
                t = t.t().contiguous()
            f.write(t.cpu().numpy().tobytes())

    print(f'Saved {len(WEIGHT_MAP)} weight tensors to {output_path}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python convert_weights.py <model_path> <output.bin> [float32|float16]')
        sys.exit(1)
    dtype = sys.argv[3] if len(sys.argv) > 3 else 'float32'
    convert(sys.argv[1], sys.argv[2], dtype)
