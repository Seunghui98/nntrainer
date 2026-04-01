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
        ("embeddings.word_embeddings.weight", "none"),
        ("embeddings.token_type_embeddings.weight", "none"),
        ("embeddings.position_embeddings.weight", "none"),
        ("embeddings.LayerNorm.weight", "none"),
        ("embeddings.LayerNorm.bias", "none"),
        ("encoder.layer.0.attention.self.query.weight", "transpose"),
        ("encoder.layer.0.attention.self.query.bias", "none"),
        ("encoder.layer.0.attention.self.key.weight", "transpose"),
        ("encoder.layer.0.attention.self.key.bias", "none"),
        ("encoder.layer.0.attention.self.value.weight", "transpose"),
        ("encoder.layer.0.attention.self.value.bias", "none"),
        ("encoder.layer.0.attention.output.dense.weight", "transpose"),
        ("encoder.layer.0.attention.output.dense.bias", "none"),
        ("encoder.layer.0.attention.output.LayerNorm.weight", "none"),
        ("encoder.layer.0.attention.output.LayerNorm.bias", "none"),
        ("encoder.layer.0.intermediate.dense.weight", "transpose"),
        ("encoder.layer.0.intermediate.dense.bias", "none"),
        ("encoder.layer.0.output.dense.weight", "transpose"),
        ("encoder.layer.0.output.dense.bias", "none"),
        ("encoder.layer.0.output.LayerNorm.weight", "none"),
        ("encoder.layer.0.output.LayerNorm.bias", "none"),
        ("encoder.layer.1.attention.self.query.weight", "transpose"),
        ("encoder.layer.1.attention.self.query.bias", "none"),
        ("encoder.layer.1.attention.self.key.weight", "transpose"),
        ("encoder.layer.1.attention.self.key.bias", "none"),
        ("encoder.layer.1.attention.self.value.weight", "transpose"),
        ("encoder.layer.1.attention.self.value.bias", "none"),
        ("encoder.layer.1.attention.output.dense.weight", "transpose"),
        ("encoder.layer.1.attention.output.dense.bias", "none"),
        ("encoder.layer.1.attention.output.LayerNorm.weight", "none"),
        ("encoder.layer.1.attention.output.LayerNorm.bias", "none"),
        ("encoder.layer.1.intermediate.dense.weight", "transpose"),
        ("encoder.layer.1.intermediate.dense.bias", "none"),
        ("encoder.layer.1.output.dense.weight", "transpose"),
        ("encoder.layer.1.output.dense.bias", "none"),
        ("encoder.layer.1.output.LayerNorm.weight", "none"),
        ("encoder.layer.1.output.LayerNorm.bias", "none"),
        ("encoder.layer.2.attention.self.query.weight", "transpose"),
        ("encoder.layer.2.attention.self.query.bias", "none"),
        ("encoder.layer.2.attention.self.key.weight", "transpose"),
        ("encoder.layer.2.attention.self.key.bias", "none"),
        ("encoder.layer.2.attention.self.value.weight", "transpose"),
        ("encoder.layer.2.attention.self.value.bias", "none"),
        ("encoder.layer.2.attention.output.dense.weight", "transpose"),
        ("encoder.layer.2.attention.output.dense.bias", "none"),
        ("encoder.layer.2.attention.output.LayerNorm.weight", "none"),
        ("encoder.layer.2.attention.output.LayerNorm.bias", "none"),
        ("encoder.layer.2.intermediate.dense.weight", "transpose"),
        ("encoder.layer.2.intermediate.dense.bias", "none"),
        ("encoder.layer.2.output.dense.weight", "transpose"),
        ("encoder.layer.2.output.dense.bias", "none"),
        ("encoder.layer.2.output.LayerNorm.weight", "none"),
        ("encoder.layer.2.output.LayerNorm.bias", "none"),
        ("encoder.layer.3.attention.self.query.weight", "transpose"),
        ("encoder.layer.3.attention.self.query.bias", "none"),
        ("encoder.layer.3.attention.self.key.weight", "transpose"),
        ("encoder.layer.3.attention.self.key.bias", "none"),
        ("encoder.layer.3.attention.self.value.weight", "transpose"),
        ("encoder.layer.3.attention.self.value.bias", "none"),
        ("encoder.layer.3.attention.output.dense.weight", "transpose"),
        ("encoder.layer.3.attention.output.dense.bias", "none"),
        ("encoder.layer.3.attention.output.LayerNorm.weight", "none"),
        ("encoder.layer.3.attention.output.LayerNorm.bias", "none"),
        ("encoder.layer.3.intermediate.dense.weight", "transpose"),
        ("encoder.layer.3.intermediate.dense.bias", "none"),
        ("encoder.layer.3.output.dense.weight", "transpose"),
        ("encoder.layer.3.output.dense.bias", "none"),
        ("encoder.layer.3.output.LayerNorm.weight", "none"),
        ("encoder.layer.3.output.LayerNorm.bias", "none"),
        ("pooler.dense.weight", "transpose"),
        ("pooler.dense.bias", "none"),
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
