"""Builder for the runtime ``nntr_config.json`` consumed by the C++ runner.

The runner reads this file to pick batch size, sequence lengths, and tensor
dtypes that the INI graph alone does not encode (the INI is shape-symbolic).
Field names match what ``Applications/CausalLM/api/model_config.cpp`` expects
so existing CausalLM tooling can also consume the same JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class RuntimeConfig:
    # Tensor type used by ``model->setProperty``. ``"FP32-FP32"`` means
    # FP32 weights and FP32 activations; the runner picks safe defaults.
    model_tensor_type: str = "FP32-FP32"
    batch_size: int = 1
    init_seq_len: int = 8
    max_seq_len: int = 8
    num_to_generate: int = 0
    embedding_dtype: str = "FP32"
    fc_layer_dtype: str = "FP32"
    lmhead_dtype: str = "FP32"
    # Path to the safetensors blob, relative to the JSON or absolute.
    model_file_name: str = "model.safetensors"
    # Optional auxiliary settings; empty string indicates absent.
    tokenizer_file: str = ""
    # Optional metadata: name and architecture identifier for diagnostics.
    model_name: str = ""
    architecture: str = ""
    # Free-form: extra keys consumers may want to attach.
    extra: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        """Render to JSON, flattening ``extra`` into the top-level object."""
        d = asdict(self)
        extra = d.pop("extra", {}) or {}
        merged = {**d, **extra}
        return json.dumps(merged, indent=2, sort_keys=True) + "\n"
