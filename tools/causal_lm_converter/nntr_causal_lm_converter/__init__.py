"""nntr_causal_lm_converter

Converts HuggingFace causal LM models into NNTrainer-native artifacts:
  - <model>.ini       : graph topology (consumed by model->loadFromConfig)
  - <model>.safetensors : weights, with name-based lookup matching INI sections
  - nntr_config.json  : runtime hyperparameters (batch, seq lens, dtypes)

The package is intentionally small. New architectures are added by registering
an `ArchitectureBuilder` subclass in `architectures/`.

Public API:

    from nntr_causal_lm_converter import convert, list_architectures

    paths = convert(hf_config=..., output_dir=..., model_name=...,
                    state_dict=..., dtype="float32")
"""

from .architectures import get_builder, list_architectures  # noqa: F401
from .cli import convert  # noqa: F401
