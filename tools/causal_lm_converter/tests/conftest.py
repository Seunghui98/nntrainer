"""Shared pytest fixtures for the causal_lm_converter tests.

The package directory is one level above the tests directory; we add it to
``sys.path`` so test modules can ``import nntr_causal_lm_converter`` without
installing the package.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


# A canonical, very small Qwen3 config used across multiple tests.
# Keeping this central avoids drift between test files.
TINY_QWEN3_CFG = {
    "model_type": "qwen3",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "tie_word_embeddings": True,
    "architectures": ["Qwen3ForCausalLM"],
}


@pytest.fixture
def tiny_qwen3_cfg():
    """Return a fresh copy so test mutation does not leak."""
    return dict(TINY_QWEN3_CFG)


@pytest.fixture(scope="session")
def repo_root():
    """Absolute path to the nntrainer repository root."""
    return os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
