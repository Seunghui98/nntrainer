"""Weight conversion: HF state_dict -> NNTrainer-named safetensors.

NNTrainer addresses safetensors entries by ``"<layer_name>:<weight_role>"``
(e.g. ``layer0_wq:weight``, ``output_norm:gamma``). The role suffix is decided
by the C++ layer implementation:

  * ``fully_connected``  -> ``:weight`` (transposed) and optional ``:bias``
  * ``embedding_layer``  -> ``:Embedding``
  * ``rms_norm``         -> ``:gamma``
  * ``reshaped_rms_norm``-> ``:gamma``
  * ``tie_word_embeddings`` -> ``:Embedding`` (and optional ``:bias``)

Architecture builders return a list of :class:`WeightBinding`. This module
handles transposing FC weights (HF stores ``[out, in]``; NNTrainer expects
``[in, out]``) and writing the safetensors blob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass
class WeightBinding:
    """One entry mapping an HF tensor to an NNTrainer-named tensor.

    Attributes:
      hf_key:   key in the HuggingFace ``state_dict``.
      nntr_key: target name in the safetensors header
                (``"<layer_name>:<role>"``).
      transpose: if True, transpose a 2D tensor before writing
                 (used for FC weights).
    """

    hf_key: str
    nntr_key: str
    transpose: bool = False


# Supported numpy dtypes -> safetensors dtype tags.
# Only these are exercised today; expand as needed.
_DTYPE_TAGS = {
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
}


def _to_numpy(t) -> np.ndarray:
    """Convert a torch.Tensor (or anything with .detach().cpu().numpy()) to
    a contiguous numpy array. Falls back to np.asarray for plain arrays.
    """
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    arr = np.ascontiguousarray(t)
    return arr


def materialize(
    bindings: list[WeightBinding],
    state_dict: Mapping[str, object],
    dtype: str = "float32",
) -> dict[str, np.ndarray]:
    """Resolve HF tensors and apply per-binding transforms.

    Returns an ordered ``dict`` where keys follow NNTrainer's
    ``layer:role`` convention. Order matches ``bindings`` so the resulting
    safetensors offsets are stable across runs (helpful for diffs).
    """
    if dtype not in ("float32", "float16"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    target = np.float32 if dtype == "float32" else np.float16

    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for b in bindings:
        if b.hf_key not in state_dict:
            missing.append(b.hf_key)
            continue
        arr = _to_numpy(state_dict[b.hf_key])
        if b.transpose:
            if arr.ndim != 2:
                raise ValueError(
                    f"Cannot transpose non-2D tensor {b.hf_key!r} "
                    f"(shape={arr.shape})"
                )
            arr = arr.T
            arr = np.ascontiguousarray(arr)
        if arr.dtype != target:
            arr = arr.astype(target, copy=False)
        out[b.nntr_key] = arr
    if missing:
        raise KeyError(
            f"{len(missing)} HF tensors missing from state_dict, "
            f"first few: {missing[:5]}"
        )
    return out


def write_safetensors(tensors: dict[str, np.ndarray], path: str) -> None:
    """Write tensors using the official ``safetensors`` package.

    We delegate to the upstream library so the exact JSON header layout
    (``data_offsets`` per entry plus a 64-bit length prefix) matches what
    NNTrainer's loader parses in ``neuralnet.cpp``.
    """
    # Imported lazily so unit tests that do not write files can run without
    # the safetensors package installed.
    from safetensors.numpy import save_file

    # safetensors requires contiguous arrays of supported dtypes.
    sanitized: dict[str, np.ndarray] = {}
    for name, arr in tensors.items():
        if arr.dtype not in _DTYPE_TAGS:
            raise ValueError(
                f"Unsupported dtype {arr.dtype} for tensor {name!r}; "
                f"convert to float32 or float16 first"
            )
        sanitized[name] = np.ascontiguousarray(arr)
    save_file(sanitized, path)
