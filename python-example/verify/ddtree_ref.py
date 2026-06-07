from __future__ import annotations
# AUTO-EXTRACTED verbatim from python-example/ddtree.py (tree + mask + compact
# functions) with trivial stubs. __future__ annotations defers type-hint eval.
import heapq
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

DDTREE_TREE_BUILD_STAGE_ORDER = ("tree_build_copy", "tree_build_heap", "tree_build_visibility")
_CPP_COMPACT_ENABLED = False

def cuda_time():
    return 0.0

def empty_stage_times(order):
    return {k: 0.0 for k in order}

def load_cpp_compact_module():
    return None

def get_model_text_config(target):
    cfg = target.config
    return getattr(cfg, "text_config", cfg)


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor, dict[str, float]]:
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    copy_start = cuda_time()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    heap_start = time.perf_counter()
    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs_np[depth - 1, rank]) + float(top_log_probs_np[depth - 1, rank + 1])
            heapq.heappush(heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw))

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility, build_subtimes


def compile_ddtree_tree(
    root_token_id: torch.Tensor,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    current_length = 1 + int(node_token_ids.numel())

    attention_mask_buffer.zero_()

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids, non_blocking=False)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths, non_blocking=False)
        verify_position_ids[0, 1:current_length].add_(start)

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(visibility_cpu, non_blocking=False)

    tree_block = attention_mask_buffer[0, 0, :current_length, past_length : past_length + current_length]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)

    attention_mask = attention_mask_buffer[:, :, :current_length, : past_length + current_length]
    return verify_input_ids, verify_position_ids, attention_mask, past_length, current_length


def follow_verified_tree(child_maps: list[dict[int, int]], posterior: torch.Tensor) -> tuple[list[int], int]:
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def prepare_ddtree_attention_mask_for_target(
    target: AutoModelForCausalLM,
    attention_mask: torch.Tensor,
    verify_position_ids: torch.Tensor,
    past_length: int,
    current_length: int,
) -> torch.Tensor | dict[str, torch.Tensor]:
    target_config = get_model_text_config(target)
    layer_types = getattr(target_config, "layer_types", None)
    if not layer_types or "sliding_attention" not in layer_types:
        return attention_mask

    sliding_window = getattr(target_config, "sliding_window", None)
    if sliding_window is None or sliding_window <= 0:
        return {
            "full_attention": attention_mask,
            "sliding_attention": attention_mask,
        }

    kv_length = past_length + current_length
    key_positions = torch.arange(kv_length, dtype=verify_position_ids.dtype, device=verify_position_ids.device)
    key_positions[past_length:kv_length] = verify_position_ids[0, :current_length]
    query_positions = verify_position_ids[0, :current_length]
    sliding_visible = (
        (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
        & (key_positions.unsqueeze(0) > (query_positions.unsqueeze(1) - int(sliding_window)))
    )

    sliding_attention_mask = attention_mask.clone()
    sliding_attention_mask[0, 0, :current_length, :kv_length].masked_fill_(
        ~sliding_visible,
        torch.finfo(attention_mask.dtype).min,
    )
    return {
        "full_attention": attention_mask,
        "sliding_attention": sliding_attention_mask,
    }


def _compact_appended_window(
    cache_tensor: torch.Tensor,
    past_length: int,
    keep_current_indices: torch.Tensor,
    expected_current_length: int | None = None,
) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if expected_current_length is not None and current_length != expected_current_length:
        raise RuntimeError(
            "DDTree cache compaction expects a full dense target cache whose "
            "physical KV tail is exactly the verified tree window. Got "
            f"past_length={past_length}, cache_seq_len={cache_tensor.shape[-2]}, "
            f"tail_length={current_length}, expected_tail_length={expected_current_length}. "
            "For Gemma4, create the target cache with DynamicCache() rather than "
            "DynamicCache(config=...), because HF sliding-window cache layers only "
            "store a window tail and cannot be compacted by absolute DDTree slots."
        )
    if current_length <= 0:
        return

    keep_count = keep_current_indices.numel()
    if keep_count > current_length:
        raise RuntimeError(
            f"DDTree accepted {keep_count} cache indices from a verified tail of "
            f"length {current_length}."
        )
    if keep_count > 0:
        max_keep_index = int(keep_current_indices.max().item())
        min_keep_index = int(keep_current_indices.min().item())
        if min_keep_index < 0 or max_keep_index >= current_length:
            raise RuntimeError(
                "DDTree accepted cache index outside the verified tail: "
                f"min={min_keep_index}, max={max_keep_index}, tail_length={current_length}."
            )
    if keep_count == 0:
        return
    if keep_count == current_length:
        identity_indices = torch.arange(
            current_length,
            dtype=keep_current_indices.dtype,
            device=keep_current_indices.device,
        )
        if bool(torch.equal(keep_current_indices, identity_indices)):
            return
        kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(-2, keep_current_indices)
        cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)
        return

    if _CPP_COMPACT_ENABLED:
        module = load_cpp_compact_module()
        if module is not None:
            module.compact_tail_inplace(cache_tensor, past_length, keep_current_indices)
            return

    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(-2, keep_current_indices)
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def compact_dynamic_cache(
    past_key_values: DynamicCache,
    past_length: int,
    keep_current_indices: list[int],
    expected_current_length: int | None = None,
) -> None:
    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(keep_current_indices, dtype=torch.long, device=device)
        return keep_tensor_by_device[device]

    seen_cache_views: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()

    def cache_view_key(cache_tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
        return (
            cache_tensor.untyped_storage().data_ptr(),
            cache_tensor.storage_offset(),
            tuple(cache_tensor.shape),
            tuple(cache_tensor.stride()),
        )

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            if key_cache is None or value_cache is None or key_cache.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(key_cache.device)
            for cache_tensor in (key_cache, value_cache):
                view_key = cache_view_key(cache_tensor)
                if view_key in seen_cache_views:
                    continue
                seen_cache_views.add(view_key)
                _compact_appended_window(
                    cache_tensor,
                    past_length,
                    keep_tensor,
                    expected_current_length=expected_current_length,
                )
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            for cache_tensor in (layer.keys, layer.values):
                view_key = cache_view_key(cache_tensor)
                if view_key in seen_cache_views:
                    continue
                seen_cache_views.add(view_key)
                _compact_appended_window(
                    cache_tensor,
                    past_length,
                    keep_tensor,
                    expected_current_length=expected_current_length,
                )
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError("Unsupported DynamicCache layout for DDTree cache compaction.")

