"""Helpers for FlexAttention mask construction."""

from __future__ import annotations

import time
from typing import Callable

import torch

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:  # pragma: no cover - depends on torch build
    BlockMask = None


_WINDOW_SINK_BACKEND_KEY = "flex_attention_window_sink"
_WINDOW_SINK_BLOCK_MASK_CACHE: dict[tuple[object, ...], "BlockMask"] = {}
_WINDOW_SINK_BLOCK_LAYOUT_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
_WINDOW_SINK_BLOCK_MASK_CACHE_HITS = 0
_WINDOW_SINK_BLOCK_MASK_CACHE_MISSES = 0
_WINDOW_SINK_BLOCK_LAYOUT_CACHE_HITS = 0
_WINDOW_SINK_BLOCK_LAYOUT_CACHE_MISSES = 0
_WINDOW_SINK_BLOCK_MASK_CACHE_LOOKUP_SECONDS = 0.0
_WINDOW_SINK_BLOCK_LAYOUT_CACHE_LOOKUP_SECONDS = 0.0
_WINDOW_SINK_BLOCK_LAYOUT_BUILD_SECONDS = 0.0
_WINDOW_SINK_BLOCK_MASK_BUILD_SECONDS = 0.0


def _device_cache_key(device: torch.device) -> object:
    """Build a stable cache key for a device."""

    if device.index is None:
        return device.type
    return (device.type, device.index)


def make_recent_sink_mask_mod(window_size: int, sink_tokens: int) -> Callable:
    """Build a recent-window plus sink-token mask_mod compatible with create_block_mask."""

    def mask_mod(batch_idx: torch.Tensor, head_idx: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        del batch_idx, head_idx
        causal = q_idx >= kv_idx
        allow_sink = kv_idx < sink_tokens
        allow_recent = (q_idx - kv_idx) < window_size
        return causal & (allow_sink | allow_recent)

    return mask_mod


def resolve_window_sink_block_shape(query_length: int, block_size: int) -> tuple[int, int]:
    """Choose an asymmetric BlockMask shape for decode-friendly workloads."""

    if query_length <= 1:
        return 1, int(block_size)
    return min(int(query_length), int(block_size)), int(block_size)


def bucket_num_blocks(length: int, block_size: int) -> int:
    """Bucket a sequence length into block units."""

    return max(1, (int(length) + int(block_size) - 1) // int(block_size))


def build_window_sink_block_layout(
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    window_size: int,
    sink_tokens: int,
    q_block_size: int,
    kv_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cached KV block tables for recent-window plus sink-token sparsity."""

    global _WINDOW_SINK_BLOCK_LAYOUT_CACHE_HITS, _WINDOW_SINK_BLOCK_LAYOUT_CACHE_MISSES
    global _WINDOW_SINK_BLOCK_LAYOUT_CACHE_LOOKUP_SECONDS, _WINDOW_SINK_BLOCK_LAYOUT_BUILD_SECONDS

    num_q_blocks = bucket_num_blocks(query_length, q_block_size)
    num_kv_blocks = bucket_num_blocks(key_length, kv_block_size)
    layout_key = (
        _WINDOW_SINK_BACKEND_KEY,
        _device_cache_key(device),
        int(batch_size),
        int(num_q_blocks),
        int(num_kv_blocks),
        int(window_size),
        int(sink_tokens),
        int(q_block_size),
        int(kv_block_size),
    )
    cache_lookup_started_at = time.perf_counter()
    cached = _WINDOW_SINK_BLOCK_LAYOUT_CACHE.get(layout_key)
    _WINDOW_SINK_BLOCK_LAYOUT_CACHE_LOOKUP_SECONDS += time.perf_counter() - cache_lookup_started_at
    if cached is not None:
        _WINDOW_SINK_BLOCK_LAYOUT_CACHE_HITS += 1
        return cached

    build_started_at = time.perf_counter()
    sink_block_count = 0 if sink_tokens <= 0 else bucket_num_blocks(sink_tokens, kv_block_size)
    row_blocks: list[list[int]] = []
    max_blocks_per_row = 1

    for q_block_idx in range(num_q_blocks):
        q_start = q_block_idx * q_block_size
        q_end = min(query_length, q_start + q_block_size)
        recent_start = max(0, q_start - window_size + 1)
        recent_end = max(0, q_end)

        block_indices = set(range(sink_block_count))
        if recent_end > 0:
            recent_start_block = recent_start // kv_block_size
            recent_end_block = (recent_end - 1) // kv_block_size
            block_indices.update(range(recent_start_block, recent_end_block + 1))

        filtered = sorted(block_idx for block_idx in block_indices if 0 <= block_idx < num_kv_blocks)
        row_blocks.append(filtered)
        max_blocks_per_row = max(max_blocks_per_row, len(filtered))

    kv_num_blocks = torch.zeros((batch_size, 1, num_q_blocks), dtype=torch.int32, device=device)
    kv_indices = torch.zeros((batch_size, 1, num_q_blocks, max_blocks_per_row), dtype=torch.int32, device=device)

    for row_idx, blocks in enumerate(row_blocks):
        kv_num_blocks[:, 0, row_idx] = len(blocks)
        if blocks:
            kv_indices[:, 0, row_idx, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)

    _WINDOW_SINK_BLOCK_LAYOUT_CACHE[layout_key] = (kv_num_blocks, kv_indices)
    _WINDOW_SINK_BLOCK_LAYOUT_CACHE_MISSES += 1
    _WINDOW_SINK_BLOCK_LAYOUT_BUILD_SECONDS += time.perf_counter() - build_started_at
    return kv_num_blocks, kv_indices


def build_window_sink_block_mask(
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    window_size: int,
    sink_tokens: int,
    block_size: int,
) -> "BlockMask":
    """Build a BlockMask for recent-window plus sink-token attention.

    This prefers a direct ``BlockMask.from_kv_blocks(...)`` path and reuses
    bucketed KV block layouts across layers and across nearby decode lengths.
    """

    global _WINDOW_SINK_BLOCK_MASK_CACHE_HITS, _WINDOW_SINK_BLOCK_MASK_CACHE_MISSES
    global _WINDOW_SINK_BLOCK_MASK_CACHE_LOOKUP_SECONDS, _WINDOW_SINK_BLOCK_MASK_BUILD_SECONDS

    if BlockMask is None:
        raise RuntimeError("torch.nn.attention.flex_attention.BlockMask is unavailable in the current PyTorch build.")
    if query_length < 1 or key_length < 1:
        raise ValueError("BlockMask construction requires query_length >= 1 and key_length >= 1.")

    q_block_size, kv_block_size = resolve_window_sink_block_shape(
        query_length=int(query_length),
        block_size=int(block_size),
    )
    query_num_blocks = bucket_num_blocks(int(query_length), int(q_block_size))
    key_num_blocks = bucket_num_blocks(int(key_length), int(kv_block_size))
    exact_mask_key = (
        _WINDOW_SINK_BACKEND_KEY,
        _device_cache_key(device),
        int(batch_size),
        int(query_length),
        int(key_length),
        int(window_size),
        int(sink_tokens),
        int(q_block_size),
        int(kv_block_size),
        int(query_num_blocks),
        int(key_num_blocks),
    )
    cache_lookup_started_at = time.perf_counter()
    cached = _WINDOW_SINK_BLOCK_MASK_CACHE.get(exact_mask_key)
    _WINDOW_SINK_BLOCK_MASK_CACHE_LOOKUP_SECONDS += time.perf_counter() - cache_lookup_started_at
    if cached is not None:
        _WINDOW_SINK_BLOCK_MASK_CACHE_HITS += 1
        return cached

    kv_num_blocks, kv_indices = build_window_sink_block_layout(
        batch_size=int(batch_size),
        query_length=int(query_length),
        key_length=int(key_length),
        device=device,
        window_size=int(window_size),
        sink_tokens=int(sink_tokens),
        q_block_size=int(q_block_size),
        kv_block_size=int(kv_block_size),
    )
    mask_mod = make_recent_sink_mask_mod(window_size=window_size, sink_tokens=sink_tokens)
    build_started_at = time.perf_counter()
    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        BLOCK_SIZE=(q_block_size, kv_block_size),
        mask_mod=mask_mod,
        seq_lengths=(int(query_length), int(key_length)),
    )
    _WINDOW_SINK_BLOCK_MASK_BUILD_SECONDS += time.perf_counter() - build_started_at
    _WINDOW_SINK_BLOCK_MASK_CACHE[exact_mask_key] = block_mask
    _WINDOW_SINK_BLOCK_MASK_CACHE_MISSES += 1
    return block_mask


def get_window_sink_block_mask_cache_stats() -> dict[str, int]:
    """Return cache statistics for window-sink BlockMask construction."""

    return {
        "mask_entries": len(_WINDOW_SINK_BLOCK_MASK_CACHE),
        "mask_hits": _WINDOW_SINK_BLOCK_MASK_CACHE_HITS,
        "mask_misses": _WINDOW_SINK_BLOCK_MASK_CACHE_MISSES,
        "layout_entries": len(_WINDOW_SINK_BLOCK_LAYOUT_CACHE),
        "layout_hits": _WINDOW_SINK_BLOCK_LAYOUT_CACHE_HITS,
        "layout_misses": _WINDOW_SINK_BLOCK_LAYOUT_CACHE_MISSES,
    }


def get_window_sink_block_mask_perf_stats() -> dict[str, float]:
    """Return cumulative timing counters for BlockMask construction internals."""

    return {
        "mask_cache_lookup_seconds": _WINDOW_SINK_BLOCK_MASK_CACHE_LOOKUP_SECONDS,
        "layout_cache_lookup_seconds": _WINDOW_SINK_BLOCK_LAYOUT_CACHE_LOOKUP_SECONDS,
        "layout_build_seconds": _WINDOW_SINK_BLOCK_LAYOUT_BUILD_SECONDS,
        "block_mask_build_seconds": _WINDOW_SINK_BLOCK_MASK_BUILD_SECONDS,
    }
