"""Helpers for FlexAttention mask construction."""

from __future__ import annotations

from typing import Callable

import torch

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
except ImportError:  # pragma: no cover - depends on torch build
    BlockMask = None
    create_block_mask = None


_WINDOW_SINK_BLOCK_MASK_CACHE: dict[tuple[object, int, int, int, int, int, int], "BlockMask"] = {}
_WINDOW_SINK_BLOCK_MASK_CACHE_HITS = 0
_WINDOW_SINK_BLOCK_MASK_CACHE_MISSES = 0


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


def build_window_sink_block_mask(
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    window_size: int,
    sink_tokens: int,
    block_size: int,
) -> "BlockMask":
    """Build a BlockMask for recent-window plus sink-token attention."""

    global _WINDOW_SINK_BLOCK_MASK_CACHE_HITS, _WINDOW_SINK_BLOCK_MASK_CACHE_MISSES

    if create_block_mask is None or BlockMask is None:
        raise RuntimeError("torch.nn.attention.flex_attention.create_block_mask is unavailable in the current PyTorch build.")
    if query_length < 1 or key_length < 1:
        raise ValueError("BlockMask construction requires query_length >= 1 and key_length >= 1.")

    cache_key = (
        _device_cache_key(device),
        int(batch_size),
        int(query_length),
        int(key_length),
        int(window_size),
        int(sink_tokens),
        int(block_size),
    )
    cached = _WINDOW_SINK_BLOCK_MASK_CACHE.get(cache_key)
    if cached is not None:
        _WINDOW_SINK_BLOCK_MASK_CACHE_HITS += 1
        return cached

    mask_mod = make_recent_sink_mask_mod(window_size=window_size, sink_tokens=sink_tokens)
    block_mask = create_block_mask(
        mask_mod=mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=device,
        BLOCK_SIZE=int(block_size),
        _compile=False,
    )
    _WINDOW_SINK_BLOCK_MASK_CACHE[cache_key] = block_mask
    _WINDOW_SINK_BLOCK_MASK_CACHE_MISSES += 1
    return block_mask


def get_window_sink_block_mask_cache_stats() -> dict[str, int]:
    """Return cache statistics for window-sink BlockMask construction."""

    return {
        "entries": len(_WINDOW_SINK_BLOCK_MASK_CACHE),
        "hits": _WINDOW_SINK_BLOCK_MASK_CACHE_HITS,
        "misses": _WINDOW_SINK_BLOCK_MASK_CACHE_MISSES,
    }
