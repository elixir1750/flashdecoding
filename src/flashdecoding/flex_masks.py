"""Helpers for FlexAttention mask construction."""

from __future__ import annotations

from typing import Callable

import torch

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
except ImportError:  # pragma: no cover - depends on torch build
    BlockMask = None
    create_block_mask = None


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
) -> "BlockMask":
    """Build a BlockMask for recent-window plus sink-token attention."""

    if create_block_mask is None or BlockMask is None:
        raise RuntimeError("torch.nn.attention.flex_attention.create_block_mask is unavailable in the current PyTorch build.")
    if query_length < 1 or key_length < 1:
        raise ValueError("BlockMask construction requires query_length >= 1 and key_length >= 1.")

    mask_mod = make_recent_sink_mask_mod(window_size=window_size, sink_tokens=sink_tokens)
    return create_block_mask(
        mask_mod=mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=device,
        _compile=False,
    )
