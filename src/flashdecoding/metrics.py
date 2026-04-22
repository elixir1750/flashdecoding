"""Timing and memory helpers for decoding benchmarks."""

from __future__ import annotations

import platform
import resource

import torch


def compute_tokens_per_second(generated_tokens: int, elapsed_seconds: float | None) -> float | None:
    """Return overall generated tokens per second."""

    if elapsed_seconds is None or elapsed_seconds <= 0 or generated_tokens <= 0:
        return None
    return generated_tokens / elapsed_seconds


def compute_tpot_seconds(
    generated_tokens: int,
    elapsed_seconds: float | None,
    ttft_seconds: float | None,
) -> float | None:
    """Return average per-token latency after the first generated token."""

    if (
        elapsed_seconds is None
        or ttft_seconds is None
        or generated_tokens <= 1
        or elapsed_seconds <= ttft_seconds
    ):
        return None
    return (elapsed_seconds - ttft_seconds) / (generated_tokens - 1)


def synchronize_if_needed(device: torch.device) -> None:
    """Synchronize CUDA work before reading timing results."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    """Reset the peak memory tracker before a benchmarked run."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_bytes(device: torch.device) -> tuple[int, str]:
    """Return peak memory in bytes and the data source used."""

    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device)), "torch.cuda.max_memory_allocated"

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(usage), "resource.ru_maxrss_bytes"
    return int(usage * 1024), "resource.ru_maxrss_kib"
