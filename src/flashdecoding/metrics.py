"""Timing and memory helpers for decoding benchmarks."""

from __future__ import annotations

import platform
import resource

import torch


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
