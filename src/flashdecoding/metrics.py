"""Timing and memory helpers for decoding benchmarks."""

from __future__ import annotations

import platform
from typing import Final

import torch

try:
    import resource  # type: ignore
except ImportError:  # pragma: no cover - only triggered on platforms without resource
    resource = None


_WINDOWS: Final[str] = "Windows"


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


def _get_windows_peak_working_set_bytes() -> tuple[int, str]:
    """Return peak working set size via Win32 APIs on Windows."""

    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE

    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    process_handle = get_current_process()

    ok = get_process_memory_info(process_handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")

    return int(counters.PeakWorkingSetSize), "GetProcessMemoryInfo.PeakWorkingSetSize"


def _get_process_peak_rss_bytes() -> tuple[int, str]:
    """Return peak process memory in bytes using a platform-appropriate source."""

    system = platform.system()
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if system == "Darwin":
            return int(usage), "resource.ru_maxrss_bytes"
        return int(usage * 1024), "resource.ru_maxrss_kib"

    if system == _WINDOWS:
        return _get_windows_peak_working_set_bytes()

    raise RuntimeError(
        "Peak CPU memory measurement is unavailable on this platform because neither the resource module nor the Windows memory API path is available."
    )


def get_peak_memory_bytes(device: torch.device) -> tuple[int, str]:
    """Return peak memory in bytes and the data source used."""

    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device)), "torch.cuda.max_memory_allocated"

    return _get_process_peak_rss_bytes()
