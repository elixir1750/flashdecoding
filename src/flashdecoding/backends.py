"""Attention backend dispatch for decoding experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class BackendNotAvailableError(RuntimeError):
    """Raised when a requested attention backend cannot be used."""


@dataclass(frozen=True)
class BackendResolution:
    """Resolved backend configuration for model loading."""

    name: str
    hf_attn_implementation: str | None
    notes: str


def detect_flash_decode_support(device: str) -> tuple[bool, str]:
    """Report whether the placeholder flash_decode backend is usable."""

    if device != "cuda":
        return False, "flash_decode is reserved for CUDA-oriented decoding backends and is unavailable on non-CUDA devices."
    if not torch.cuda.is_available():
        return False, "flash_decode requires CUDA, but torch.cuda.is_available() is False."
    return False, "flash_decode is only a placeholder in this scaffold. No real Flash-Decoding kernel/backend has been integrated yet."


def resolve_backend(name: str, device: str) -> BackendResolution:
    """Resolve a user-facing backend name to a concrete Hugging Face setting."""

    backend = name.lower()

    if backend == "vanilla":
        return BackendResolution(
            name="vanilla",
            hf_attn_implementation="eager",
            notes="Using Hugging Face eager attention for the stable baseline.",
        )

    if backend == "sdpa":
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise BackendNotAvailableError(
                "sdpa requires torch.nn.functional.scaled_dot_product_attention, but the current PyTorch build does not provide it."
            )
        return BackendResolution(
            name="sdpa",
            hf_attn_implementation="sdpa",
            notes="Using Hugging Face SDPA attention via attn_implementation='sdpa'.",
        )

    if backend == "flash_decode":
        supported, reason = detect_flash_decode_support(device=device)
        if not supported:
            raise BackendNotAvailableError(reason)
        return BackendResolution(
            name="flash_decode",
            hf_attn_implementation=None,
            notes="Reserved for a future Flash-Decoding-style backend.",
        )

    raise ValueError(f"Unknown backend '{name}'. Expected one of: vanilla, sdpa, flash_decode.")
