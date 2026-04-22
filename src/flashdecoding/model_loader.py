"""Model and tokenizer loading helpers."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backends import BackendResolution, resolve_backend


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(requested_device: str) -> torch.device:
    """Resolve a CLI device string to a torch.device."""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return torch.device(requested_device)


def resolve_dtype(requested_dtype: str, device: torch.device) -> torch.dtype:
    """Resolve a CLI dtype string to a torch dtype."""

    if requested_dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return _DTYPE_MAP[requested_dtype]


def load_model_and_tokenizer(
    model_name: str,
    backend_name: str,
    requested_device: str,
    requested_dtype: str,
) -> tuple[Any, Any, torch.device, torch.dtype, BackendResolution]:
    """Load tokenizer and model for single-example causal decoding."""

    device = resolve_device(requested_device)
    dtype = resolve_dtype(requested_dtype, device)
    backend = resolve_backend(name=backend_name, device=device.type)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if backend.hf_attn_implementation is not None:
        model_kwargs["attn_implementation"] = backend.hf_attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    return model, tokenizer, device, dtype, backend
