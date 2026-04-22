"""Model and tokenizer loading helpers for backend-aware decoding."""

from __future__ import annotations

import types
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_neox import modeling_gpt_neox

from .backends import BackendNotAvailableError, BackendResolution, update_support_with_runtime_result, resolve_backend


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


_ORIGINAL_GPT_NEOX_CREATE_CAUSAL_MASK = modeling_gpt_neox.create_causal_mask
_GPT_NEOX_FLEX_MASK_PATCHED = False

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


def make_recent_sink_mask(window_size: int, sink_tokens: int):
    """Build a causal-compatible recent-window plus sink-token mask."""

    def mask_fn(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        allow_sink = kv_idx < sink_tokens
        allow_recent = (q_idx - kv_idx) < window_size
        return allow_sink | allow_recent

    return mask_fn


def ensure_gpt_neox_flex_mask_patch() -> None:
    """Patch GPTNeoX causal mask creation so config-driven FlexAttention overlays can be injected."""

    global _GPT_NEOX_FLEX_MASK_PATCHED
    if _GPT_NEOX_FLEX_MASK_PATCHED:
        return

    def patched_create_causal_mask(
        config: Any,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values: Any,
        position_ids: torch.Tensor | None = None,
        or_mask_function: Any = None,
        and_mask_function: Any = None,
    ):
        experiment = getattr(config, "_flex_attention_experiment", None)
        if experiment == "window_sink":
            window_size = getattr(config, "_flex_window_size", None)
            sink_tokens = getattr(config, "_flex_sink_tokens", None)
            if window_size is None or sink_tokens is None:
                raise ValueError("flex_attention_window_sink requires both _flex_window_size and _flex_sink_tokens in the config.")
            experiment_mask = make_recent_sink_mask(window_size=window_size, sink_tokens=sink_tokens)
            if and_mask_function is None:
                and_mask_function = experiment_mask
            else:
                from transformers.masking_utils import and_masks

                and_mask_function = and_masks(and_mask_function, experiment_mask)

        return _ORIGINAL_GPT_NEOX_CREATE_CAUSAL_MASK(
            config=config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
        )

    modeling_gpt_neox.create_causal_mask = patched_create_causal_mask
    _GPT_NEOX_FLEX_MASK_PATCHED = True


def configure_flex_attention_experiment(model: Any, backend_name: str, flex_window_size: int, flex_sink_tokens: int) -> None:
    """Attach experimental FlexAttention settings to the loaded model config."""

    if backend_name != "flex_attention_window_sink":
        return
    if flex_window_size < 1:
        raise ValueError("flex_attention_window_sink requires --flex-window-size >= 1.")
    if flex_sink_tokens < 0:
        raise ValueError("flex_attention_window_sink requires --flex-sink-tokens >= 0.")

    ensure_gpt_neox_flex_mask_patch()
    model.config._flex_attention_experiment = "window_sink"
    model.config._flex_window_size = int(flex_window_size)
    model.config._flex_sink_tokens = int(flex_sink_tokens)


@torch.inference_mode()
def run_backend_smoke_test(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    backend: BackendResolution,
) -> BackendResolution:
    """Run a tiny local runtime smoke test for backends that need honest runtime validation."""

    if backend.name not in {"flex_attention", "flex_attention_window_sink"}:
        return backend
    if backend.support_report is None:
        return backend

    encoded = tokenizer("Hello", return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        if outputs.logits is None:
            raise RuntimeError("Smoke test forward pass returned no logits.")
    except Exception as exc:
        support_report = update_support_with_runtime_result(
            backend.support_report,
            local_runtime_support=False,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
        raise BackendNotAvailableError(support_report.format_multiline(), support_report=support_report) from exc

    support_report = update_support_with_runtime_result(
        backend.support_report,
        local_runtime_support=True,
    )
    return BackendResolution(
        name=backend.name,
        hf_attn_implementation=backend.hf_attn_implementation,
        notes=backend.notes,
        support_report=support_report,
    )


def load_model_and_tokenizer(
    model_name: str,
    backend_name: str,
    requested_device: str,
    requested_dtype: str,
    local_files_only: bool = False,
    flex_window_size: int = 256,
    flex_sink_tokens: int = 4,
) -> tuple[Any, Any, torch.device, torch.dtype, BackendResolution]:
    """Load tokenizer and model for backend-aware single-example decoding."""

    device = resolve_device(requested_device)
    dtype = resolve_dtype(requested_dtype, device)
    backend = resolve_backend(name=backend_name, device=device.type)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "local_files_only": local_files_only,
    }
    if backend.hf_attn_implementation is not None:
        model_kwargs["attn_implementation"] = backend.hf_attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    configure_flex_attention_experiment(
        model=model,
        backend_name=backend.name,
        flex_window_size=flex_window_size,
        flex_sink_tokens=flex_sink_tokens,
    )
    backend = run_backend_smoke_test(
        model=model,
        tokenizer=tokenizer,
        device=device,
        backend=backend,
    )

    return model, tokenizer, device, dtype, backend
