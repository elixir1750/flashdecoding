"""Attention backend dispatch for decoding experiments."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class BackendNotAvailableError(RuntimeError):
    """Raised when a requested attention backend cannot be used."""

    def __init__(self, message: str, support_report: "BackendSupportReport | None" = None):
        super().__init__(message)
        self.support_report = support_report


@dataclass(frozen=True)
class BackendResolution:
    """Resolved backend configuration for model loading."""

    name: str
    hf_attn_implementation: str | None
    notes: str
    support_report: "BackendSupportReport | None" = None


@dataclass(frozen=True)
class BackendSupportReport:
    """Structured support report for one backend on the current environment."""

    backend: str
    upstream_support: bool
    integration_support: bool
    local_runtime_support: bool | None
    recommended_device: str
    failure_reason: str | None = None
    details: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert the support report to a JSON-serializable dictionary."""

        return {
            "backend": self.backend,
            "upstream_support": self.upstream_support,
            "integration_support": self.integration_support,
            "local_runtime_support": self.local_runtime_support,
            "recommended_device": self.recommended_device,
            "failure_reason": self.failure_reason,
            "details": self.details,
        }

    def format_multiline(self) -> str:
        """Render the support report as a readable multiline message."""

        lines = [
            f"{self.backend} support report:",
            f"  upstream_support={self.upstream_support}",
            f"  integration_support={self.integration_support}",
            f"  local_runtime_support={self.local_runtime_support}",
            f"  recommended_device={self.recommended_device}",
            f"  failure_reason={self.failure_reason or '-'}",
        ]
        if self.details:
            lines.append(f"  details={self.details}")
        return "\n".join(lines)


def _module_available(module_name: str) -> bool:
    """Return whether a Python module can be imported."""

    return importlib.util.find_spec(module_name) is not None


def detect_flex_attention_support() -> BackendSupportReport:
    """Report upstream and integration support for the experimental flex_attention backend."""

    upstream_support = _module_available("torch.nn.attention.flex_attention")
    integration_support = "flex_attention" in ALL_ATTENTION_FUNCTIONS
    failure_reason = None
    if not upstream_support:
        failure_reason = "torch.nn.attention.flex_attention cannot be imported in the current PyTorch build."
    elif not integration_support:
        failure_reason = "Transformers does not expose attn_implementation='flex_attention' through ALL_ATTENTION_FUNCTIONS."

    return BackendSupportReport(
        backend="flex_attention",
        upstream_support=upstream_support,
        integration_support=integration_support,
        local_runtime_support=None,
        recommended_device="cuda",
        failure_reason=failure_reason,
        details="CUDA is recommended for performance experiments, but local runtime support should be confirmed by a smoke test on the requested device.",
    )


def update_support_with_runtime_result(
    report: BackendSupportReport,
    local_runtime_support: bool,
    failure_reason: str | None = None,
) -> BackendSupportReport:
    """Return a support report with local runtime smoke-test status filled in."""

    return BackendSupportReport(
        backend=report.backend,
        upstream_support=report.upstream_support,
        integration_support=report.integration_support,
        local_runtime_support=local_runtime_support,
        recommended_device=report.recommended_device,
        failure_reason=failure_reason if not local_runtime_support else None,
        details=report.details,
    )


def detect_flash_decode_support(device: str) -> tuple[bool, str]:
    """Report whether the placeholder flash_decode backend is usable."""

    checks = {
        "cuda_device_requested": device == "cuda",
        "torch_cuda_available": torch.cuda.is_available(),
        "flash_attn_installed": _module_available("flash_attn"),
        "flash_attn_3_installed": _module_available("flash_attn_3"),
        "flex_attention_importable": _module_available("torch.nn.attention.flex_attention"),
    }
    reason = (
        "flash_decode is a placeholder and is not implemented in this project yet. "
        f"Capability snapshot: cuda_device_requested={checks['cuda_device_requested']}, "
        f"torch_cuda_available={checks['torch_cuda_available']}, "
        f"flash_attn_installed={checks['flash_attn_installed']}, "
        f"flash_attn_3_installed={checks['flash_attn_3_installed']}, "
        f"flex_attention_importable={checks['flex_attention_importable']}."
    )
    return False, reason


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

    if backend == "flex_attention":
        support_report = detect_flex_attention_support()
        if not support_report.upstream_support or not support_report.integration_support:
            raise BackendNotAvailableError(support_report.format_multiline(), support_report=support_report)
        return BackendResolution(
            name="flex_attention",
            hf_attn_implementation="flex_attention",
            notes="Using Hugging Face flex_attention as an experimental backend. This is not equivalent to true Flash-Decoding. CUDA is recommended for performance experiments, but runtime support is determined by a local smoke test.",
            support_report=support_report,
        )

    if backend == "flex_attention_window_sink":
        support_report = detect_flex_attention_support()
        if not support_report.upstream_support or not support_report.integration_support:
            raise BackendNotAvailableError(support_report.format_multiline(), support_report=support_report)
        return BackendResolution(
            name="flex_attention_window_sink",
            hf_attn_implementation="flex_attention",
            notes="Using Hugging Face flex_attention with an experimental recent-window plus sink-token mask. This is a FlexAttention/FlexDecoding-style experiment, not true Flash-Decoding. CUDA is recommended for performance experiments, but runtime support is determined by a local smoke test.",
            support_report=support_report,
        )

    if backend == "flash_decode":
        supported, reason = detect_flash_decode_support(device=device)
        if not supported:
            raise BackendNotAvailableError(reason)
        return BackendResolution(
            name="flash_decode",
            hf_attn_implementation=None,
            notes="Placeholder only. flash_decode is not implemented in this project.",
        )

    raise ValueError(
        f"Unknown backend '{name}'. Expected one of: vanilla, sdpa, flex_attention, flex_attention_window_sink, flash_decode."
    )
