"""Rich-based terminal UI helpers for backend comparison demos."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .metrics import compute_tokens_per_second, compute_tpot_seconds


@dataclass
class DemoPaneState:
    """Render state for one backend comparison pane."""

    backend: str
    model_name: str
    status: str = "waiting"
    notes: str = ""
    device: str = ""
    dtype: str = ""
    prompt_tokens: int | None = None
    generated_tokens: int = 0
    ttft_seconds: float | None = None
    elapsed_seconds: float | None = None
    total_latency_seconds: float | None = None
    peak_memory_bytes: int | None = None
    text: str = ""
    error: str | None = None
    failure_reason: str | None = None
    support_report: dict[str, object] | None = None
    flex_window_size: int | None = None
    flex_sink_tokens: int | None = None


def _format_seconds(value: float | None) -> str:
    """Format timing values for terminal display."""

    if value is None:
        return "-"
    return f"{value:.3f}s"


def _format_toks_per_second(value: float | None) -> str:
    """Format throughput values for terminal display."""

    if value is None:
        return "-"
    return f"{value:.2f} tok/s"


def _format_memory(value: int | None) -> str:
    """Format memory values for terminal display."""

    if value is None:
        return "-"
    return f"{value / (1024 ** 2):.1f} MiB"


def _status_style(status: str) -> str:
    """Map a pane status to a Rich style."""

    if status == "done":
        return "green"
    if status == "error":
        return "bold red"
    if status == "running":
        return "cyan"
    if status == "loading":
        return "yellow"
    return "white"


def _format_support_value(value: object | None) -> str:
    """Format support report values for display."""

    if value is None:
        return "-"
    return str(value)


def render_support_report(state: DemoPaneState) -> RenderableType | None:
    """Render support-layer status for FlexAttention-style backends."""

    if state.support_report is None:
        return None

    support = Table.grid(padding=(0, 1))
    support.add_column(style="bold")
    support.add_column()
    support.add_row("Upstream", _format_support_value(state.support_report.get("upstream_support")))
    support.add_row("Integration", _format_support_value(state.support_report.get("integration_support")))
    support.add_row("Local Runtime", _format_support_value(state.support_report.get("local_runtime_support")))
    support.add_row("Recommended", _format_support_value(state.support_report.get("recommended_device")))

    failure_reason = state.failure_reason or state.support_report.get("failure_reason")
    details = state.support_report.get("details")
    detail_lines = []
    if failure_reason:
        detail_lines.append(f"failure_reason={failure_reason}")
    if details:
        detail_lines.append(f"details={details}")

    detail_text = Text("\n".join(detail_lines) if detail_lines else "No extra support details.", overflow="fold")
    return Panel(Group(support, Text(""), detail_text), title="Support Report", border_style="magenta")


def render_demo_view(prompt: str, left: DemoPaneState, right: DemoPaneState) -> RenderableType:
    """Render the live side-by-side comparison view."""

    prompt_text = Text(prompt.strip() or "(empty prompt)", style="bold")
    panes = Table.grid(expand=True)
    panes.add_column(ratio=1)
    panes.add_column(ratio=1)
    panes.add_row(render_pane(left), render_pane(right))
    return Group(
        Panel(prompt_text, title="Prompt", border_style="blue"),
        panes,
    )


def render_pane(state: DemoPaneState) -> Panel:
    """Render one backend pane."""

    metrics = Table.grid(padding=(0, 1))
    metrics.add_column(style="bold")
    metrics.add_column()
    metrics.add_row("Status", Text(state.status, style=_status_style(state.status)))
    metrics.add_row("Model", state.model_name)
    metrics.add_row("Backend", state.backend)
    metrics.add_row("Device", state.device or "-")
    metrics.add_row("DType", state.dtype or "-")
    metrics.add_row("Elapsed", _format_seconds(state.elapsed_seconds))
    metrics.add_row("TTFT", _format_seconds(state.ttft_seconds))
    metrics.add_row("Tokens", str(state.generated_tokens))
    metrics.add_row(
        "Tok/s",
        _format_toks_per_second(compute_tokens_per_second(state.generated_tokens, state.elapsed_seconds)),
    )
    metrics.add_row(
        "TPOT",
        _format_seconds(compute_tpot_seconds(state.generated_tokens, state.elapsed_seconds, state.ttft_seconds)),
    )
    metrics.add_row("Peak Mem", _format_memory(state.peak_memory_bytes))
    if state.flex_window_size is not None:
        metrics.add_row("Window", str(state.flex_window_size))
    if state.flex_sink_tokens is not None:
        metrics.add_row("Sink Tokens", str(state.flex_sink_tokens))

    body = Text(state.text or "(no text yet)", overflow="fold")
    if state.error:
        body = Text(state.error, style="bold red", overflow="fold")

    renderables: list[RenderableType] = [metrics]
    support_panel = render_support_report(state)
    if support_panel is not None:
        renderables.extend([Text(""), support_panel])
    renderables.extend([Text(""), body])
    group = Group(*renderables)
    title = f"{state.backend} [{state.status}]"
    return Panel(group, title=title, border_style=_status_style(state.status))


def render_summary(left: DemoPaneState, right: DemoPaneState) -> RenderableType:
    """Render the final side-by-side summary table."""

    table = Table(title="Final Comparison Summary", expand=True)
    table.add_column("Metric", style="bold")
    table.add_column(left.backend)
    table.add_column(right.backend)

    left_toks = compute_tokens_per_second(left.generated_tokens, left.total_latency_seconds or left.elapsed_seconds)
    right_toks = compute_tokens_per_second(right.generated_tokens, right.total_latency_seconds or right.elapsed_seconds)
    left_tpot = compute_tpot_seconds(left.generated_tokens, left.total_latency_seconds or left.elapsed_seconds, left.ttft_seconds)
    right_tpot = compute_tpot_seconds(right.generated_tokens, right.total_latency_seconds or right.elapsed_seconds, right.ttft_seconds)

    table.add_row("Status", left.status, right.status)
    table.add_row("Model", left.model_name, right.model_name)
    table.add_row("Elapsed", _format_seconds(left.total_latency_seconds or left.elapsed_seconds), _format_seconds(right.total_latency_seconds or right.elapsed_seconds))
    table.add_row("TTFT", _format_seconds(left.ttft_seconds), _format_seconds(right.ttft_seconds))
    table.add_row("Generated Tokens", str(left.generated_tokens), str(right.generated_tokens))
    table.add_row("Tok/s", _format_toks_per_second(left_toks), _format_toks_per_second(right_toks))
    table.add_row("TPOT", _format_seconds(left_tpot), _format_seconds(right_tpot))
    table.add_row("Peak Memory", _format_memory(left.peak_memory_bytes), _format_memory(right.peak_memory_bytes))

    if left.error or right.error:
        error_text = Text()
        if left.error:
            error_text.append(f"{left.backend}: {left.error}\n", style="bold red")
            if left.failure_reason:
                error_text.append(f"  failure_reason={left.failure_reason}\n", style="red")
        if right.error:
            error_text.append(f"{right.backend}: {right.error}", style="bold red")
            if right.failure_reason:
                error_text.append(f"\n  failure_reason={right.failure_reason}", style="red")

        support_summary = Table(title="Support Layers", expand=True)
        support_summary.add_column("Layer", style="bold")
        support_summary.add_column(left.backend)
        support_summary.add_column(right.backend)
        for key, label in [
            ("upstream_support", "Upstream"),
            ("integration_support", "Integration"),
            ("local_runtime_support", "Local Runtime"),
            ("recommended_device", "Recommended Device"),
        ]:
            left_value = _format_support_value(left.support_report.get(key) if left.support_report else None)
            right_value = _format_support_value(right.support_report.get(key) if right.support_report else None)
            support_summary.add_row(label, left_value, right_value)

        return Group(table, support_summary, Panel(error_text, title="Errors", border_style="red"))

    verdict_lines: list[str] = []
    left_elapsed = left.total_latency_seconds or left.elapsed_seconds
    right_elapsed = right.total_latency_seconds or right.elapsed_seconds
    if left_elapsed is not None and right_elapsed is not None:
        faster = left.backend if left_elapsed < right_elapsed else right.backend
        verdict_lines.append(f"Faster overall: {faster}")
    if left.ttft_seconds is not None and right.ttft_seconds is not None:
        faster_ttft = left.backend if left.ttft_seconds < right.ttft_seconds else right.backend
        verdict_lines.append(f"Lower TTFT: {faster_ttft}")
    if left.peak_memory_bytes is not None and right.peak_memory_bytes is not None:
        lower_mem = left.backend if left.peak_memory_bytes < right.peak_memory_bytes else right.backend
        verdict_lines.append(f"Lower peak memory: {lower_mem}")

    verdict = Panel(Text("\n".join(verdict_lines) or "No clean verdict available."), title="Summary Verdict", border_style="green")
    return Group(table, verdict)
