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

    body = Text(state.text or "(no text yet)", overflow="fold")
    if state.error:
        body = Text(state.error, style="bold red", overflow="fold")

    group = Group(metrics, Text(""), body)
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
        if right.error:
            error_text.append(f"{right.backend}: {right.error}", style="bold red")
        return Group(table, Panel(error_text, title="Errors", border_style="red"))

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
