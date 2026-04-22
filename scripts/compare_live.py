#!/usr/bin/env python3
"""Render a live side-by-side terminal comparison across decoding backends."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashdecoding.generation import generate_stream
from flashdecoding.metrics import get_peak_memory_bytes
from flashdecoding.model_loader import load_model_and_tokenizer


@dataclass
class PanelState:
    """Terminal render state for one backend panel."""

    backend: str
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the live compare demo."""

    parser = argparse.ArgumentParser(description="Live terminal comparison across decoding backends.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Inline prompt text.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a text file containing the prompt.")

    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["vanilla", "sdpa"],
        choices=["vanilla", "sdpa", "flash_decode"],
        help="Choose 2 or 3 backends to compare side by side.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-repeat", type=int, default=1, help="Repeat the prompt this many times to lengthen the demo.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Only load local Hugging Face cache files.")
    parser.add_argument("--refresh-interval", type=float, default=0.05, help="Terminal refresh interval in seconds.")
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    """Load prompt text from CLI arguments."""

    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = args.prompt_file.read_text(encoding="utf-8")

    if args.prompt_repeat < 1:
        raise ValueError("--prompt-repeat must be >= 1.")
    if args.prompt_repeat == 1:
        return prompt
    return "\n\n".join(prompt for _ in range(args.prompt_repeat))


def validate_backends(backends: list[str]) -> list[str]:
    """Ensure the demo uses a clean list of 2 or 3 unique backends."""

    unique = list(dict.fromkeys(backends))
    if len(unique) < 2 or len(unique) > 3:
        raise ValueError("compare_live.py requires 2 or 3 unique backends.")
    return unique


def worker_main(
    backend_name: str,
    model_name: str,
    requested_device: str,
    requested_dtype: str,
    local_files_only: bool,
    prompt: str,
    max_new_tokens: int,
    seed: int | None,
    event_queue: mp.Queue,
) -> None:
    """Load one backend and stream token progress back to the terminal process."""

    try:
        event_queue.put({"backend": backend_name, "event": "status", "status": "loading"})
        model, tokenizer, device, dtype, backend = load_model_and_tokenizer(
            model_name=model_name,
            backend_name=backend_name,
            requested_device=requested_device,
            requested_dtype=requested_dtype,
            local_files_only=local_files_only,
        )
        event_queue.put(
            {
                "backend": backend_name,
                "event": "ready",
                "status": "running",
                "device": str(device),
                "dtype": str(dtype),
                "notes": backend.notes,
            }
        )

        last_event: dict[str, object] | None = None
        for token_event in generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            seed=seed,
        ):
            last_event = token_event
            token_event = dict(token_event)
            token_event["backend"] = backend_name
            event_queue.put(token_event)

        if last_event is None:
            raise RuntimeError("No tokens were generated.")

        event_queue.put(
            {
                "backend": backend_name,
                "event": "done",
                "status": "done",
                "generated_tokens": int(last_event["generated_tokens"]),
                "generated_text": str(last_event["generated_text"]),
                "ttft_seconds": float(last_event["ttft_seconds"]),
                "total_latency_seconds": float(last_event["elapsed_seconds"]),
                "peak_memory_bytes": int(get_peak_memory_bytes(device)[0]),
            }
        )
    except Exception as exc:
        event_queue.put({"backend": backend_name, "event": "error", "status": "error", "error": str(exc)})


def format_seconds(value: float | None) -> str:
    """Format a small timing value for terminal display."""

    if value is None:
        return "-"
    return f"{value:.3f}s"


def format_memory(value: int | None) -> str:
    """Format memory usage in MiB for terminal display."""

    if value is None:
        return "-"
    return f"{value / (1024 ** 2):.1f} MiB"


def render_panel(panel: PanelState, width: int, height: int) -> list[str]:
    """Render one backend panel into terminal-width constrained lines."""

    content = [
        f"[{panel.backend}] {panel.status}",
        f"device: {panel.device or '-'}",
        f"dtype: {panel.dtype or '-'}",
        f"prompt_tokens: {panel.prompt_tokens if panel.prompt_tokens is not None else '-'}",
        f"generated_tokens: {panel.generated_tokens}",
        f"ttft: {format_seconds(panel.ttft_seconds)}",
        f"elapsed: {format_seconds(panel.elapsed_seconds)}",
        f"total: {format_seconds(panel.total_latency_seconds)}",
        f"peak_memory: {format_memory(panel.peak_memory_bytes)}",
    ]

    if panel.notes:
        content.extend(textwrap.wrap(f"notes: {panel.notes}", width=max(10, width - 2)))
    if panel.error:
        content.extend(textwrap.wrap(f"error: {panel.error}", width=max(10, width - 2)))

    content.append("")
    wrapped_text = textwrap.wrap(panel.text or "", width=max(10, width - 2), replace_whitespace=False, drop_whitespace=False)
    if not wrapped_text:
        wrapped_text = ["(no text yet)"]
    content.extend(wrapped_text)

    lines = [line[:width].ljust(width) for line in content[:height]]
    while len(lines) < height:
        lines.append(" " * width)
    return lines


def render_screen(prompt: str, panels: list[PanelState]) -> str:
    """Render the full terminal screen with prompt and side-by-side panels."""

    term_size = shutil.get_terminal_size((120, 36))
    column_gap = 3
    panel_width = max(28, (term_size.columns - column_gap * (len(panels) - 1)) // len(panels))
    panel_height = max(16, term_size.lines - 6)

    prompt_header = textwrap.shorten(prompt.replace("\n", " "), width=max(40, term_size.columns - 12), placeholder=" ...")
    header = [
        "Live Backend Compare",
        f"prompt: {prompt_header}",
        "",
    ]

    rendered_panels = [render_panel(panel=panel, width=panel_width, height=panel_height) for panel in panels]
    panel_lines: list[str] = []
    for row in range(panel_height):
        panel_lines.append((" " * column_gap).join(panel[row] for panel in rendered_panels))

    return "\x1b[2J\x1b[H" + "\n".join(header + panel_lines)


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    prompt = load_prompt(args)

    try:
        backends = validate_backends(args.backends)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    mp.set_start_method("spawn", force=True)
    event_queue: mp.Queue = mp.Queue()
    panels = {backend: PanelState(backend=backend) for backend in backends}

    processes = [
        mp.Process(
            target=worker_main,
            args=(
                backend,
                args.model_name,
                args.device,
                args.dtype,
                args.local_files_only,
                prompt,
                args.max_new_tokens,
                None if args.seed is None else args.seed + index,
                event_queue,
            ),
        )
        for index, backend in enumerate(backends)
    ]

    for process in processes:
        process.start()

    try:
        while True:
            saw_event = False
            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break

                saw_event = True
                panel = panels[event["backend"]]
                panel.status = str(event.get("status", panel.status))
                panel.notes = str(event.get("notes", panel.notes))
                panel.device = str(event.get("device", panel.device))
                panel.dtype = str(event.get("dtype", panel.dtype))
                if "prompt_tokens" in event:
                    panel.prompt_tokens = int(event["prompt_tokens"])
                if "generated_tokens" in event:
                    panel.generated_tokens = int(event["generated_tokens"])
                if "ttft_seconds" in event:
                    panel.ttft_seconds = float(event["ttft_seconds"])
                if "elapsed_seconds" in event:
                    panel.elapsed_seconds = float(event["elapsed_seconds"])
                if "total_latency_seconds" in event:
                    panel.total_latency_seconds = float(event["total_latency_seconds"])
                if "peak_memory_bytes" in event:
                    panel.peak_memory_bytes = int(event["peak_memory_bytes"])
                if "generated_text" in event:
                    panel.text = str(event["generated_text"])
                if "error" in event:
                    panel.error = str(event["error"])

            if saw_event:
                print(render_screen(prompt=prompt, panels=[panels[name] for name in backends]), end="", flush=True)

            if all(not process.is_alive() for process in processes):
                break

            time.sleep(args.refresh_interval)

        while True:
            try:
                event = event_queue.get_nowait()
            except queue.Empty:
                break
            panel = panels[event["backend"]]
            panel.status = str(event.get("status", panel.status))
            if "total_latency_seconds" in event:
                panel.total_latency_seconds = float(event["total_latency_seconds"])
            if "generated_text" in event:
                panel.text = str(event["generated_text"])
            if "error" in event:
                panel.error = str(event["error"])

        print(render_screen(prompt=prompt, panels=[panels[name] for name in backends]), end="", flush=True)
        print()
    finally:
        for process in processes:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
