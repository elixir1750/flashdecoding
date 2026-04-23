#!/usr/bin/env python3
"""Rich-based side-by-side streaming demo for two decoding backends."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashdecoding.generation import generate_stream
from flashdecoding.metrics import get_peak_memory_bytes
from flashdecoding.model_loader import load_model_and_tokenizer
from flashdecoding.ui import DemoPaneState, render_demo_view, render_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the side-by-side comparison demo."""

    parser = argparse.ArgumentParser(description="Side-by-side streaming demo for two decoding backends.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Inline prompt text.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a text file containing the prompt.")

    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument(
        "--left-backend",
        type=str,
        default="vanilla",
        choices=["vanilla", "sdpa", "flex_attention", "flex_attention_window_sink", "flash_decode"],
    )
    parser.add_argument(
        "--right-backend",
        type=str,
        default="sdpa",
        choices=["vanilla", "sdpa", "flex_attention", "flex_attention_window_sink", "flash_decode"],
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--flex-window-size", type=int, default=128, help="Recent-window size for flex_attention_window_sink.")
    parser.add_argument("--flex-sink-tokens", type=int, default=4, help="Number of sink/prefix tokens always visible in flex_attention_window_sink.")
    parser.add_argument("--flex-block-size", type=int, default=64, help="Block granularity for BlockMask construction in flex_attention_window_sink.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Only load local Hugging Face cache files.")
    parser.add_argument("--refresh-interval", type=float, default=0.05, help="TUI refresh interval in seconds.")
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    """Load prompt text from CLI arguments."""

    if args.prompt is not None:
        return args.prompt
    return args.prompt_file.read_text(encoding="utf-8")


def worker_main(
    side: str,
    backend_name: str,
    model_name: str,
    requested_device: str,
    requested_dtype: str,
    local_files_only: bool,
    prompt: str,
    max_new_tokens: int,
    flex_window_size: int,
    flex_sink_tokens: int,
    flex_block_size: int,
    seed: int | None,
    event_queue: mp.Queue,
) -> None:
    """Run one backend in a worker process and stream updates to the parent."""

    try:
        event_queue.put({"side": side, "backend": backend_name, "event": "status", "status": "loading"})
        model, tokenizer, device, dtype, backend = load_model_and_tokenizer(
            model_name=model_name,
            backend_name=backend_name,
            requested_device=requested_device,
            requested_dtype=requested_dtype,
            local_files_only=local_files_only,
            flex_window_size=flex_window_size,
            flex_sink_tokens=flex_sink_tokens,
            flex_block_size=flex_block_size,
        )
        event_queue.put(
            {
                "side": side,
                "backend": backend_name,
                "event": "ready",
                "status": "running",
                "device": str(device),
                "dtype": str(dtype),
                "notes": backend.notes,
                "support_report": backend.support_report.to_dict() if backend.support_report is not None else None,
                "flex_window_size": flex_window_size if backend.name == "flex_attention_window_sink" else None,
                "flex_sink_tokens": flex_sink_tokens if backend.name == "flex_attention_window_sink" else None,
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
            payload = dict(token_event)
            payload["side"] = side
            payload["backend"] = backend_name
            payload["status"] = "running"
            event_queue.put(payload)

        if last_event is None:
            raise RuntimeError("No tokens were generated.")

        event_queue.put(
            {
                "side": side,
                "backend": backend_name,
                "event": "done",
                "status": "done",
                "generated_tokens": int(last_event["generated_tokens"]),
                "generated_text": str(last_event["generated_text"]),
                "ttft_seconds": float(last_event["ttft_seconds"]),
                "elapsed_seconds": float(last_event["elapsed_seconds"]),
                "total_latency_seconds": float(last_event["elapsed_seconds"]),
                "peak_memory_bytes": int(get_peak_memory_bytes(device)[0]),
            }
        )
    except Exception as exc:
        support_report = getattr(exc, "support_report", None)
        event_queue.put(
            {
                "side": side,
                "backend": backend_name,
                "event": "error",
                "status": "error",
                "error": str(exc),
                "support_report": support_report.to_dict() if support_report is not None else None,
                "failure_reason": support_report.failure_reason if support_report is not None else str(exc),
            }
        )


def apply_event(state: DemoPaneState, event: dict[str, object]) -> None:
    """Merge one worker event into the UI state."""

    state.status = str(event.get("status", state.status))
    state.notes = str(event.get("notes", state.notes))
    state.device = str(event.get("device", state.device))
    state.dtype = str(event.get("dtype", state.dtype))
    if "prompt_tokens" in event:
        state.prompt_tokens = int(event["prompt_tokens"])
    if "generated_tokens" in event:
        state.generated_tokens = int(event["generated_tokens"])
    if "ttft_seconds" in event:
        state.ttft_seconds = float(event["ttft_seconds"])
    if "elapsed_seconds" in event:
        state.elapsed_seconds = float(event["elapsed_seconds"])
    if "total_latency_seconds" in event:
        state.total_latency_seconds = float(event["total_latency_seconds"])
    if "peak_memory_bytes" in event:
        state.peak_memory_bytes = int(event["peak_memory_bytes"])
    if "generated_text" in event:
        state.text = str(event["generated_text"])
    if "support_report" in event:
        state.support_report = event["support_report"]
    if "flex_window_size" in event and event["flex_window_size"] is not None:
        state.flex_window_size = int(event["flex_window_size"])
    if "flex_sink_tokens" in event and event["flex_sink_tokens"] is not None:
        state.flex_sink_tokens = int(event["flex_sink_tokens"])
    if "failure_reason" in event:
        state.failure_reason = str(event["failure_reason"])
    if "error" in event:
        state.error = str(event["error"])


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    prompt = load_prompt(args)
    console = Console()

    left = DemoPaneState(backend=args.left_backend, model_name=args.model_name)
    right = DemoPaneState(backend=args.right_backend, model_name=args.model_name)
    state_by_side = {
        "left": left,
        "right": right,
    }

    mp.set_start_method("spawn", force=True)
    event_queue: mp.Queue = mp.Queue()
    processes = [
        mp.Process(
            target=worker_main,
            args=(
                "left",
                args.left_backend,
                args.model_name,
                args.device,
                args.dtype,
                args.local_files_only,
                prompt,
                args.max_new_tokens,
                args.flex_window_size,
                args.flex_sink_tokens,
                args.flex_block_size,
                args.seed,
                event_queue,
            ),
        ),
        mp.Process(
            target=worker_main,
            args=(
                "right",
                args.right_backend,
                args.model_name,
                args.device,
                args.dtype,
                args.local_files_only,
                prompt,
                args.max_new_tokens,
                args.flex_window_size,
                args.flex_sink_tokens,
                args.flex_block_size,
                None if args.seed is None else args.seed + 1,
                event_queue,
            ),
        ),
    ]

    for process in processes:
        process.start()

    try:
        with Live(render_demo_view(prompt, left, right), console=console, refresh_per_second=max(1, int(1 / args.refresh_interval)), transient=False) as live:
            while True:
                saw_event = False
                while True:
                    try:
                        event = event_queue.get_nowait()
                    except queue.Empty:
                        break

                    saw_event = True
                    side = str(event["side"])
                    apply_event(state_by_side[side], event)

                if saw_event:
                    live.update(render_demo_view(prompt, left, right))

                if all(not process.is_alive() for process in processes):
                    break

                time.sleep(args.refresh_interval)

            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break
                side = str(event["side"])
                apply_event(state_by_side[side], event)

            live.update(render_demo_view(prompt, left, right))
    finally:
        for process in processes:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()

    console.print()
    console.print(render_summary(left, right))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
