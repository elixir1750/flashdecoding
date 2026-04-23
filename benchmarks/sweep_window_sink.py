#!/usr/bin/env python3
"""Sweep flex_attention_window_sink window/block settings and save JSON results."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashdecoding.generation import generate_once
from flashdecoding.model_loader import get_flex_experiment_metadata, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the window-sink sweep."""

    parser = argparse.ArgumentParser(description="Sweep flex_attention_window_sink settings.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Inline prompt text.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a text file containing the prompt.")

    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--flex-sink-tokens", type=int, default=4)
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[64, 96, 128, 160, 256],
        help="Window sizes to sweep.",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Block sizes to sweep.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: outputs/benchmarks/window_sink_sweep_<timestamp>.json",
    )
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    """Load prompt text from CLI arguments."""

    if args.prompt is not None:
        return args.prompt
    return args.prompt_file.read_text(encoding="utf-8")


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated runs for one configuration."""

    if not runs:
        return {
            "runs": 0,
            "ttft_seconds_mean": None,
            "total_latency_seconds_mean": None,
            "tpot_seconds_mean": None,
            "peak_memory_bytes_max": None,
        }

    tpot_values = [run["tpot_seconds"] for run in runs if run["tpot_seconds"] is not None]
    return {
        "runs": len(runs),
        "ttft_seconds_mean": statistics.mean(run["ttft_seconds"] for run in runs),
        "total_latency_seconds_mean": statistics.mean(run["total_latency_seconds"] for run in runs),
        "tpot_seconds_mean": statistics.mean(tpot_values) if tpot_values else None,
        "peak_memory_bytes_max": max(run["peak_memory_bytes"] for run in runs),
    }


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    prompt = load_prompt(args)
    if args.output is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        args.output = ROOT / "outputs" / "benchmarks" / f"window_sink_sweep_{timestamp}.json"

    payload: dict[str, Any] = {
        "metadata": {
            "model_name": args.model_name,
            "device_request": args.device,
            "dtype_request": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "flex_sink_tokens": args.flex_sink_tokens,
            "window_sizes": args.window_sizes,
            "block_sizes": args.block_sizes,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
        },
        "results": [],
    }

    for window_size in args.window_sizes:
        for block_size in args.block_sizes:
            print(
                f"[sweep] backend=flex_attention_window_sink window={window_size} block={block_size}",
                flush=True,
            )
            try:
                model, tokenizer, device, dtype, backend = load_model_and_tokenizer(
                    model_name=args.model_name,
                    backend_name="flex_attention_window_sink",
                    requested_device=args.device,
                    requested_dtype=args.dtype,
                    local_files_only=False,
                    flex_window_size=window_size,
                    flex_sink_tokens=args.flex_sink_tokens,
                    flex_block_size=block_size,
                )
            except Exception as exc:
                payload["results"].append(
                    {
                        "window_size": window_size,
                        "block_size": block_size,
                        "status": "unavailable",
                        "error": {
                            "message": str(exc),
                            "support_report": getattr(exc, "support_report", None).to_dict()
                            if getattr(exc, "support_report", None) is not None
                            else None,
                        },
                    }
                )
                continue

            for _ in range(args.warmup):
                generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )

            runs: list[dict[str, Any]] = []
            for run_index in range(args.repeat):
                result = generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    seed=None if args.seed is None else args.seed + run_index,
                )
                result["run_index"] = run_index
                runs.append(result)

            payload["results"].append(
                {
                    "window_size": window_size,
                    "block_size": block_size,
                    "status": "ok",
                    "backend_notes": backend.notes,
                    "backend_support_report": backend.support_report.to_dict()
                    if backend.support_report is not None
                    else None,
                    "flex_experiment_metadata": get_flex_experiment_metadata(model),
                    "summary": summarize_runs(runs),
                    "runs": runs,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sweep output to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
