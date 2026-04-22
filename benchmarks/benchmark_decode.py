#!/usr/bin/env python3
"""Benchmark single-prompt decoding and save machine-readable metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashdecoding.generation import generate_once
from flashdecoding.model_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the benchmark entry point."""

    parser = argparse.ArgumentParser(description="Benchmark single-prompt decoding for pythia-70m.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Inline prompt text.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a text file containing the prompt.")

    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--backend", type=str, default="vanilla", choices=["vanilla", "sdpa", "flash_decode"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True, help="Write results to a .json or .csv file.")
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    """Load prompt text from CLI arguments."""

    if args.prompt is not None:
        return args.prompt
    return args.prompt_file.read_text(encoding="utf-8")


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate benchmark metrics across repeated runs."""

    ttft_values = [run["ttft_seconds"] for run in runs]
    total_values = [run["total_latency_seconds"] for run in runs]
    peak_values = [run["peak_memory_bytes"] for run in runs]
    tpot_values = [run["tpot_seconds"] for run in runs if run["tpot_seconds"] is not None]

    summary: dict[str, Any] = {
        "runs": len(runs),
        "ttft_seconds_mean": statistics.mean(ttft_values),
        "ttft_seconds_min": min(ttft_values),
        "ttft_seconds_max": max(ttft_values),
        "total_latency_seconds_mean": statistics.mean(total_values),
        "total_latency_seconds_min": min(total_values),
        "total_latency_seconds_max": max(total_values),
        "peak_memory_bytes_max": max(peak_values),
    }

    if tpot_values:
        summary["tpot_seconds_mean"] = statistics.mean(tpot_values)
        summary["tpot_seconds_min"] = min(tpot_values)
        summary["tpot_seconds_max"] = max(tpot_values)
    else:
        summary["tpot_seconds_mean"] = None
        summary["tpot_seconds_min"] = None
        summary["tpot_seconds_max"] = None

    return summary


def write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-run benchmark rows as CSV."""

    fieldnames = [
        "run_index",
        "prompt_tokens",
        "generated_tokens",
        "ttft_seconds",
        "tpot_seconds",
        "total_latency_seconds",
        "peak_memory_bytes",
        "peak_memory_source",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write structured benchmark output as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    prompt = load_prompt(args)

    try:
        model, tokenizer, device, dtype, backend = load_model_and_tokenizer(
            model_name=args.model_name,
            backend_name=args.backend,
            requested_device=args.device,
            requested_dtype=args.dtype,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for _ in range(args.warmup):
        generate_once(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_on_eos=not args.ignore_eos,
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
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_on_eos=not args.ignore_eos,
            seed=None if args.seed is None else args.seed + run_index,
        )
        result["run_index"] = run_index
        runs.append(result)

    summary = summarize_runs(runs)
    metadata = {
        "model_name": args.model_name,
        "backend": backend.name,
        "backend_notes": backend.notes,
        "device": str(device),
        "dtype": str(dtype),
        "prompt_tokens": runs[0]["prompt_tokens"] if runs else None,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "stop_on_eos": not args.ignore_eos,
    }

    if args.output.suffix.lower() == ".csv":
        write_csv(args.output, runs)
    else:
        payload = {"metadata": metadata, "summary": summary, "runs": runs}
        write_json(args.output, payload)

    print("=== Benchmark Summary ===")
    print(json.dumps({"metadata": metadata, "summary": summary}, indent=2, ensure_ascii=False))
    print(f"\nSaved benchmark output to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
