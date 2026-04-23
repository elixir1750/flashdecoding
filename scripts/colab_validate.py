#!/usr/bin/env python3
"""Colab-friendly validation entry point for backend experiments."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashdecoding.generation import generate_once
from flashdecoding.model_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Colab validation."""

    parser = argparse.ArgumentParser(description="Colab-friendly validation runner for flashdecoding backends.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Inline prompt text.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a text file containing the prompt.")

    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["vanilla", "sdpa", "flex_attention", "flex_attention_window_sink"],
        choices=["vanilla", "sdpa", "flex_attention", "flex_attention_window_sink", "flash_decode"],
        help="Backends to validate in sequence.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--flex-window-size", type=int, default=256)
    parser.add_argument("--flex-sink-tokens", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-add-special-tokens",
        action="store_true",
        help="Disable tokenizer special tokens for debugging prompt tokenization behavior.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Default: outputs/colab/colab_validation_<timestamp>.json",
    )
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    """Load prompt text from CLI arguments."""

    if args.prompt is not None:
        return args.prompt
    return args.prompt_file.read_text(encoding="utf-8")


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Colab validation runs."""

    if not runs:
        return {
            "runs": 0,
            "ttft_seconds_mean": None,
            "total_latency_seconds_mean": None,
            "tpot_seconds_mean": None,
            "peak_memory_bytes_max": None,
            "first_generated_token_is_eos_any": None,
        }

    ttft_values = [run["ttft_seconds"] for run in runs]
    total_values = [run["total_latency_seconds"] for run in runs]
    tpot_values = [run["tpot_seconds"] for run in runs if run["tpot_seconds"] is not None]
    peak_values = [run["peak_memory_bytes"] for run in runs]

    return {
        "runs": len(runs),
        "ttft_seconds_mean": sum(ttft_values) / len(ttft_values),
        "total_latency_seconds_mean": sum(total_values) / len(total_values),
        "tpot_seconds_mean": (sum(tpot_values) / len(tpot_values)) if tpot_values else None,
        "peak_memory_bytes_max": max(peak_values),
        "first_generated_token_is_eos_any": any(run["first_generated_token_is_eos"] for run in runs),
    }


def build_environment_report() -> dict[str, Any]:
    """Build a machine/environment report for Colab validation output."""

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None,
    }


def run_backend_validation(args: argparse.Namespace, prompt: str, backend_name: str) -> dict[str, Any]:
    """Run generation and repeated validation for one backend."""

    metadata = {
        "requested_backend": backend_name,
        "model_name": args.model_name,
        "device_request": args.device,
        "dtype_request": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "add_special_tokens": not args.no_add_special_tokens,
        "flex_window_size": args.flex_window_size,
        "flex_sink_tokens": args.flex_sink_tokens,
    }

    print(f"[colab] loading backend={backend_name} model={args.model_name}", flush=True)
    try:
        model, tokenizer, device, dtype, backend = load_model_and_tokenizer(
            model_name=args.model_name,
            backend_name=backend_name,
            requested_device=args.device,
            requested_dtype=args.dtype,
            local_files_only=False,
            flex_window_size=args.flex_window_size,
            flex_sink_tokens=args.flex_sink_tokens,
        )
    except Exception as exc:
        return {
            "metadata": metadata,
            "status": "unavailable",
            "error": {
                "message": str(exc),
                "support_report": getattr(exc, "support_report", None).to_dict() if getattr(exc, "support_report", None) is not None else None,
            },
            "warmup_runs": 0,
            "summary": summarize_runs([]),
            "runs": [],
        }

    for _ in range(args.warmup):
        generate_once(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            add_special_tokens=not args.no_add_special_tokens,
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
            add_special_tokens=not args.no_add_special_tokens,
        )
        result["run_index"] = run_index
        runs.append(result)

    resolved_metadata = {
        **metadata,
        "backend": backend.name,
        "backend_notes": backend.notes,
        "backend_support_report": backend.support_report.to_dict() if backend.support_report is not None else None,
        "device": str(device),
        "dtype": str(dtype),
    }

    return {
        "metadata": resolved_metadata,
        "status": "ok",
        "error": None,
        "warmup_runs": args.warmup,
        "summary": summarize_runs(runs),
        "runs": runs,
    }


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    prompt = load_prompt(args)
    if args.output is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        args.output = ROOT / "outputs" / "colab" / f"colab_validation_{timestamp}.json"

    payload = {
        "environment": build_environment_report(),
        "prompt_preview": prompt[:200],
        "results": [],
    }

    for backend_name in args.backends:
        payload["results"].append(run_backend_validation(args=args, prompt=prompt, backend_name=backend_name))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Colab Validation Summary ===")
    print(json.dumps(payload["environment"], indent=2, ensure_ascii=False))
    for result in payload["results"]:
        summary = {
            "backend": result["metadata"]["requested_backend"],
            "status": result["status"],
            "summary": result["summary"],
            "error": result["error"],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nSaved Colab validation output to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
