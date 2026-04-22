#!/usr/bin/env python3
"""Run a single generation from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashdecoding.generation import generate_once
from flashdecoding.model_loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-prompt generation."""

    parser = argparse.ArgumentParser(description="Single-prompt generation for pythia-70m backends.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Inline prompt text.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a text file containing the prompt.")

    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--backend", type=str, default="vanilla", choices=["vanilla", "sdpa", "flash_decode"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Only load local Hugging Face cache files.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save the result as JSON.")
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    """Load prompt text from CLI arguments."""

    if args.prompt is not None:
        return args.prompt
    return args.prompt_file.read_text(encoding="utf-8")


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    prompt = load_prompt(args)

    print(f"Loading tokenizer and model: {args.model_name} (backend={args.backend})", flush=True)
    try:
        model, tokenizer, device, dtype, backend = load_model_and_tokenizer(
            model_name=args.model_name,
            backend_name=args.backend,
            requested_device=args.device,
            requested_dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded model on {device} with dtype {dtype}. Starting generation...", flush=True)
    result = generate_once(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    result.update(
        {
            "model_name": args.model_name,
            "backend": backend.name,
            "backend_notes": backend.notes,
            "device": str(device),
            "dtype": str(dtype),
        }
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "model_name": result["model_name"],
        "backend": result["backend"],
        "device": result["device"],
        "dtype": result["dtype"],
        "prompt_tokens": result["prompt_tokens"],
        "generated_tokens": result["generated_tokens"],
        "ttft_seconds": result["ttft_seconds"],
        "tpot_seconds": result["tpot_seconds"],
        "total_latency_seconds": result["total_latency_seconds"],
        "peak_memory_bytes": result["peak_memory_bytes"],
        "peak_memory_source": result["peak_memory_source"],
    }

    print("=== Generation Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== Generated Text ===")
    print(result["generated_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
