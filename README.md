# flashdecoding

Minimal decoding scaffold for studying long-context generation on `EleutherAI/pythia-70m-deduped`.

Chinese version: [README_zh.md](./README_zh.md)

This repository intentionally starts small:

- load `EleutherAI/pythia-70m-deduped` with Hugging Face Transformers
- run single-prompt generation from the CLI
- measure `TTFT`, `TPOT`, `total latency`, and `peak memory`
- save benchmark results as JSON
- keep the vanilla baseline stable while adding backend dispatch incrementally

## Project structure

```text
flashdecoding/
├── AGENTS.md
├── README.md
├── README_zh.md
├── requirements.txt
├── benchmarks/
│   └── benchmark_decode.py
├── scripts/
│   ├── compare_demo.py
│   ├── compare_live.py
│   └── generate.py
└── src/
    └── flashdecoding/
        ├── __init__.py
        ├── backends.py
        ├── generation.py
        ├── metrics.py
        ├── model_loader.py
        └── ui.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Single generation

```bash
python3 scripts/generate.py \
  --prompt "Write a short note about long-context decoding." \
  --model-name EleutherAI/pythia-70m-deduped \
  --backend vanilla \
  --max-new-tokens 32
```

Useful options:

- `--backend {vanilla,sdpa,flash_decode}`
- `--device {auto,cpu,cuda}`
- `--dtype {auto,float32,float16,bfloat16}`
- `--seed`

## Benchmark

The benchmark script always writes machine-readable output as JSON.

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt-file README.md \
  --model-name EleutherAI/pythia-70m-deduped \
  --backend vanilla \
  --max-new-tokens 32 \
  --repeat 3 \
  --warmup 1
```

The output JSON contains:

- run metadata
- aggregate summary statistics
- per-run measurements
- by default the file is written to `outputs/benchmarks/benchmark_<backend>_<timestamp>.json`
- the `outputs/` directory is ignored by Git to keep the repository clean

## Backend status

- `vanilla`: available now through Hugging Face eager attention
- `sdpa`: available now through Hugging Face `attn_implementation="sdpa"`
- `flash_decode`: placeholder only for now; capability check is implemented, but no real Flash-Decoding backend is integrated yet

Important: `flash_decode` does not silently fall back. If you request it and the current scaffold cannot support it, the CLI exits with a clear error.

## Comparison benchmark examples

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt "Hello from Pythia." \
  --backend vanilla \
  --local-files-only \
  --repeat 5 \
  --warmup 1
```

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt "Hello from Pythia." \
  --backend sdpa \
  --local-files-only \
  --repeat 5 \
  --warmup 1
```

## Live terminal compare

For demos, the recommended entry point is a Rich-based side-by-side terminal view that streams two backends on the same prompt:

```bash
python3 scripts/compare_demo.py \
  --prompt "Hello from Pythia." \
  --model-name EleutherAI/pythia-70m-deduped \
  --left-backend vanilla \
  --right-backend sdpa \
  --local-files-only \
  --max-new-tokens 32
```

If you want to show failure isolation explicitly, you can point one side at the placeholder backend:

```bash
python3 scripts/compare_demo.py \
  --prompt "Hello from Pythia." \
  --model-name EleutherAI/pythia-70m-deduped \
  --left-backend vanilla \
  --right-backend flash_decode \
  --local-files-only \
  --max-new-tokens 320
```

The demo shows:

- a left pane and a right pane instead of top/bottom stacking
- backend name and model name in each pane
- streaming generated text
- live `elapsed`, `TTFT`, `tokens`, `tok/s`, `TPOT`, and `peak memory`
- a final summary table after both sides finish

Important: this live compare mode is for visual demonstration, not rigorous measurement. Both backends run concurrently and may contend for CPU/GPU resources. Use `benchmarks/benchmark_decode.py` for cleaner timing comparisons.

## Notes and limitations

- This repo is inference-only. No training code is included.
- Metrics are measured for batch size 1.
- The decoding loop and metric collection are shared across backends so the benchmark comparison stays consistent.
- Peak memory reports CUDA peak allocation on GPU and process peak RSS on CPU.
- `TTFT` is measured as prompt prefill plus the first generated token.
- `TPOT` is measured over generated tokens after the first token. It is `null` if fewer than two tokens are generated.
