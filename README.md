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
│   └── generate.py
└── src/
    └── flashdecoding/
        ├── __init__.py
        ├── backends.py
        ├── generation.py
        ├── metrics.py
        └── model_loader.py
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

For demos, there is also a live side-by-side terminal view that streams token progress from 2 or 3 backends at once:

```bash
python3 scripts/compare_live.py \
  --prompt "Hello from Pythia." \
  --backends vanilla sdpa \
  --local-files-only \
  --prompt-repeat 16 \
  --max-new-tokens 32
```

You can also include the placeholder backend to show capability failure explicitly:

```bash
python3 scripts/compare_live.py \
  --prompt "Hello from Pythia." \
  --backends vanilla sdpa flash_decode \
  --local-files-only \
  --prompt-repeat 16 \
  --max-new-tokens 32
```

Important: this live compare mode is for visual demonstration, not rigorous measurement. Multiple backends are loaded at the same time and may contend for CPU/GPU resources. Use `benchmarks/benchmark_decode.py` for cleaner timing comparisons.

## Notes and limitations

- This repo is inference-only. No training code is included.
- Metrics are measured for batch size 1.
- The decoding loop and metric collection are shared across backends so the benchmark comparison stays consistent.
- Peak memory reports CUDA peak allocation on GPU and process peak RSS on CPU.
- `TTFT` is measured as prompt prefill plus the first generated token.
- `TPOT` is measured over generated tokens after the first token. It is `null` if fewer than two tokens are generated.
