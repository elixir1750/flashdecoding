# flashdecoding

Minimal course-project scaffold for studying long-context decoding acceleration on `EleutherAI/pythia-70m`.

Chinese version: [README_zh.md](./README_zh.md)

This repository intentionally starts small:

- load `EleutherAI/pythia-70m` with Hugging Face Transformers
- run single-prompt generation from the CLI
- measure `TTFT`, `TPOT`, `total latency`, and `peak memory`
- switch attention backends through a small dispatch layer
- keep a benchmark entry point ready for later long-context experiments

## Project structure

```text
flashdecoding/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   └── README.md
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
  --max-new-tokens 32 \
  --backend vanilla
```

Useful options:

- `--backend {vanilla,sdpa,flash_decode}`
- `--device {auto,cpu,cuda}`
- `--dtype {auto,float32,float16,bfloat16}`
- `--do-sample`
- `--temperature`
- `--top-k`
- `--top-p`
- `--seed`

## Benchmark

The benchmark script always writes machine-readable output.

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt-file README.md \
  --max-new-tokens 32 \
  --backend vanilla \
  --repeat 3 \
  --warmup 1 \
  --output benchmark_vanilla.json
```

Use a `.json` suffix for structured summary output, or `.csv` for per-run rows.

## Backend status

- `vanilla`: implemented through Hugging Face eager attention
- `sdpa`: implemented through Hugging Face `attn_implementation="sdpa"` when available
- `flash_decode`: interface reserved, capability check implemented, real kernel/backend integration not added yet

Important: `flash_decode` does not silently fall back. The CLI exits with a clear error when it is requested in the current minimal scaffold.

## Notes and limitations

- This repo is inference-only. No training code is included.
- Metrics are measured for batch size 1.
- Peak memory reports CUDA peak allocation on GPU and process peak RSS on CPU.
- `TTFT` is measured as prompt prefill plus the first generated token.
- `TPOT` is measured over generated tokens after the first token. It is `null` if fewer than two tokens are generated.
