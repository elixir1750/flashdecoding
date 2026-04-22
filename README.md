# flashdecoding

Minimal baseline scaffold for studying long-context decoding on `EleutherAI/pythia-70m`.

Chinese version: [README_zh.md](./README_zh.md)

This repository intentionally starts small:

- load `EleutherAI/pythia-70m` with Hugging Face Transformers
- run stable vanilla single-prompt generation from the CLI
- measure `TTFT`, `TPOT`, `total latency`, and `peak memory`
- save benchmark results as JSON
- keep the baseline easy to run before adding any experimental backend work

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
  --max-new-tokens 32
```

Useful options:

- `--device {auto,cpu,cuda}`
- `--dtype {auto,float32,float16,bfloat16}`
- `--seed`

## Benchmark

The benchmark script always writes machine-readable output as JSON.

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt-file README.md \
  --max-new-tokens 32 \
  --repeat 3 \
  --warmup 1 \
  --output benchmark_vanilla.json
```

The output JSON contains:

- run metadata
- aggregate summary statistics
- per-run measurements

## Notes and limitations

- This repo is inference-only. No training code is included.
- Metrics are measured for batch size 1.
- The current baseline uses Hugging Face eager attention (`vanilla`) only.
- Peak memory reports CUDA peak allocation on GPU and process peak RSS on CPU.
- `TTFT` is measured as prompt prefill plus the first generated token.
- `TPOT` is measured over generated tokens after the first token. It is `null` if fewer than two tokens are generated.
