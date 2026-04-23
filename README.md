# flashdecoding

Minimal decoding scaffold for studying long-context generation on `EleutherAI/pythia-70m-deduped`.

Chinese version: [README_zh.md](./README_zh.md)
Windows + CUDA guide: [docs/windows_cuda_validation.md](./docs/windows_cuda_validation.md)
Colab guide: [docs/colab_validation.md](./docs/colab_validation.md)

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

- `--backend {vanilla,sdpa,flex_attention,flex_attention_window_sink,flash_decode}`
- `--device {auto,cpu,cuda}`
- `--dtype {auto,float32,float16,bfloat16}`
- `--flex-window-size`
- `--flex-sink-tokens`
- `--seed`

Current default for `flex_attention_window_sink`:

- `flex_window_size = 128`
- `flex_sink_tokens = 4`

`dtype=auto` currently uses a small stability heuristic:

- `vanilla` on CUDA resolves to `float32`
- `sdpa` and the FlexAttention-style backends on CUDA resolve to `float16`
- CPU resolves to `float32`

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
- backend support information and failure reasons when an experimental backend is unavailable
- by default the file is written to `outputs/benchmarks/benchmark_<backend>_<timestamp>.json`
- the `outputs/` directory is ignored by Git to keep the repository clean

## Backend status

| Backend | Status | Meaning |
| --- | --- | --- |
| `vanilla` | Stable | Uses Hugging Face eager attention for the baseline. |
| `sdpa` | Implemented | Uses Hugging Face `attn_implementation="sdpa"`. |
| `flex_attention` | Experimental | Uses Hugging Face `attn_implementation="flex_attention"`. CUDA is recommended for performance experiments, but the project now uses a local runtime smoke test instead of hard-coding CUDA-only support. This is not equivalent to true Flash-Decoding. |
| `flex_attention_window_sink` | Experimental optimization | Uses Hugging Face `flex_attention` plus a recent-window and sink-token mask overlay. This is a FlexAttention/FlexDecoding-style decode-friendly optimization experiment, not true Flash-Decoding. Runtime support is validated by a smoke test on the requested device. |
| `flash_decode` | Placeholder | Not implemented. It remains a semantic placeholder for a future Flash-Decoding-style backend and should not be described as a real Flash-Decoding implementation. |

Important:

- `flash_decode` does not silently fall back. The CLI reports that it is still a placeholder and shows a concrete capability snapshot.
- `flex_attention` is exposed as a separate experimental backend and is intentionally not wrapped as `flash_decode`.
- `flex_attention_window_sink` is the first optimization path in this repo's new FlexAttention/FlexDecoding-style research direction.
- For `flex_attention`-based backends, support is reported in three layers: `upstream_support`, `integration_support`, and `local_runtime_support`.

The support layers mean:

- `upstream_support`: whether the current PyTorch build exposes the needed `flex_attention` interfaces.
- `integration_support`: whether the current `transformers` and repo loading path can theoretically request `attn_implementation="flex_attention"`.
- `local_runtime_support`: whether a tiny runtime smoke test on the requested device actually succeeds.

## Comparison benchmark examples

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt "Hello from Pythia." \
  --backend vanilla \
  --local-files-only \
  --repeat 5 \
  --warmup 1
```

For `flex_attention_window_sink`, you can also sweep sparse settings with:

```bash
python3 benchmarks/sweep_window_sink.py \
  --prompt "Hello from Pythia." \
  --model-name EleutherAI/pythia-70m \
  --device cuda \
  --dtype auto \
  --max-new-tokens 320 \
  --window-sizes 64 96 128 160 256 \
  --block-sizes 32 64 128 \
  --repeat 3 \
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

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt "Hello from Pythia." \
  --backend flex_attention \
  --local-files-only \
  --repeat 5 \
  --warmup 1
```

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt "Hello from Pythia." \
  --backend flex_attention_window_sink \
  --flex-window-size 256 \
  --flex-sink-tokens 4 \
  --local-files-only \
  --repeat 5 \
  --warmup 1
```

## Windows + CUDA self-validation

This repository does not claim that `flex_attention` or `flex_attention_window_sink` is already verified on your target CUDA machine. The expected workflow is:

1. Move the repo to your Windows + RTX 3050 + CUDA environment.
2. Install matching `torch`, `transformers`, and any CUDA dependencies there.
3. Run the provided commands on that machine.
4. Use the runtime smoke-test result to decide whether the backend is locally runnable.

Suggested validation order:

```bash
python scripts/generate.py ^
  --prompt "Hello from Pythia." ^
  --model-name EleutherAI/pythia-70m ^
  --backend flex_attention ^
  --device cuda ^
  --dtype auto
```

```bash
python scripts/generate.py ^
  --prompt "Hello from Pythia." ^
  --model-name EleutherAI/pythia-70m ^
  --backend flex_attention_window_sink ^
  --flex-window-size 256 ^
  --flex-sink-tokens 4 ^
  --device cuda ^
  --dtype auto
```

```bash
python benchmarks/benchmark_decode.py ^
  --prompt "Hello from Pythia." ^
  --model-name EleutherAI/pythia-70m ^
  --backend flex_attention ^
  --device cuda ^
  --dtype auto ^
  --repeat 5 ^
  --warmup 1 ^
  --output outputs/benchmarks/benchmark_flex_attention_cuda.json
```

```bash
python benchmarks/benchmark_decode.py ^
  --prompt "Hello from Pythia." ^
  --model-name EleutherAI/pythia-70m ^
  --backend flex_attention_window_sink ^
  --flex-window-size 256 ^
  --flex-sink-tokens 4 ^
  --device cuda ^
  --dtype auto ^
  --repeat 5 ^
  --warmup 1 ^
  --output outputs/benchmarks/benchmark_flex_window_sink_cuda.json
```

What to inspect on that machine:

- `backend_support_report.upstream_support`
- `backend_support_report.integration_support`
- `backend_support_report.local_runtime_support`
- `backend_support_report.failure_reason`
- `flex_window_size` and `flex_sink_tokens` in the saved JSON metadata

If `local_runtime_support=false`, treat that runtime result as the authoritative answer for that machine. Do not infer local availability from README text alone.

If you want one Windows-oriented checklist that covers environment setup, baseline, benchmark, and compare, use:

- [docs/windows_cuda_validation.md](./docs/windows_cuda_validation.md)
- [scripts/windows_cuda_roundtrip.ps1](./scripts/windows_cuda_roundtrip.ps1)

## Live terminal compare

For demos, the recommended entry point is a Rich-based side-by-side terminal view that streams two backends on the same prompt:

```bash
python3 scripts/compare_demo.py \
  --prompt "Hello from Pythia." \
  --model-name EleutherAI/pythia-70m-deduped \
  --left-backend vanilla \
  --right-backend sdpa \
  --local-files-only \
  --max-new-tokens 320
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
- support-layer status for FlexAttention-style backends
- a final summary table after both sides finish

Important: this live compare mode is for visual demonstration, not rigorous measurement. Both backends run concurrently and may contend for CPU/GPU resources. Use `benchmarks/benchmark_decode.py` for cleaner timing comparisons.

## FlexAttention optimization experiment

The current experimental optimization path is:

- `flex_attention_window_sink`

It keeps `flex_attention` as the underlying backend, then overlays a decode-friendly mask that:

- always keeps the first `sink_tokens` visible
- only keeps a recent attention window of size `window_size`

This is intended as a FlexAttention/FlexDecoding-style long-context optimization experiment. It is not a true Flash-Decoding kernel and should not be described that way.

## Notes and limitations

- This repo is inference-only. No training code is included.
- Metrics are measured for batch size 1.
- The decoding loop and metric collection are shared across backends so the benchmark comparison stays consistent.
- Peak memory reports CUDA peak allocation on GPU and process peak RSS on CPU.
- On Windows CPU paths, peak process memory is collected through the Win32 `GetProcessMemoryInfo` API instead of the Unix-only `resource` module.
- `TTFT` is measured as prompt prefill plus the first generated token.
- `TPOT` is measured over generated tokens after the first token. It is `null` if fewer than two tokens are generated.
