# Windows + CUDA Validation Guide

This guide is for running one full validation round on a separate Windows + NVIDIA CUDA machine, such as a laptop with an RTX 3050.

It does not assume that any experimental backend is already working on that machine. Treat the runtime smoke test result on the target machine as the source of truth.

## What this guide covers

- Windows environment setup
- CUDA-enabled PyTorch installation
- repository dependency installation
- baseline generation and benchmark
- FlexAttention generation and benchmark
- side-by-side compare demo

## 1. Prerequisites

- Windows 10 or later
- Python 3.10 or later
- NVIDIA driver installed and working
- PowerShell

PyTorch's official Windows install page says:

- Windows is supported
- Python 3.10 or later is required for the latest stable release
- CUDA-enabled builds should be installed from the official selector for your target CUDA version

Use the official PyTorch selector first:

- [PyTorch Start Locally](https://pytorch.org/get-started/locally/)
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions)

## 2. Create a virtual environment

From PowerShell in the repo root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If your Windows machine uses a different Python launcher version, replace `3.11` with the installed one.

## 3. Install CUDA-enabled PyTorch

Choose the exact command from the official PyTorch selector for your machine.

If you want a concrete example that matches the repo's local `torch 2.8.0` line, the official previous-versions page lists Windows wheels such as:

```powershell
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

Other CUDA wheel variants shown on the official page may be more appropriate for your machine. Pick the one that matches your Windows CUDA environment.

Then install the repo dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 4. Check the machine before running experiments

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count())"
```

If `torch.cuda.is_available()` is `False`, do not trust any CUDA benchmark result yet. Fix the environment first.

## 5. Baseline generation

Run the stable baseline first:

```powershell
python scripts/generate.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend vanilla --device cuda --dtype auto --max-new-tokens 64
```

Then test `sdpa`:

```powershell
python scripts/generate.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend sdpa --device cuda --dtype auto --max-new-tokens 64
```

## 6. Baseline benchmark round

```powershell
python benchmarks/benchmark_decode.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend vanilla --device cuda --dtype auto --max-new-tokens 64 --repeat 5 --warmup 1 --output outputs/benchmarks/benchmark_vanilla_cuda.json
```

```powershell
python benchmarks/benchmark_decode.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend sdpa --device cuda --dtype auto --max-new-tokens 64 --repeat 5 --warmup 1 --output outputs/benchmarks/benchmark_sdpa_cuda.json
```

## 7. FlexAttention validation

### `flex_attention`

```powershell
python scripts/generate.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend flex_attention --device cuda --dtype auto --max-new-tokens 64
```

```powershell
python benchmarks/benchmark_decode.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend flex_attention --device cuda --dtype auto --max-new-tokens 64 --repeat 5 --warmup 1 --output outputs/benchmarks/benchmark_flex_attention_cuda.json
```

### `flex_attention_window_sink`

```powershell
python scripts/generate.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend flex_attention_window_sink --flex-window-size 256 --flex-sink-tokens 4 --device cuda --dtype auto --max-new-tokens 64
```

```powershell
python benchmarks/benchmark_decode.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend flex_attention_window_sink --flex-window-size 256 --flex-sink-tokens 4 --device cuda --dtype auto --max-new-tokens 64 --repeat 5 --warmup 1 --output outputs/benchmarks/benchmark_flex_window_sink_cuda.json
```

## 8. Side-by-side compare demo

Baseline vs `sdpa`:

```powershell
python scripts/compare_demo.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --left-backend vanilla --right-backend sdpa --device cuda --dtype auto --max-new-tokens 128
```

Baseline vs `flex_attention`:

```powershell
python scripts/compare_demo.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --left-backend vanilla --right-backend flex_attention --device cuda --dtype auto --max-new-tokens 128
```

`flex_attention` vs `flex_attention_window_sink`:

```powershell
python scripts/compare_demo.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --left-backend flex_attention --right-backend flex_attention_window_sink --flex-window-size 256 --flex-sink-tokens 4 --device cuda --dtype auto --max-new-tokens 128
```

## 9. What to inspect

For single generation:

- terminal error output
- `backend_support_report`
- `flex_window_size`
- `flex_sink_tokens`

For benchmark JSON:

- `metadata.requested_backend`
- `metadata.backend`
- `metadata.backend_support_report.upstream_support`
- `metadata.backend_support_report.integration_support`
- `metadata.backend_support_report.local_runtime_support`
- `metadata.backend_support_report.failure_reason`
- `metadata.flex_window_size`
- `metadata.flex_sink_tokens`
- `summary.ttft_seconds_mean`
- `summary.total_latency_seconds_mean`
- `summary.tpot_seconds_mean`
- `summary.peak_memory_bytes_max`

On Windows:

- CPU peak memory uses the Win32 `GetProcessMemoryInfo` path.
- CUDA peak memory still uses `torch.cuda.max_memory_allocated`.

For compare demo:

- pane-level support report
- whether the backend reaches `running` or stops at `error`
- final summary panel

## 10. Recommended order for one full round

1. Verify PyTorch + CUDA availability.
2. Run `vanilla` single generation.
3. Run `sdpa` single generation.
4. Run `vanilla` benchmark.
5. Run `sdpa` benchmark.
6. Run `flex_attention` single generation.
7. Run `flex_attention_window_sink` single generation.
8. Run `flex_attention` benchmark.
9. Run `flex_attention_window_sink` benchmark.
10. Run one or two `compare_demo.py` visual demos.

If a FlexAttention-based backend fails, use the runtime smoke test result and failure reason from that machine as the authoritative result.

## 11. Optional one-command PowerShell roundtrip

The repo also includes a helper script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_cuda_roundtrip.ps1
```

If you also want it to launch one side-by-side compare demo at the end:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_cuda_roundtrip.ps1 -RunCompareDemo
```

Useful overrides:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_cuda_roundtrip.ps1 -ModelName "EleutherAI/pythia-70m" -Prompt "Hello from Pythia." -Device cuda -DType auto -MaxNewTokens 64 -Repeat 5 -Warmup 1 -FlexWindowSize 256 -FlexSinkTokens 4
```
