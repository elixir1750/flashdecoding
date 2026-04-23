# Colab Validation Guide

This repo now includes a Colab-friendly validation runner:

- [scripts/colab_validate.py](../scripts/colab_validate.py)

It is designed for notebook or Colab environments where:

- sequential backend validation is easier than a terminal TUI
- JSON output is more useful than Rich live rendering
- you want one saved report covering baseline and experimental backends

## Recommended Colab flow

## 1. Enable a GPU runtime

In Colab, switch the notebook runtime to a GPU-backed runtime before running the commands below.

## 2. Clone the repo and install dependencies

```bash
!git clone https://github.com/elixir1750/flashdecoding.git
%cd flashdecoding
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
```

## 3. Check the environment

```bash
!python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count()); print('gpu_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## 4. Run one full Colab validation pass

```bash
!python scripts/colab_validate.py \
  --prompt "Hello from Pythia." \
  --model-name EleutherAI/pythia-70m \
  --backends vanilla sdpa flex_attention flex_attention_window_sink \
  --device cuda \
  --dtype auto \
  --max-new-tokens 64 \
  --repeat 3 \
  --warmup 1
```

## 5. Debug prompt tokenization if needed

If you suspect prompt tokenization or EOS behavior, run a focused single-backend check:

```bash
!python scripts/generate.py \
  --prompt "Hello," \
  --model-name EleutherAI/pythia-70m \
  --backend vanilla \
  --device cuda \
  --dtype auto \
  --max-new-tokens 16
```

Or compare against no special tokens:

```bash
!python scripts/generate.py \
  --prompt "Hello," \
  --model-name EleutherAI/pythia-70m \
  --backend vanilla \
  --device cuda \
  --dtype auto \
  --max-new-tokens 16 \
  --no-add-special-tokens
```

## Output format

The Colab runner writes one JSON file under:

```text
outputs/colab/colab_validation_<timestamp>.json
```

Each backend result includes:

- requested backend
- resolved backend metadata
- environment report
- support report when available
- repeated generation runs
- mean latency summary
- error block if the backend is unavailable

## Notes

- `compare_demo.py` may render poorly in Colab notebook cells because it is a Rich terminal UI.
- For Colab validation, prefer `scripts/colab_validate.py` plus targeted `scripts/generate.py` runs.
- `flash_decode` remains a placeholder and should not be treated as a real Flash-Decoding implementation.
