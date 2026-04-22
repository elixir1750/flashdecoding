# flashdecoding

一个面向课程项目的最小脚手架，用来研究 `EleutherAI/pythia-70m` 在长上下文场景下的 decoding 加速。

英文版说明见：[README.md](./README.md)

这个仓库刻意保持“小而清晰”：

- 使用 Hugging Face Transformers 加载 `EleutherAI/pythia-70m`
- 提供单条 prompt 的命令行生成
- 记录 `TTFT`、`TPOT`、`total latency` 和 `peak memory`
- 通过一个很小的 dispatch 层切换 attention backend
- 为后续长上下文 benchmark 预留独立入口

## 项目结构

```text
flashdecoding/
├── AGENTS.md
├── README.md
├── README_zh.md
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

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 单次生成

```bash
python3 scripts/generate.py \
  --prompt "Write a short note about long-context decoding." \
  --max-new-tokens 32 \
  --backend vanilla
```

常用参数：

- `--backend {vanilla,sdpa,flash_decode}`
- `--device {auto,cpu,cuda}`
- `--dtype {auto,float32,float16,bfloat16}`
- `--do-sample`
- `--temperature`
- `--top-k`
- `--top-p`
- `--seed`

## Benchmark

benchmark 脚本会始终输出机器可读结果。

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt-file README.md \
  --max-new-tokens 32 \
  --backend vanilla \
  --repeat 3 \
  --warmup 1 \
  --output benchmark_vanilla.json
```

- 使用 `.json` 后缀时，输出结构化汇总和每次运行的详细结果。
- 使用 `.csv` 后缀时，输出逐次运行的表格行。

## Backend 状态

- `vanilla`：已实现，使用 Hugging Face eager attention，作为稳定 baseline
- `sdpa`：已实现，使用 Hugging Face `attn_implementation="sdpa"`
- `flash_decode`：目前只预留接口并实现能力探测，尚未接入真实的 Flash-Decoding 内核或自定义 backend

重要说明：`flash_decode` 当前不会静默回退到其他实现；如果请求了它但当前工程并不支持，CLI 会直接报出明确错误。

## 当前限制

- 当前仓库仅包含推理相关代码，不包含训练代码。
- 指标测量默认基于 batch size 1。
- `peak memory` 在 GPU 上使用 CUDA peak allocation，在 CPU 上使用进程峰值 RSS。
- `TTFT` 定义为 prompt prefill 加上第一个生成 token 的耗时。
- `TPOT` 定义为第一个 token 之后各生成 token 的平均耗时；如果最终只生成了 1 个 token，则该值为 `null`。
