# flashdecoding

一个面向课程项目的最小 baseline 脚手架，用来研究 `EleutherAI/pythia-70m` 在长上下文场景下的 decoding。

英文版说明见：[README.md](./README.md)

这个仓库刻意保持“小而清晰”：

- 使用 Hugging Face Transformers 加载 `EleutherAI/pythia-70m`
- 提供稳定的 vanilla 单条 prompt 命令行生成
- 记录 `TTFT`、`TPOT`、`total latency` 和 `peak memory`
- 将 benchmark 结果保存为 JSON
- 在引入实验性 backend 之前，先把 baseline 跑稳

## 项目结构

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
  --max-new-tokens 32
```

常用参数：

- `--device {auto,cpu,cuda}`
- `--dtype {auto,float32,float16,bfloat16}`
- `--seed`

## Benchmark

benchmark 脚本会始终输出 JSON 结果。

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt-file README.md \
  --max-new-tokens 32 \
  --repeat 3 \
  --warmup 1 \
  --output benchmark_vanilla.json
```

- 输出 JSON 中包含运行元数据、汇总统计以及每次 run 的详细测量结果。

## 当前限制

- 当前仓库仅包含推理相关代码，不包含训练代码。
- 指标测量默认基于 batch size 1。
- 当前 baseline 只使用 Hugging Face eager attention，也就是 `vanilla` 路径。
- `peak memory` 在 GPU 上使用 CUDA peak allocation，在 CPU 上使用进程峰值 RSS。
- `TTFT` 定义为 prompt prefill 加上第一个生成 token 的耗时。
- `TPOT` 定义为第一个 token 之后各生成 token 的平均耗时；如果最终只生成了 1 个 token，则该值为 `null`。
