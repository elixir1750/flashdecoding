# flashdecoding

一个面向课程项目的最小 decoding 脚手架，用来研究 `EleutherAI/pythia-70m-deduped` 在长上下文场景下的生成。

英文版说明见：[README.md](./README.md)

这个仓库刻意保持“小而清晰”：

- 使用 Hugging Face Transformers 加载 `EleutherAI/pythia-70m-deduped`
- 提供单条 prompt 的命令行生成
- 记录 `TTFT`、`TPOT`、`total latency` 和 `peak memory`
- 将 benchmark 结果保存为 JSON
- 在保持 vanilla baseline 稳定的前提下，逐步加入 backend dispatch

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
  --model-name EleutherAI/pythia-70m-deduped \
  --backend vanilla \
  --max-new-tokens 32
```

常用参数：

- `--backend {vanilla,sdpa,flex_attention,flex_attention_window_sink,flash_decode}`
- `--device {auto,cpu,cuda}`
- `--dtype {auto,float32,float16,bfloat16}`
- `--flex-window-size`
- `--flex-sink-tokens`
- `--seed`

## Benchmark

benchmark 脚本会始终输出 JSON 结果。

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt-file README.md \
  --model-name EleutherAI/pythia-70m-deduped \
  --backend vanilla \
  --max-new-tokens 32 \
  --repeat 3 \
  --warmup 1
```

- 输出 JSON 中包含运行元数据、汇总统计以及每次 run 的详细测量结果。
- 默认会写到 `outputs/benchmarks/benchmark_<backend>_<timestamp>.json`。
- `outputs/` 目录已加入 Git 忽略，避免 benchmark 结果把仓库弄乱。

## Backend 状态

| Backend | 当前状态 | 说明 |
| --- | --- | --- |
| `vanilla` | 稳定 | 使用 Hugging Face eager attention，作为 baseline。 |
| `sdpa` | 已实现 | 使用 Hugging Face `attn_implementation="sdpa"`。 |
| `flex_attention` | 实验性 | 使用 Hugging Face `attn_implementation="flex_attention"`。CUDA 仍然是性能实验的推荐设备，但项目现在不再把它硬编码成 CUDA-only，而是通过本地 runtime smoke test 判断当前 device 能否运行。它不等价于真正的 Flash-Decoding。 |
| `flex_attention_window_sink` | 实验性优化路径 | 使用 Hugging Face `flex_attention`，再叠加 recent-window + sink-token 的 mask 优化。这是 FlexAttention/FlexDecoding 风格的 decode-friendly 优化实验，不是真正的 Flash-Decoding。最终是否可运行由当前 device 上的 smoke test 决定。 |
| `flash_decode` | 占位 | 尚未实现。它只是为未来的 Flash-Decoding 风格 backend 预留的语义占位，不应被称为真正 Flash-Decoding。 |

重要说明：

- `flash_decode` 当前不会静默回退；CLI 会明确提示它还是 placeholder，并输出更具体的 capability 检查信息。
- `flex_attention` 是单独暴露的实验 backend，不会被包装成 `flash_decode`。
- `flex_attention_window_sink` 是当前仓库里第一条真正进入实现阶段的 FlexAttention/FlexDecoding 风格优化实验路径。
- 对 `flex_attention` 系后端，项目会分三层报告支持状态：`upstream_support`、`integration_support`、`local_runtime_support`。

## 对比 Benchmark 示例

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

## 终端实时对比

如果你想做展示效果，推荐直接使用现在这版基于 Rich 的左右双栏终端对比：

```bash
python3 scripts/compare_demo.py \
  --prompt "Hello from Pythia." \
  --model-name EleutherAI/pythia-70m-deduped \
  --left-backend vanilla \
  --right-backend sdpa \
  --local-files-only \
  --max-new-tokens 320
```

如果你想把占位中的 `flash_decode` 也展示出来，让它在右侧明确报不支持，也可以这样跑：

```bash
python3 scripts/compare_demo.py \
  --prompt "Hello from Pythia." \
  --model-name EleutherAI/pythia-70m-deduped \
  --left-backend vanilla \
  --right-backend flash_decode \
  --local-files-only \
  --max-new-tokens 320
```

这个 demo 现在会显示：

- 左右两个 pane，而不是上下堆叠
- 每侧的 backend 名称和 model 名称
- 流式生成文本
- 实时的 `elapsed`、`TTFT`、`tokens`、`tok/s`、`TPOT` 和 `peak memory`
- 双侧结束后的最终 summary

注意：这个实时对比模式主要是为了做展示，不等同于严谨 benchmark。因为两个 backend 会并发运行并竞争 CPU/GPU 资源，绝对时间可能会受影响。真正做速度对比时，还是建议使用 `benchmarks/benchmark_decode.py`。

## FlexAttention 优化实验

当前最小可行优化路径是：

- `flex_attention_window_sink`

它保持 `flex_attention` 作为底层 attention backend，然后叠加一个 decode-friendly mask：

- 前 `sink_tokens` 个 token 永远可见
- 其余位置只保留最近 `window_size` 的 attention window

这个路径应该被描述为 FlexAttention/FlexDecoding 风格的长上下文优化实验，不是真正的 Flash-Decoding kernel。

## 当前限制

- 当前仓库仅包含推理相关代码，不包含训练代码。
- 指标测量默认基于 batch size 1。
- 不同 backend 共享同一套 decoding 与计时逻辑，便于做一致的 benchmark 对比。
- `peak memory` 在 GPU 上使用 CUDA peak allocation，在 CPU 上使用进程峰值 RSS。
- `TTFT` 定义为 prompt prefill 加上第一个生成 token 的耗时。
- `TPOT` 定义为第一个 token 之后各生成 token 的平均耗时；如果最终只生成了 1 个 token，则该值为 `null`。
