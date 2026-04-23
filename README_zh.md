# flashdecoding

一个面向课程项目的最小 decoding 脚手架，用来研究 `EleutherAI/pythia-70m-deduped` 在长上下文场景下的生成。

英文版说明见：[README.md](./README.md)
Windows + CUDA 操作说明见：[docs/windows_cuda_validation_zh.md](./docs/windows_cuda_validation_zh.md)
Colab 操作说明见：[docs/colab_validation_zh.md](./docs/colab_validation_zh.md)

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

当前 `flex_attention_window_sink` 的默认参数是：

- `flex_window_size = 128`
- `flex_sink_tokens = 4`
- `flex_block_size = 64`

当前 `dtype=auto` 还有一个很小的稳定性策略：

- CUDA 上的 `vanilla` 会解析成 `float32`
- CUDA 上的 `sdpa` 和 FlexAttention 类 backend 会解析成 `float16`
- CPU 上会解析成 `float32`

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
- 如果实验 backend 不可用，输出 JSON 也会保留 backend support 信息和 failure reason。
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

## 当前实现到哪一步了

当前仓库里的 `flex_attention_window_sink` 已经实现到下面这一步：

- 模型加载仍然走 Hugging Face Transformers 的 `attn_implementation="flex_attention"`
- 实验入口在 [src/flashdecoding/model_loader.py](./src/flashdecoding/model_loader.py)，通过 patch GPT-NeoX 的 causal mask 创建路径接入
- 当前阶段策略是 `prefill_dense_decode_sparse`
- prefill（`query_length > 1`）保持 dense，不强行套稀疏 mask
- decode（`query_length == 1`）才启用 sparse 的 window-sink 路线
- 稀疏路径优先使用直接的 `BlockMask.from_kv_blocks(...)` 构造
- decode 路径当前采用非对称块：`Q_BLOCK_SIZE=1`，`KV_BLOCK_SIZE=flex_block_size`
- recent-window + sink-token 的稀疏规则实现在 [src/flashdecoding/flex_masks.py](./src/flashdecoding/flex_masks.py)
- KV block layout 会跨层复用缓存
- 精确 shape 的 `BlockMask` 也会单独缓存
- 对“整块完全可见”的 KV blocks，会显式标记成 full blocks，减少不必要的 `mask_mod` 开销
- 当前输出 metadata 里会带出：
  - `flex_phase_policy`
  - `flex_mask_representation`
  - `flex_block_mask_path`
  - `flex_block_mask_cache_stats`
  - `flex_block_mask_perf_stats`

当前仍然没有做的事情：

- 没有 true Flash-Decoding kernel
- 没有 paged KV cache
- 没有训练代码，也没有改模型权重
- 不宣称 `flex_attention_window_sink` 已经在所有 GPU 上都比 `sdpa` 更快

三层 support 的含义是：

- `upstream_support`：当前 PyTorch 构建是否暴露了 `flex_attention` 所需接口。
- `integration_support`：当前 `transformers` 加上本仓库的 model loading 路径，是否理论上支持 `attn_implementation="flex_attention"`。
- `local_runtime_support`：在当前请求的 device 上做一次很小的 runtime smoke test 后，是否真的能跑通。

## 对比 Benchmark 示例

```bash
python3 benchmarks/benchmark_decode.py \
  --prompt "Hello from Pythia." \
  --backend vanilla \
  --local-files-only \
  --repeat 5 \
  --warmup 1
```

如果你想专门扫 `flex_attention_window_sink` 的稀疏参数，也可以直接用：

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

## Windows + CUDA 自行验证

当前仓库不会宣称 `flex_attention` 或 `flex_attention_window_sink` 已经在你的目标 CUDA 机器上验证通过。更稳妥的流程是：

1. 把仓库带到你的 Windows + RTX 3050 + CUDA 环境。
2. 在那台机器上安装匹配的 `torch`、`transformers` 和 CUDA 依赖。
3. 在那台机器上运行下面的命令。
4. 以 runtime smoke test 的结果作为该机器上的最终判断依据。

建议的验证顺序：

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

你在那台机器上主要看这些字段：

- `backend_support_report.upstream_support`
- `backend_support_report.integration_support`
- `backend_support_report.local_runtime_support`
- `backend_support_report.failure_reason`
- 保存下来的 JSON metadata 里的 `flex_window_size` 和 `flex_sink_tokens`

如果 `local_runtime_support=false`，就应该把这次 runtime 结果视为那台机器上的真实结论，而不是根据 README 文本做推断。

如果你想要一份覆盖环境安装、baseline、benchmark 和 compare 的 Windows 清单，可以直接看：

- [docs/windows_cuda_validation_zh.md](./docs/windows_cuda_validation_zh.md)
- [scripts/windows_cuda_roundtrip.ps1](./scripts/windows_cuda_roundtrip.ps1)

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
- 对 FlexAttention 类 backend 显示 support report 三层状态
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
- 在 Windows 的 CPU 路径下，进程峰值内存会通过 Win32 `GetProcessMemoryInfo` API 获取，而不是依赖 Unix 才有的 `resource` 模块。
- `TTFT` 定义为 prompt prefill 加上第一个生成 token 的耗时。
- `TPOT` 定义为第一个 token 之后各生成 token 的平均耗时；如果最终只生成了 1 个 token，则该值为 `null`。
