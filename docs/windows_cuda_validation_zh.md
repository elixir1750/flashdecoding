# Windows + CUDA 验证指南

这份文档用于在另一台 Windows + NVIDIA CUDA 机器上跑完整一轮验证，例如带 RTX 3050 的笔记本。

它不会预设任何实验 backend 在那台机器上一定可用。最终请以目标机器上的 runtime smoke test 结果为准。

## 这份指南包含什么

- Windows 环境安装
- CUDA 版 PyTorch 安装
- 仓库依赖安装
- baseline 单次生成与 benchmark
- FlexAttention 单次生成与 benchmark
- side-by-side compare demo

## 1. 前置条件

- Windows 10 或更高版本
- Python 3.10 或更高版本
- 已安装并可用的 NVIDIA 驱动
- PowerShell

PyTorch 官方 Windows 安装页目前说明：

- Windows 是支持平台
- 最新 stable 版要求 Python 3.10 或以上
- CUDA 版应从官方 selector 里按目标 CUDA 版本选择安装命令

先看官方页面：

- [PyTorch Start Locally](https://pytorch.org/get-started/locally/)
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions)

## 2. 创建虚拟环境

在仓库根目录打开 PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果你的 Windows 机器安装的是别的 Python 小版本，把 `3.11` 换成实际版本即可。

## 3. 安装 CUDA 版 PyTorch

先到官方 PyTorch selector 里选出适合你机器的命令。

如果你想先用一个和本仓库当前本地 `torch 2.8.0` 路线一致的例子，官方 previous-versions 页面列出了类似这样的 Windows wheel：

```powershell
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

官方页面还列出了其他 CUDA wheel 变体，你应当选择和自己 Windows CUDA 环境匹配的那一个。

然后安装仓库依赖：

```powershell
python -m pip install -r requirements.txt
```

## 4. 跑实验前先检查机器

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count())"
```

如果 `torch.cuda.is_available()` 是 `False`，就不要先相信任何 CUDA benchmark 结果，先把环境修通。

## 5. Baseline 单次生成

先跑稳定 baseline：

```powershell
python scripts/generate.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend vanilla --device cuda --dtype auto --max-new-tokens 64
```

再测 `sdpa`：

```powershell
python scripts/generate.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend sdpa --device cuda --dtype auto --max-new-tokens 64
```

## 6. Baseline benchmark 一轮

```powershell
python benchmarks/benchmark_decode.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend vanilla --device cuda --dtype auto --max-new-tokens 64 --repeat 5 --warmup 1 --output outputs/benchmarks/benchmark_vanilla_cuda.json
```

```powershell
python benchmarks/benchmark_decode.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --backend sdpa --device cuda --dtype auto --max-new-tokens 64 --repeat 5 --warmup 1 --output outputs/benchmarks/benchmark_sdpa_cuda.json
```

## 7. FlexAttention 验证

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

baseline 对 `sdpa`：

```powershell
python scripts/compare_demo.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --left-backend vanilla --right-backend sdpa --device cuda --dtype auto --max-new-tokens 128
```

baseline 对 `flex_attention`：

```powershell
python scripts/compare_demo.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --left-backend vanilla --right-backend flex_attention --device cuda --dtype auto --max-new-tokens 128
```

`flex_attention` 对 `flex_attention_window_sink`：

```powershell
python scripts/compare_demo.py --prompt "Hello from Pythia." --model-name EleutherAI/pythia-70m --left-backend flex_attention --right-backend flex_attention_window_sink --flex-window-size 256 --flex-sink-tokens 4 --device cuda --dtype auto --max-new-tokens 128
```

## 9. 重点看哪些输出

单次生成主要看：

- 终端错误输出
- `backend_support_report`
- `flex_window_size`
- `flex_sink_tokens`

benchmark JSON 主要看：

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

在 Windows 上：

- CPU 峰值内存走 Win32 `GetProcessMemoryInfo` 路径。
- CUDA 峰值内存仍然走 `torch.cuda.max_memory_allocated`。

compare demo 主要看：

- 每个 pane 的 support report
- backend 是进入 `running` 还是停在 `error`
- 最后的 summary panel

## 10. 推荐的一轮顺序

1. 先检查 PyTorch + CUDA 是否可用。
2. 跑 `vanilla` 单次生成。
3. 跑 `sdpa` 单次生成。
4. 跑 `vanilla` benchmark。
5. 跑 `sdpa` benchmark。
6. 跑 `flex_attention` 单次生成。
7. 跑 `flex_attention_window_sink` 单次生成。
8. 跑 `flex_attention` benchmark。
9. 跑 `flex_attention_window_sink` benchmark。
10. 跑 1 到 2 组 `compare_demo.py` 可视化展示。

如果 FlexAttention 类 backend 失败，就把那台机器上的 runtime smoke test 和 failure reason 当成真实结论。

## 11. 可选的一键 PowerShell 脚本

仓库里还带了一份辅助脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_cuda_roundtrip.ps1
```

如果你还想让它在最后顺手打开一组 side-by-side compare demo，可以这样：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_cuda_roundtrip.ps1 -RunCompareDemo
```

常用覆盖参数示例：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_cuda_roundtrip.ps1 -ModelName "EleutherAI/pythia-70m" -Prompt "Hello from Pythia." -Device cuda -DType auto -MaxNewTokens 64 -Repeat 5 -Warmup 1 -FlexWindowSize 256 -FlexSinkTokens 4
```
