# Colab 验证指南

这个仓库现在带了一个更适合 Colab 的验证入口：

- [scripts/colab_validate.py](../scripts/colab_validate.py)

它适合 notebook / Colab 环境，因为：

- 顺序跑多个 backend 比终端 TUI 更稳
- JSON 输出比 Rich 动态界面更适合保存
- 你可以一次拿到 baseline 和实验 backend 的统一报告

## 推荐的 Colab 流程

## 1. 启用 GPU runtime

在 Colab 里，先把 notebook runtime 切到带 GPU 的环境，再运行下面的命令。

## 2. 克隆仓库并安装依赖

```bash
!git clone https://github.com/elixir1750/flashdecoding.git
%cd flashdecoding
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
```

## 3. 先检查环境

```bash
!python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count()); print('gpu_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## 4. 跑一轮完整 Colab 验证

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

## 5. 如果怀疑 prompt tokenization / EOS 行为

可以先做单 backend 的聚焦排查：

```bash
!python scripts/generate.py \
  --prompt "Hello," \
  --model-name EleutherAI/pythia-70m \
  --backend vanilla \
  --device cuda \
  --dtype auto \
  --max-new-tokens 16
```

再和禁用 special tokens 的情况做对照：

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

## 输出格式

Colab 验证脚本会把 JSON 输出写到：

```text
outputs/colab/colab_validation_<timestamp>.json
```

每个 backend 的结果里都会包含：

- 请求的 backend
- 实际解析后的 backend 元数据
- 环境报告
- support report
- 重复生成的 runs
- 平均时延汇总
- backend 不可用时的 error block

## 说明

- `compare_demo.py` 在 Colab notebook 里可能显示效果不佳，因为它是面向终端的 Rich TUI。
- 在 Colab 里更推荐 `scripts/colab_validate.py`，再配合少量 `scripts/generate.py` 做定位。
- `flash_decode` 仍然只是 placeholder，不应被当成真正 Flash-Decoding 实现。
