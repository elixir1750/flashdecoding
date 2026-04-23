param(
    [string]$ModelName = "EleutherAI/pythia-70m",
    [string]$Prompt = "Hello from Pythia.",
    [string]$Device = "cuda",
    [string]$DType = "auto",
    [int]$MaxNewTokens = 64,
    [int]$Repeat = 5,
    [int]$Warmup = 1,
    [int]$FlexWindowSize = 128,
    [int]$FlexSinkTokens = 4,
    [int]$FlexBlockSize = 64,
    [switch]$RunCompareDemo
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Title,
        [string[]]$Command
    )

    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host ($Command -join " ")
    & $Command[0] $Command[1..($Command.Length - 1)]
}

Run-Step -Title "Environment Check" -Command @(
    "python", "-c",
    "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count())"
)

Run-Step -Title "Generate - vanilla" -Command @(
    "python", "scripts/generate.py",
    "--prompt", $Prompt,
    "--model-name", $ModelName,
    "--backend", "vanilla",
    "--device", $Device,
    "--dtype", $DType,
    "--max-new-tokens", "$MaxNewTokens"
)

Run-Step -Title "Generate - sdpa" -Command @(
    "python", "scripts/generate.py",
    "--prompt", $Prompt,
    "--model-name", $ModelName,
    "--backend", "sdpa",
    "--device", $Device,
    "--dtype", $DType,
    "--max-new-tokens", "$MaxNewTokens"
)

Run-Step -Title "Benchmark - vanilla" -Command @(
    "python", "benchmarks/benchmark_decode.py",
    "--prompt", $Prompt,
    "--model-name", $ModelName,
    "--backend", "vanilla",
    "--device", $Device,
    "--dtype", $DType,
    "--max-new-tokens", "$MaxNewTokens",
    "--repeat", "$Repeat",
    "--warmup", "$Warmup",
    "--output", "outputs/benchmarks/benchmark_vanilla_cuda.json"
)

Run-Step -Title "Benchmark - sdpa" -Command @(
    "python", "benchmarks/benchmark_decode.py",
    "--prompt", $Prompt,
    "--model-name", $ModelName,
    "--backend", "sdpa",
    "--device", $Device,
    "--dtype", $DType,
    "--max-new-tokens", "$MaxNewTokens",
    "--repeat", "$Repeat",
    "--warmup", "$Warmup",
    "--output", "outputs/benchmarks/benchmark_sdpa_cuda.json"
)

Run-Step -Title "Generate - flex_attention" -Command @(
    "python", "scripts/generate.py",
    "--prompt", $Prompt,
    "--model-name", $ModelName,
    "--backend", "flex_attention",
    "--device", $Device,
    "--dtype", $DType,
    "--max-new-tokens", "$MaxNewTokens"
)

Run-Step -Title "Generate - flex_attention_window_sink" -Command @(
    "python", "scripts/generate.py",
    "--prompt", $Prompt,
    "--model-name", $ModelName,
    "--backend", "flex_attention_window_sink",
    "--flex-window-size", "$FlexWindowSize",
    "--flex-sink-tokens", "$FlexSinkTokens",
    "--flex-block-size", "$FlexBlockSize",
    "--device", $Device,
    "--dtype", $DType,
    "--max-new-tokens", "$MaxNewTokens"
)

Run-Step -Title "Benchmark - flex_attention" -Command @(
    "python", "benchmarks/benchmark_decode.py",
    "--prompt", $Prompt,
    "--model-name", $ModelName,
    "--backend", "flex_attention",
    "--device", $Device,
    "--dtype", $DType,
    "--max-new-tokens", "$MaxNewTokens",
    "--repeat", "$Repeat",
    "--warmup", "$Warmup",
    "--output", "outputs/benchmarks/benchmark_flex_attention_cuda.json"
)

Run-Step -Title "Benchmark - flex_attention_window_sink" -Command @(
    "python", "benchmarks/benchmark_decode.py",
    "--prompt", $Prompt,
    "--model-name", $ModelName,
    "--backend", "flex_attention_window_sink",
    "--flex-window-size", "$FlexWindowSize",
    "--flex-sink-tokens", "$FlexSinkTokens",
    "--flex-block-size", "$FlexBlockSize",
    "--device", $Device,
    "--dtype", $DType,
    "--max-new-tokens", "$MaxNewTokens",
    "--repeat", "$Repeat",
    "--warmup", "$Warmup",
    "--output", "outputs/benchmarks/benchmark_flex_window_sink_cuda.json"
)

if ($RunCompareDemo) {
    Run-Step -Title "Compare Demo - vanilla vs sdpa" -Command @(
        "python", "scripts/compare_demo.py",
        "--prompt", $Prompt,
        "--model-name", $ModelName,
        "--left-backend", "vanilla",
        "--right-backend", "sdpa",
        "--device", $Device,
        "--dtype", $DType,
        "--max-new-tokens", "128"
    )
} else {
    Write-Host ""
    Write-Host "Full round completed. You can now run one compare demo manually, for example:" -ForegroundColor Green
    Write-Host "python scripts/compare_demo.py --prompt `"$Prompt`" --model-name $ModelName --left-backend vanilla --right-backend sdpa --device $Device --dtype $DType --max-new-tokens 128"
    Write-Host "Or rerun this script with -RunCompareDemo."
}
