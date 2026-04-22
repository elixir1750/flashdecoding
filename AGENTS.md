# AGENTS.md

## Project goal
This repo studies long-context decoding acceleration for EleutherAI/pythia-70m, with a focus on Flash-Decoding-style attention backends.

## Priorities
1. Keep the repo minimal and runnable
2. Preserve a strong vanilla baseline
3. Add backend dispatch incrementally
4. Make benchmarking easy and reproducible

## Rules
- Do not add training code
- Do not change model weights
- Prefer small patches over large refactors
- Expose all experiment parameters through CLI or config
- Do not silently fall back when an experimental backend is unavailable
- Always explain assumptions and limitations
- Benchmark scripts must save machine-readable outputs (json or csv)

## Metrics
- TTFT
- TPOT
- total latency
- peak memory

## Coding style
- Keep functions short
- Add docstrings for new public functions
- Avoid unnecessary abstractions
- Prefer explicit file/function names over cleverness