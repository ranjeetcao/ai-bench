# ai-bench

A collection of inference-benchmarking proofs-of-concept across models, serving
engines, and hardware. Each POC lives in its own subdirectory with a self-contained
harness, raw results, and a writeup.

## POCs

| POC | Model | Engine | Hardware | What it measures |
|---|---|---|---|---|
| [`qwen3.6-35b-a3b-awq/`](qwen3.6-35b-a3b-awq/) | Qwen3.6-35B-A3B-AWQ (MoE) | vLLM 0.22 | 1× NVIDIA L40S | Single-stream latency + throughput scaling |

## Conventions

- Each POC folder is runnable on its own (`run_*.sh` + `aggregate.py`), with paths
  resolved relative to the script — no hardcoded absolute paths.
- Structured results (`*.json`) are committed; verbose logs (`*.stdout`,
  `server.log`) and warm-up/smoke runs are git-ignored.
