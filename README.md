# ai-bench

A collection of inference-benchmarking proofs-of-concept across models, serving
engines, and hardware. Each POC lives in its own subdirectory with a self-contained
harness, raw results, and a writeup.

## POCs

| POC | Model | Engine | Hardware | What it measures |
|---|---|---|---|---|
| [`qwen3.6-35b-a3b-awq/`](qwen3.6-35b-a3b-awq/) | Qwen3.6-35B-A3B-AWQ (MoE) | vLLM 0.22 | 1× NVIDIA L40S | Single-stream latency + throughput scaling |

## AWQ lineup sweep

[`run_all_models.sh`](run_all_models.sh) benchmarks a lineup of INT4 AWQ instruct
models on a single L40S, one at a time (download → serve → sweep → free GPU). Each
model is swept across 3 shapes (256/256, 1024/512, 4096/512 in/out tokens) × 4
concurrency levels (1, 8, 32, 64) = 12 runs, saved under `<model>/results/`.
[`compare_models.py`](compare_models.py) aggregates across models.

| Model | Repo | Result |
|---|---|---|
| [`phi-4-awq/`](phi-4-awq/) | `stelterlab/phi-4-AWQ` | ✅ swept |
| [`qwen3.6-27b-awq/`](qwen3.6-27b-awq/) | `QuantTrio/Qwen3.6-27B-AWQ` | ⚠️ retry — see [`retry_qwen3.6-27b.sh`](retry_qwen3.6-27b.sh) |
| [`qwen3-32b-awq/`](qwen3-32b-awq/) | `Qwen/Qwen3-32B-AWQ` | ✅ swept |
| [`gemma-4-26b-a4b-awq/`](gemma-4-26b-a4b-awq/) | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | ✅ swept |

> `qwen3.6-27b-awq` OOM'd at serve time during the lineup run: weights are only
> ~19.7 GiB, but CUDA-graph capture (batch sizes up to 512) pushed usage to 43.4 GiB
> before KV allocation. `retry_qwen3.6-27b.sh` re-runs it with `--max-num-seqs 64`
> (caps graph capture to the batch sizes we actually sweep) plus
> `expandable_segments`.

## Conventions

- Each POC folder is runnable on its own (`run_*.sh` + `aggregate.py`), with paths
  resolved relative to the script — no hardcoded absolute paths.
- Structured results (`*.json`) are committed; verbose logs (`*.stdout`,
  `server.log`) and warm-up/smoke runs are git-ignored.
