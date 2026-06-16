# Qwen3.6-35B-A3B-AWQ — vLLM inference speed (NVIDIA L40S)

Proof-of-concept benchmark: **single-stream latency** and **throughput scaling**
for `QuantTrio/Qwen3.6-35B-A3B-AWQ` (Qwen3.5-MoE, ~3B active / 35B total params,
AWQ 4-bit) served by vLLM on a single **NVIDIA L40S (46 GB)**.

## Environment

| | |
|---|---|
| GPU | NVIDIA L40S, 46 GB (AWS `g6e.8xlarge`) |
| Engine | vLLM 0.22.0, `awq_marlin` kernel, torch 2.11 / CUDA 13 |
| Model | `QuantTrio/Qwen3.6-35B-A3B-AWQ` (MoE, ~3B active, AWQ 4-bit, group 128) |
| Server | `vllm serve QuantTrio/Qwen3.6-35B-A3B-AWQ --max-model-len 8192 --gpu-memory-utilization 0.90` |
| KV cache | 541,416 tokens (~66 concurrent full-length 8k requests) |

> **Caveat:** vLLM had **no tuned fused-MoE kernel** for this expert layout
> (E=256, N=512) on the L40S and fell back to a generic config
> ("performance might be sub-optimal"). **All numbers below are a conservative
> floor** — a tuned MoE kernel is the most likely lever to beat them.

## Reproduce

```bash
conda activate vllm
# 1. start the server (separate shell), wait until /health returns 200
vllm serve QuantTrio/Qwen3.6-35B-A3B-AWQ --max-model-len 8192 --gpu-memory-utilization 0.90
# 2. run the sweep + aggregate
bash run_sweep.sh        # writes results/*.json
python3 aggregate.py     # prints the summary tables below
```

## Matrix

- Request shapes (input/output tokens): **256/256, 1024/512, 4096/512**
- Concurrency: **1, 8, 32, 64**
- Output length pinned with `--ignore-eos` so runs are directly comparable.

## Results

### Single-stream latency (concurrency = 1)

| Shape in/out | TTFT p50 | TTFT p99 | TPOT | Decode speed | E2E p50 |
|---|---|---|---|---|---|
| 256 / 256 | 42.8 ms | 47.0 ms | 8.53 ms/tok | **117 tok/s** | 2,219 ms *(256 out)* |
| 1024 / 512 | 89.4 ms | 94.0 ms | 8.56 ms/tok | **117 tok/s** | 4,463 ms |
| 4096 / 512 | 227.7 ms | 230.6 ms | 8.64 ms/tok | **116 tok/s** | 4,642 ms |

Per-stream decode speed is ~flat at **116–117 tok/s** regardless of prompt length
(bound by the ~3B active params + memory bandwidth, not context size). Only TTFT
grows with input — and cheaply (~228 ms to ingest a 4k-token prompt).

### Throughput scaling

**256 / 256 (short chat)**

| Conc | Out tok/s | Total tok/s | req/s | TTFT p50 | Per-stream tok/s |
|---|---|---|---|---|---|
| 1 | 115 | 231 | 0.45 | 43 ms | 117 |
| 8 | 525 | 1,050 | 2.05 | 132 ms | 70 |
| 32 | 1,133 | 2,266 | 4.43 | 314 ms | 37 |
| 64 | **1,533** | 3,066 | 5.99 | 430 ms | 25 |

**1024 / 512 (medium / RAG-ish)**

| Conc | Out tok/s | Total tok/s | req/s | TTFT p50 | Per-stream tok/s |
|---|---|---|---|---|---|
| 1 | 115 | 344 | 0.22 | 89 ms | 117 |
| 8 | 516 | 1,549 | 1.01 | 225 ms | 67 |
| 32 | 1,039 | 3,117 | 2.03 | 461 ms | 34 |
| 64 | **1,352** | 4,057 | 2.64 | 635 ms | 22 |

**4096 / 512 (long context)**

| Conc | Out tok/s | Total tok/s | req/s | TTFT p50 | Per-stream tok/s |
|---|---|---|---|---|---|
| 1 | 110 | 992 | 0.22 | 228 ms | 116 |
| 8 | 434 | 3,906 | 0.85 | 659 ms | 58 |
| 32 | 716 | 6,443 | 1.40 | 739 ms | 24 |
| 64 | 748 | **6,736** | 1.46 | 1,275 ms | 13 |

## Takeaways

- **Decode throughput saturates around c=32–64** (~1,500 out tok/s peak on short
  prompts; lower for longer outputs as each decode step pays more attention cost).
  Past c≈32 you mostly trade latency for little extra throughput.
- **Prefill is highly parallel** — total token throughput reaches ~6,700 tok/s on
  the 4k shape, so this box is strong at ingesting long / RAG prompts.
- **Latency stays usable under load** — even at c=64 a single user still gets
  22–25 tok/s (short/medium shapes) with sub-second TTFT.
- **Sweet spot ≈ c=32** for balancing aggregate throughput and per-user latency.

Raw per-run output (`results/*.stdout`) and warm-up/smoke runs are git-ignored;
the structured `results/*.json` are committed.
