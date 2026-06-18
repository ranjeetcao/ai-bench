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

## Long-context sweep

[`run_longctx.sh`](run_longctx.sh) re-serves phi-4 and Qwen3-32B at each model's
**native** max context (no YaRN, no quality loss) on a single L40S and sweeps
in/out shapes × concurrency (1, 4, 8). `--max-num-seqs 16` caps CUDA-graph capture
just above our max concurrency to maximize the KV pool. Results land in
`<model>/results_longctx/` so the 8K baseline is untouched. Output is 512 tokens
for every shape. TTFT/e2e are medians (means are skewed by queueing at high
concurrency).

**phi-4-awq** — `max_model_len=16384`, KV pool 163,200 tok (~10× concurrency at 16K):

| in tok | conc | TTFT med (ms) | TPOT med (ms) | output tok/s | total tok/s | e2e med (s) |
|---|---|---|---|---|---|---|
| 8192  | 1 | 1,521 | 14.7 | 56.5  | 961   | 9.1  |
| 8192  | 4 | 1,097 | 24.0 | 141.2 | 2,400 | 13.3 |
| 8192  | 8 | 276   | 34.3 | 201.3 | 3,422 | 17.8 |
| 15360 | 1 | 3,195 | 16.8 | 43.4  | 1,346 | 11.8 |
| 15360 | 4 | 1,926 | 36.3 | 92.2  | 2,859 | 19.7 |
| 15360 | 8 | 5,650 | 89.2 | 79.5  | 2,465 | 51.7 |

**qwen3-32b-awq** — `max_model_len=40960`, KV pool 86,384 tok (~2.1× concurrency at 40K):

| in tok | conc | TTFT med (ms) | TPOT med (ms) | output tok/s | total tok/s | e2e med (s) |
|---|---|---|---|---|---|---|
| 8192  | 1 | 3,575   | 29.3  | 27.5 | 468   | 18.6  |
| 8192  | 4 | 2,336   | 40.0  | 75.5 | 1,283 | 23.6  |
| 8192  | 8 | 6,981   | 96.9  | 72.6 | 1,233 | 56.6  |
| 32768 | 1 | 19,780  | 38.2  | 13.1 | 849   | 39.3  |
| 32768 | 4 | 83,994  | 87.1  | 15.9 | 1,032 | 127.7 |
| 32768 | 8 | 215,949 | 88.1  | 15.7 | 1,020 | 261.1 |
| 40448 | 1 | 26,124  | 41.0  | 10.9 | 871   | 47.1  |
| 40448 | 4 | 110,325 | 105.3 | 12.5 | 1,001 | 163.4 |
| 40448 | 8 | 271,689 | 105.2 | 12.6 | 1,006 | 326.1 |

> Qwen3-32B's KV pool only fits ~2.1 full 40K requests, so at 32K/40K input the
> server can't keep all concurrent requests resident — prefills queue and TTFT
> blows up (3.6 min median at 40K/c8) while output throughput stays flat (~13 tok/s).
> phi-4 at 16K has ~10× headroom and scales cleanly with concurrency. A single L40S
> is comfortable for 16K-class long context; Qwen3-32B at its full 40K window wants
> a second GPU or a larger KV budget for any real concurrency.

## Conventions

- Each POC folder is runnable on its own (`run_*.sh` + `aggregate.py`), with paths
  resolved relative to the script — no hardcoded absolute paths.
- Structured results (`*.json`) are committed; verbose logs (`*.stdout`,
  `server.log`) and warm-up/smoke runs are git-ignored.
