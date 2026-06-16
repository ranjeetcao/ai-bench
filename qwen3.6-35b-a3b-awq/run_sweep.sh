#!/usr/bin/env bash
# Inference-speed sweep for Qwen3.6-35B-A3B-AWQ on vLLM (single NVIDIA L40S).
#
# Runs a matrix of request shapes x concurrency against an already-running vLLM
# OpenAI server, saving one result JSON per run under results/. Emits ONE
# progress line per run on stdout; verbose per-run output goes to results/*.stdout.
#
# Prereqs:
#   - a conda env named `vllm` with vllm installed
#   - a vLLM server already serving $MODEL on $HOST:$PORT, e.g.:
#       vllm serve QuantTrio/Qwen3.6-35B-A3B-AWQ \
#         --max-model-len 8192 --gpu-memory-utilization 0.90
#
# NOTE: do NOT use `set -u` — conda's cuda-nvcc activate hook references an
# unbound NVCC_PREPEND_FLAGS and would abort the script on activation.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="QuantTrio/Qwen3.6-35B-A3B-AWQ"
HOST=127.0.0.1
PORT=8000
RESDIR="$SCRIPT_DIR/results"
mkdir -p "$RESDIR"

run_bench () {
  local in=$1 out=$2 conc=$3 nprompts=$4
  local tag="in${in}_out${out}_c${conc}"
  echo "RUN  $tag  (num_prompts=$nprompts)"
  vllm bench serve \
    --backend vllm \
    --model "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len "$in" --random-output-len "$out" --random-range-ratio 0 \
    --num-prompts "$nprompts" \
    --max-concurrency "$conc" \
    --ignore-eos \
    --seed 12345 \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "50,90,99" \
    --save-result --result-dir "$RESDIR" --result-filename "${tag}.json" \
    > "$RESDIR/${tag}.stdout" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "OK   $tag"; else echo "FAIL $tag (rc=$rc) see $RESDIR/${tag}.stdout"; fi
}

echo "WARMUP starting"
vllm bench serve --backend vllm --model "$MODEL" --host "$HOST" --port "$PORT" \
  --dataset-name random --random-input-len 128 --random-output-len 64 --random-range-ratio 0 \
  --num-prompts 8 --max-concurrency 4 --ignore-eos --seed 1 > "$RESDIR/warmup.stdout" 2>&1
echo "WARMUP done"

# shape: "input_len output_len"  ;  per shape we sweep concurrency / num_prompts
for shape in "256 256" "1024 512" "4096 512"; do
  set -- $shape; IN=$1; OUT=$2
  run_bench "$IN" "$OUT" 1  16
  run_bench "$IN" "$OUT" 8  48
  run_bench "$IN" "$OUT" 32 128
  run_bench "$IN" "$OUT" 64 192
done

echo "SWEEP COMPLETE"
