#!/usr/bin/env bash
# One-off retry for qwen3.6-27b-awq, which OOM'd at KV-cache allocation during the
# full lineup run (weights fit, but weights + KV cache exceeded the 0.90 budget).
# Fix: drop --gpu-memory-utilization to 0.85 and enable expandable_segments to cut
# allocator fragmentation. Same standard sweep as run_all_models.sh.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export HF_HOME=/mnt/models/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ec2-user/ranjeet/bench
HOST=127.0.0.1; PORT=8000
REPO="QuantTrio/Qwen3.6-27B-AWQ"; SUB="qwen3.6-27b-awq"
DIR="$ROOT/$SUB"; RESDIR="$DIR/results"; mkdir -p "$RESDIR"

free_gpu () {
  [ -n "${1:-}" ] && kill -9 "$1" 2>/dev/null
  sleep 3
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null
  done
  for i in $(seq 1 80); do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 3
  done
}

sweep () {
  local RESDIR=$1 MODEL=$2
  run_one () {
    local in=$1 out=$2 conc=$3 np=$4 tag="in${1}_out${2}_c${3}"
    if vllm bench serve --backend vllm --model "$MODEL" --host $HOST --port $PORT --endpoint /v1/completions \
         --dataset-name random --random-input-len "$in" --random-output-len "$out" --random-range-ratio 0 \
         --num-prompts "$np" --max-concurrency "$conc" --ignore-eos --seed 12345 \
         --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,99" \
         --save-result --result-dir "$RESDIR" --result-filename "${tag}.json" > "$RESDIR/${tag}.stdout" 2>&1; then
      echo "OK $tag" >> "$RESDIR/_progress.txt"
    else
      echo "  FAIL_RUN $SUB/$tag"
    fi
  }
  vllm bench serve --backend vllm --model "$MODEL" --host $HOST --port $PORT --dataset-name random \
    --random-input-len 128 --random-output-len 64 --random-range-ratio 0 --num-prompts 8 --max-concurrency 4 \
    --ignore-eos --seed 1 > "$RESDIR/warmup.stdout" 2>&1
  for shape in "256 256" "1024 512" "4096 512"; do
    set -- $shape; local IN=$1 OUT=$2
    run_one "$IN" "$OUT" 1  16
    run_one "$IN" "$OUT" 8  48
    run_one "$IN" "$OUT" 32 128
    run_one "$IN" "$OUT" 64 192
  done
}

echo "===== RETRY $SUB ($REPO) ====="
free_gpu
echo "SERVE $SUB (gpu-util 0.85, expandable_segments)"
vllm serve "$REPO" --host $HOST --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.85 \
  > "$DIR/server.log" 2>&1 &
SVPID=$!

ready=0
for i in $(seq 1 360); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://$HOST:$PORT/health 2>/dev/null || true)
  [ "$code" = "200" ] && { ready=1; break; }
  kill -0 $SVPID 2>/dev/null || break
  sleep 2
done
if [ "$ready" != "1" ]; then
  echo "FAIL_SERVE $SUB (see $DIR/server.log)"
  free_gpu "$SVPID"
  exit 1
fi
echo "SERVED $SUB"

sweep "$RESDIR" "$REPO"
echo "SWEEP_DONE $SUB"
free_gpu "$SVPID"
echo "MODEL_DONE $SUB"
