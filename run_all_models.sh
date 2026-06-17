#!/usr/bin/env bash
# Benchmark a lineup of models on vLLM (single L40S), ONE AT A TIME.
# Per model: download -> serve -> wait for /health -> standard sweep -> stop server.
# Stdout = coarse progress markers (one watcher event each); per-run detail -> files.
# Continues to the next model on any failure.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export HF_HOME=/mnt/models/hf            # downloads + serving cache on the scratch NVMe

ROOT=/home/ec2-user/ranjeet/bench
HOST=127.0.0.1; PORT=8000

# "repo_id|subfolder"  — all AWQ INT4, instruct
MODELS=(
  "stelterlab/phi-4-AWQ|phi-4-awq"
  "QuantTrio/Qwen3.6-27B-AWQ|qwen3.6-27b-awq"
  "Qwen/Qwen3-32B-AWQ|qwen3-32b-awq"
  "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit|gemma-4-26b-a4b-awq"
)

# Reliably free the GPU. vLLM's worker is named "VLLM::EngineCore" and holds ALL the
# GPU memory, but its cmdline does NOT contain "vllm serve" -> a `pkill -f "vllm serve"`
# leaves it orphaned. So: kill the optional tracked server PID, then kill ANYTHING still
# holding GPU memory (by nvidia-smi compute-app PID), then wait for memory to drain.
# Kill ONLY by GPU-compute PID -- never a broad `pkill -f vllm` (would also match every
# process running from the .../envs/vllm/... path, including this script's own children).
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

sweep () {   # $1=resdir  $2=model-id
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

echo "FREE GPU (cleanup any prior engine)"
free_gpu

for entry in "${MODELS[@]}"; do
  REPO="${entry%%|*}"; SUB="${entry##*|}"
  DIR="$ROOT/$SUB"; RESDIR="$DIR/results"; mkdir -p "$RESDIR"
  echo "===== MODEL $SUB ($REPO) ====="

  echo "DOWNLOAD $SUB"
  if ! hf download "$REPO" > "$DIR/download.log" 2>&1; then echo "FAIL_DOWNLOAD $SUB"; continue; fi

  free_gpu
  echo "SERVE $SUB"
  vllm serve "$REPO" --host $HOST --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.90 \
    > "$DIR/server.log" 2>&1 &
  SVPID=$!

  ready=0
  for i in $(seq 1 360); do   # up to ~12 min for load+compile+cudagraph
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://$HOST:$PORT/health 2>/dev/null || true)
    [ "$code" = "200" ] && { ready=1; break; }
    kill -0 $SVPID 2>/dev/null || break
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    echo "FAIL_SERVE $SUB (see $DIR/server.log)"
    free_gpu "$SVPID"
    continue
  fi
  echo "SERVED $SUB"

  sweep "$RESDIR" "$REPO"
  echo "SWEEP_DONE $SUB"

  free_gpu "$SVPID"
  echo "MODEL_DONE $SUB"
done
echo "ALL_MODELS_DONE"
