#!/usr/bin/env bash
# Long-context sweep at each model's NATIVE max context (no YaRN, no quality loss).
# Single L40S (44.4 GiB) can't hold 256K KV; this tests the real trained ceiling:
#   phi-4     -> 16K  (max_position_embeddings 16384)
#   Qwen3-32B -> 40K  (max_position_embeddings 40960)
# Results go to <model>/results_longctx/ so the 8K baseline is untouched.
# --max-num-seqs 16 caps CUDA-graph capture (just above our max concurrency 8) to
# maximize the KV pool and avoid the graph-capture OOM seen on qwen3.6-27b.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export HF_HOME=/mnt/models/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ec2-user/ranjeet/bench
HOST=127.0.0.1; PORT=8000

# repo_id | subfolder | max_model_len | space-separated "IN OUT" shapes
MODELS=(
  "stelterlab/phi-4-AWQ|phi-4-awq|16384|8192 512;15360 512"
  "Qwen/Qwen3-32B-AWQ|qwen3-32b-awq|40960|8192 512;32768 512;40448 512"
)
CONCS=(1 4 8)
prompts_for () { case "$1" in 1) echo 8;; 4) echo 16;; 8) echo 24;; *) echo 8;; esac; }

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

echo "FREE GPU"
free_gpu

for entry in "${MODELS[@]}"; do
  IFS='|' read -r REPO SUB MML SHAPES <<< "$entry"
  DIR="$ROOT/$SUB"; RESDIR="$DIR/results_longctx"; mkdir -p "$RESDIR"
  echo "===== LONGCTX $SUB ($REPO) max_model_len=$MML ====="

  free_gpu
  echo "SERVE $SUB (max-model-len $MML, max-num-seqs 16)"
  vllm serve "$REPO" --host $HOST --port $PORT --max-model-len "$MML" \
    --gpu-memory-utilization 0.92 --max-num-seqs 16 \
    > "$DIR/server_longctx.log" 2>&1 &
  SVPID=$!

  ready=0
  for i in $(seq 1 360); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://$HOST:$PORT/health 2>/dev/null || true)
    [ "$code" = "200" ] && { ready=1; break; }
    kill -0 $SVPID 2>/dev/null || break
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    echo "FAIL_SERVE $SUB (see $DIR/server_longctx.log)"; free_gpu "$SVPID"; continue
  fi
  echo "SERVED $SUB"
  # record the actual KV pool / concurrency the engine reports
  grep -iE "GPU KV cache size|Maximum concurrency" "$DIR/server_longctx.log" | tail -2

  # warmup (small)
  vllm bench serve --backend vllm --model "$REPO" --host $HOST --port $PORT --dataset-name random \
    --random-input-len 1024 --random-output-len 64 --random-range-ratio 0 --num-prompts 4 --max-concurrency 2 \
    --ignore-eos --seed 1 > "$RESDIR/warmup.stdout" 2>&1

  IFS=';' read -ra SHAPELIST <<< "$SHAPES"
  for shape in "${SHAPELIST[@]}"; do
    set -- $shape; IN=$1; OUT=$2
    for conc in "${CONCS[@]}"; do
      np=$(prompts_for "$conc"); tag="in${IN}_out${OUT}_c${conc}"
      if vllm bench serve --backend vllm --model "$REPO" --host $HOST --port $PORT --endpoint /v1/completions \
           --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" --random-range-ratio 0 \
           --num-prompts "$np" --max-concurrency "$conc" --ignore-eos --seed 12345 \
           --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,99" \
           --save-result --result-dir "$RESDIR" --result-filename "${tag}.json" > "$RESDIR/${tag}.stdout" 2>&1; then
        echo "OK $tag" >> "$RESDIR/_progress.txt"
      else
        echo "  FAIL_RUN $SUB/$tag"
      fi
    done
  done
  echo "SWEEP_DONE $SUB"
  free_gpu "$SVPID"
  echo "MODEL_DONE $SUB"
done
echo "ALL_LONGCTX_DONE"
