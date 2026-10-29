#!/usr/bin/env bash
# Start vLLM with one of the benchmark configs and block until it is healthy.
# Usage: ./serve.sh {baseline|prefix|spec-ngram|spec-draft|all}
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DRAFT="${DRAFT:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
CONFIG="${1:-baseline}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"

COMMON=(--port "$PORT" --gpu-memory-utilization 0.85 --max-model-len 8192 --seed 0)
NGRAM='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
DRAFT_CFG="{\"method\":\"draft_model\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":4}"

case "$CONFIG" in
  baseline)   EXTRA=(--no-enable-prefix-caching) ;;
  prefix)     EXTRA=(--enable-prefix-caching) ;;
  spec-ngram) EXTRA=(--no-enable-prefix-caching --speculative-config "$NGRAM") ;;
  spec-draft) EXTRA=(--no-enable-prefix-caching --speculative-config "$DRAFT_CFG") ;;
  all)        EXTRA=(--enable-prefix-caching --speculative-config "$DRAFT_CFG") ;;
  *) echo "unknown config: $CONFIG" >&2; exit 1 ;;
esac

# Stop any previous server and wait for it to release the GPU.
if pgrep -f "vllm serve" > /dev/null; then
  pkill -f "vllm serve" || true
  for _ in $(seq 1 60); do
    pgrep -f "vllm serve" > /dev/null || break
    sleep 1
  done
  pkill -9 -f "vllm serve" 2>/dev/null || true
  sleep 2
fi

mkdir -p logs
LOG="logs/server_${CONFIG}.log"
nohup vllm serve "$MODEL" "${COMMON[@]}" "${EXTRA[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$CONFIG" > logs/current_config
echo "Starting $CONFIG (pid $PID, log $LOG)"

for _ in $(seq 1 $((STARTUP_TIMEOUT / 2))); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null; then
    echo "Ready: $CONFIG"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Server exited during startup. Last lines of $LOG:" >&2
    tail -n 40 "$LOG" >&2
    exit 1
  fi
  sleep 2
done

echo "Server not healthy after ${STARTUP_TIMEOUT}s, see $LOG" >&2
exit 1
