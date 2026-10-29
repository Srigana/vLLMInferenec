#!/usr/bin/env bash
# Run a benchmark suite against the server started by serve.sh.
# Usage: ./run_benchmarks.sh {prefix|spec} <label>
# <label> is used in the result filenames, normally the serve.sh config name.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {prefix|spec} <label>" >&2
  exit 1
fi

SUITE="$1"
LABEL="$2"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"

BASE=(--base-url "http://localhost:${PORT}" --model "$MODEL"
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
      --save-result --result-dir results)
mkdir -p results

case "$SUITE" in
  prefix)
    # 4 shared 2048-token prefixes, each request adds a unique 256-token suffix.
    for C in 1 32; do
      vllm bench serve "${BASE[@]}" --dataset-name prefix_repetition \
        --prefix-repetition-prefix-len 2048 --prefix-repetition-suffix-len 256 \
        --prefix-repetition-num-prefixes 4 --prefix-repetition-output-len 128 \
        --num-prompts 256 --max-concurrency "$C" --request-rate inf --ignore-eos \
        --result-filename "prefix_overlap_${LABEL}_c${C}.json"
    done
    # Control: one prefix per request, so there is nothing to reuse.
    vllm bench serve "${BASE[@]}" --dataset-name prefix_repetition \
      --prefix-repetition-prefix-len 2048 --prefix-repetition-suffix-len 256 \
      --prefix-repetition-num-prefixes 256 --prefix-repetition-output-len 128 \
      --num-prompts 256 --max-concurrency 32 --request-rate inf --ignore-eos \
      --result-filename "prefix_nooverlap_${LABEL}_c32.json"
    ;;

  spec)
    # InstructCoder: code-editing prompts whose answers repeat a lot of the input,
    # which gives n-gram speculation something to match. Use --dataset-name random
    # instead to see the worst case.
    for C in 1 4 16 64; do
      N=$(( C < 16 ? 64 : 256 ))
      vllm bench serve "${BASE[@]}" --dataset-name hf --dataset-path likaixin/InstructCoder \
        --hf-output-len 256 --temperature 0 \
        --num-prompts "$N" --max-concurrency "$C" --request-rate inf \
        --result-filename "spec_${LABEL}_c${C}.json"
    done
    # Cumulative acceptance counters for the whole sweep (empty for non-spec configs).
    curl -s "http://localhost:${PORT}/metrics" | grep -E "^vllm:spec_decode" \
      > "results/spec_metrics_${LABEL}.txt" || true
    ;;

  *)
    echo "suite must be prefix or spec" >&2
    exit 1
    ;;
esac
