# vllm-serving-lab

Serving Qwen2.5-1.5B-Instruct with vLLM and measuring three serving optimizations:

1. Continuous batching vs static batching
2. Automatic prefix caching
3. Speculative decoding (n-gram and a Qwen2.5-0.5B draft model)

A regression suite checks that each configuration still produces the baseline's outputs.

## Setup

Needs Linux, an NVIDIA GPU with at least 16 GB, and Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The models download from Hugging Face the first time the server starts. To use different
models, set `MODEL` and `DRAFT`. The draft model has to use the same tokenizer as the target.

Run everything from the repo root.

## Server configs

`./serve.sh <config>` stops any running server, starts the new one in the background, and
waits until `/health` responds. Logs go to `logs/server_<config>.log`.

| Config | Prefix caching | Speculative decoding |
|---|---|---|
| `baseline` | off | off |
| `prefix` | on | off |
| `spec-ngram` | off | n-gram, 4 tokens |
| `spec-draft` | off | Qwen2.5-0.5B draft, 4 tokens |
| `all` | on | Qwen2.5-0.5B draft, 4 tokens |

Recent vLLM versions enable prefix caching by default, so it is turned off explicitly
everywhere except `prefix` and `all`.

## Experiments

### Continuous vs static batching

The vLLM scheduler is always continuous, so static batching is reproduced in the client
(`bench_batching.py`). Both modes keep at most B requests in flight. In static mode the
requests go out in waves and the next wave waits until every request in the current one has
finished. In continuous mode a new request starts as soon as any slot frees up.

Output lengths are drawn uniformly from 32 to 512 tokens, and `ignore_eos` makes each request
generate exactly that many.

```bash
./serve.sh baseline
python bench_batching.py --mode static --batch-size 32
python bench_batching.py --mode continuous --batch-size 32
python bench_batching.py --sweep                         # B = 1, 8, 32, 64, both modes
python bench_batching.py --sweep --fixed-output-len 256  # control: equal output lengths
```

The fixed-length run is a control. When every request has the same length, a static wave
no longer waits on one long request, so the gap between the two modes should mostly close.

### Prefix caching

```bash
./serve.sh baseline && ./run_benchmarks.sh prefix baseline
./serve.sh prefix   && ./run_benchmarks.sh prefix prefix
```

Uses the `prefix_repetition` dataset from `vllm bench serve`: 4 shared prefixes of 2048 tokens,
each request adding a unique 256-token suffix, at concurrency 1 and 32. A second workload gives
every request its own prefix, to measure what caching costs when nothing can be reused.

### Speculative decoding

```bash
./serve.sh baseline   && ./run_benchmarks.sh spec baseline
./serve.sh spec-ngram && ./run_benchmarks.sh spec spec-ngram
./serve.sh spec-draft && ./run_benchmarks.sh spec spec-draft
```

Runs the InstructCoder dataset at concurrency 1, 4, 16 and 64 with greedy sampling. After each
sweep the `vllm:spec_decode_*` counters are saved to `results/spec_metrics_<config>.txt`, which
gives the draft acceptance rate.

### Accuracy regression

```bash
./serve.sh baseline && python make_golden.py   # writes golden/baseline.json
./serve.sh spec-draft && pytest -s tests       # repeat for each config
```

`make_golden.py` records greedy outputs from the baseline server for 60 open-ended prompts and
the first 200 GSM8K test questions. The tests send the same prompts to the running server and
check:

- exact-match rate against the baseline outputs (at least 95%)
- GSM8K accuracy within 1.5 points of the baseline
- identical prompts within one run produce identical outputs

For every mismatch the tests record the character where the output first differs from the
baseline. Results are written to `results/regression_<suite>_<config>.json`.

Thresholds can be changed with `MIN_EXACT_MATCH`, `MAX_GSM8K_DROP` and `NUM_GSM8K`.

## Summary

```bash
python summarize.py > RESULTS.md
```

## Layout

```
serve.sh                  start vLLM in one of the configs above
bench_batching.py         static vs continuous batching client
run_benchmarks.sh         prefix caching and speculative decoding sweeps
make_golden.py            record baseline outputs
tests/common.py           client and GSM8K helpers
tests/test_regression.py  regression tests
summarize.py              results/*.json to markdown tables
```

