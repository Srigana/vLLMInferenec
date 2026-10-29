"""Print markdown tables from the JSON files in results/."""
import json
from pathlib import Path

RESULTS = Path("results")


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    batching, serving, regression = [], [], []
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text())
        if f.stem.startswith("regression_"):
            regression.append(d)
        elif "mode" in d and "batch_size" in d:
            batching.append(d)
        elif "output_throughput" in d:
            serving.append((f.stem, d))

    if batching:
        print("## Static vs continuous batching\n")
        print("| Mode | Batch | Fixed len | Output tok/s | Req/s | Latency p50 (ms) | Latency p99 (ms) | TTFT p50 (ms) |")
        print("|---|---|---|---|---|---|---|---|")
        for d in sorted(batching, key=lambda d: (d["fixed_output_len"] or 0, d["batch_size"], d["mode"])):
            print(f"| {d['mode']} | {d['batch_size']} | {d['fixed_output_len'] or 'mixed'} | "
                  f"{d['output_tok_per_s']} | {d['req_per_s']} | {d['latency_ms_p50']} | "
                  f"{d['latency_ms_p99']} | {d['ttft_ms_p50']} |")
        print()

    if serving:
        print("## vllm bench serve\n")
        print("| Run | Output tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p99 (ms) |")
        print("|---|---|---|---|---|---|")
        for name, d in serving:
            print(f"| {name} | {fmt(d.get('output_throughput'))} | {fmt(d.get('median_ttft_ms'))} | "
                  f"{fmt(d.get('p99_ttft_ms'))} | {fmt(d.get('median_tpot_ms'), 2)} | "
                  f"{fmt(d.get('p99_e2el_ms'))} |")
        print()

    if regression:
        print("## Accuracy regression\n")
        print("| Config | Suite | Exact match | Mismatches |")
        print("|---|---|---|---|")
        for d in regression:
            print(f"| {d['config']} | {d['suite']} | {d['exact_match']:.1%} | {len(d['mismatches'])} |")
        print()


if __name__ == "__main__":
    main()
