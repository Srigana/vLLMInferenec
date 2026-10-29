"""Static vs continuous batching against a running vLLM server.

vLLM always schedules continuously on the server, so static batching is
reproduced on the client. Both modes allow at most --batch-size requests in
flight:

  static      requests go out in waves of B; a wave starts only after every
              request in the previous wave has finished
  continuous  a new request starts as soon as any in-flight request finishes
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

TOPICS = [
    "the French Revolution", "how TCP works", "photosynthesis", "black holes",
    "the stock market", "sourdough baking", "the Roman Empire", "neural networks",
    "climate change", "jazz history", "the immune system", "plate tectonics",
]


def make_workload(n, seed, fixed_len=None):
    rng = random.Random(seed)
    work = []
    for i in range(n):
        prompt = f"Write a detailed explanation of {rng.choice(TOPICS)}. (request {i})"
        out_len = fixed_len if fixed_len else rng.randint(32, 512)
        work.append((prompt, out_len))
    return work


async def one_request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    stream = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        # Generate exactly max_tokens so output length is controlled by the workload.
        extra_body={"ignore_eos": True},
    )
    async for chunk in stream:
        if ttft is None and chunk.choices and chunk.choices[0].text:
            ttft = time.perf_counter() - t0
        if chunk.usage:
            n_tokens = chunk.usage.completion_tokens
    latency = time.perf_counter() - t0
    return {"latency": latency, "ttft": ttft if ttft is not None else latency, "tokens": n_tokens}


async def run_static(client, model, work, b):
    results = []
    for i in range(0, len(work), b):
        wave = work[i:i + b]
        results += await asyncio.gather(*(one_request(client, model, p, m) for p, m in wave))
    return results


async def run_continuous(client, model, work, b):
    sem = asyncio.Semaphore(b)

    async def limited(p, m):
        async with sem:
            return await one_request(client, model, p, m)

    return await asyncio.gather(*(limited(p, m) for p, m in work))


async def bench(args, mode, b):
    client = AsyncOpenAI(base_url=f"{args.url}/v1", api_key="EMPTY", timeout=600)
    model = (await client.models.list()).data[0].id
    work = make_workload(args.num_requests, args.seed, args.fixed_output_len)
    await one_request(client, model, "Hello", 8)  # warmup

    t0 = time.perf_counter()
    runner = run_static if mode == "static" else run_continuous
    res = await runner(client, model, work, b)
    wall = time.perf_counter() - t0
    await client.close()

    lat = np.array([r["latency"] for r in res]) * 1000
    ttft = np.array([r["ttft"] for r in res]) * 1000
    toks = sum(r["tokens"] for r in res)
    summary = {
        "mode": mode,
        "batch_size": b,
        "num_requests": len(res),
        "fixed_output_len": args.fixed_output_len,
        "wall_s": round(wall, 2),
        "output_tok_per_s": round(toks / wall, 1),
        "req_per_s": round(len(res) / wall, 2),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p99": round(float(np.percentile(lat, 99)), 1),
        "ttft_ms_p50": round(float(np.percentile(ttft, 50)), 1),
        "ttft_ms_p99": round(float(np.percentile(ttft, 99)), 1),
    }
    print(json.dumps(summary))

    suffix = f"_fixed{args.fixed_output_len}" if args.fixed_output_len else ""
    out = Path("results") / f"batching_{mode}_b{b}{suffix}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["static", "continuous"], default="continuous")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-requests", type=int, default=512)
    ap.add_argument("--fixed-output-len", type=int, default=None,
                    help="give every request the same output length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="run both modes at batch sizes 1, 8, 32, 64")
    args = ap.parse_args()

    if args.sweep:
        for b in (1, 8, 32, 64):
            for mode in ("static", "continuous"):
                asyncio.run(bench(args, mode, b))
    else:
        asyncio.run(bench(args, args.mode, args.batch_size))


if __name__ == "__main__":
    main()
