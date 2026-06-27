#!/usr/bin/env python3
"""Latency / throughput benchmark for the stack, through Bifrost.

Measures the metrics that actually matter for an LLM server, using streaming
so timing is real (not just end-to-end):

  - TTFT  (time to first token)      -- dominated by prompt processing / prefill
  - Prefill rate  = prompt_tokens / TTFT          (tok/s, approximate)
  - Decode rate   = completion_tokens / (total - TTFT)   (tok/s, generation)
  - E2E latency   = full request wall time
  - Aggregate output throughput across all requests at a given concurrency

Exact token counts come from the stream's usage chunk
(`stream_options.include_usage`), with a chunk-count fallback.

Examples:
  # single-stream baseline
  python scripts/benchmark.py --requests 5 --concurrency 1

  # concurrency sweep to find where it degrades
  for c in 1 2 4 8 16; do
    python scripts/benchmark.py --requests $((c*3)) --concurrency $c --quiet
  done
"""

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    sys.exit("error: `pip install openai` first (same dep as test.py/demo).")

DEFAULT_PROMPT = (
    "You are a careful technical writer. Explain, step by step and in concrete "
    "detail, how a transformer-based large language model processes a prompt and "
    "generates text: tokenization, embeddings, attention, the KV cache, sampling, "
    "and why batching matters for throughput. Use clear prose, not bullet points."
)


def pct(values, p):
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


def one_request(client, model, prompt, max_tokens):
    """Run one streaming completion; return a metrics dict (or {'error': ...})."""
    start = time.perf_counter()
    ttft = None
    content_chunks = 0
    prompt_tokens = 0
    completion_tokens = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    content_chunks += 1
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the run
        return {"error": f"{type(exc).__name__}: {exc}"}

    total = time.perf_counter() - start
    if completion_tokens == 0:  # server didn't return usage -> approximate
        completion_tokens = content_chunks
    if ttft is None:
        ttft = total
    decode_time = max(total - ttft, 1e-9)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft": ttft,
        "total": total,
        "decode_tps": completion_tokens / decode_time,
        "prefill_tps": (prompt_tokens / ttft) if (ttft > 0 and prompt_tokens) else 0.0,
    }


def summarize(label, values, unit, prec=1):
    if not values:
        print(f"  {label:<28} (no data)")
        return
    print(
        f"  {label:<28} mean {statistics.mean(values):>8.{prec}f}  "
        f"p50 {pct(values, 50):>8.{prec}f}  "
        f"p95 {pct(values, 95):>8.{prec}f}  {unit}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=os.environ.get("BIFROST_URL", "http://localhost:8080"),
                    help="Bifrost base URL (default: http://localhost:8080)")
    ap.add_argument("--vk", default=os.environ.get("BENCH_VK",
                    "sk-bf-f3a27705a3f6c8af23a6a31d9b78f292c1eb65752d346837"),
                    help="virtual key for the x-bf-vk header (or set BENCH_VK)")
    ap.add_argument("--model", default=os.environ.get(
                    "BENCH_MODEL", "local-vllm/qwen2.5-14b-instruct-awq"),
                    help="provider/model id to benchmark")
    ap.add_argument("--requests", type=int, default=10, help="total requests to send")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel in-flight requests")
    ap.add_argument("--max-tokens", type=int, default=128, help="generation length per request")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt text")
    ap.add_argument("--no-warmup", action="store_true", help="skip the warmup request")
    ap.add_argument("--quiet", action="store_true", help="one-line summary (good for sweeps)")
    args = ap.parse_args()

    client = OpenAI(
        base_url=f"{args.url}/openai",
        api_key="unused",
        default_headers={"x-bf-vk": args.vk},
        max_retries=0,
        timeout=120.0,
    )

    if not args.no_warmup:
        warm = one_request(client, args.model, "ping", 1)
        if "error" in warm:
            sys.exit(f"[FAIL] warmup request failed: {warm['error']}\n"
                     f"       check the stack is up, the model id, and the virtual key.")

    # Fire `requests` jobs, at most `concurrency` running at once.
    results, errors = [], []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(one_request, client, args.model, args.prompt, args.max_tokens)
                   for _ in range(args.requests)]
        for f in as_completed(futures):
            r = f.result()
            (errors if "error" in r else results).append(r)
    wall = time.perf_counter() - wall_start

    ok = len(results)
    out_tokens = sum(r["completion_tokens"] for r in results)
    in_tokens = sum(r["prompt_tokens"] for r in results)
    agg_out_tps = out_tokens / wall if wall > 0 else 0
    per_user_decode = statistics.mean([r["decode_tps"] for r in results]) if ok else 0

    if args.quiet:
        print(f"concurrency={args.concurrency:<3} ok={ok}/{args.requests} "
              f"errors={len(errors)}  agg_out={agg_out_tps:6.1f} tok/s  "
              f"per_req_decode≈{per_user_decode:5.1f} tok/s  "
              f"ttft_p95={pct([r['ttft'] for r in results],95):.2f}s")
        return

    print("\n" + "=" * 64)
    print(f"  model        {args.model}")
    print(f"  requests     {args.requests}   concurrency {args.concurrency}   "
          f"max_tokens {args.max_tokens}")
    print(f"  ok {ok}   errors {len(errors)}   wall {wall:.2f}s")
    print("=" * 64)
    summarize("TTFT (prompt processing)", [r["ttft"] for r in results], "s", prec=3)
    summarize("Decode rate (per request)", [r["decode_tps"] for r in results], "tok/s")
    summarize("Prefill rate (per request)", [r["prefill_tps"] for r in results], "tok/s")
    summarize("E2E latency", [r["total"] for r in results], "s", prec=3)
    print("-" * 64)
    print(f"  prompt tokens (total)        {in_tokens}")
    print(f"  output tokens (total)        {out_tokens}")
    print(f"  AGGREGATE output throughput  {agg_out_tps:.1f} tok/s "
          f"(across {args.concurrency} concurrent)")
    if errors:
        print("-" * 64)
        print(f"  first error: {errors[0]['error']}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
