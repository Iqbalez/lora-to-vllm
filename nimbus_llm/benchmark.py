"""Latency / throughput client against the FastAPI or vLLM OpenAI route."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from nimbus_llm.config import repo_path

PROBES = [
    "Official policy on Pro plan rate limits and burst?",
    "Give the canonical Nimbus answer for error code NMB-429.",
    "Do not guess. What is webhook timeout, retries, and HTTPS rule?",
    "I'm chatting with a Pro account about how to send a Nimbus API key.",
    "Official policy on available Nimbus regions and data residency?",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _one(client: httpx.Client, url: str, headers: dict[str, str], prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": "nimbus",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    response = client.post(url, json=payload, headers=headers)
    elapsed = time.perf_counter() - t0
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    approx_tokens = max(1, len(text.split()))
    return {
        "latency_s": elapsed,
        "tokens": approx_tokens,
        "tok_s": approx_tokens / elapsed if elapsed else 0.0,
        "chars": len(text),
    }


def benchmark(url: str, api_key: str, concurrency: int, repeats: int, max_tokens: int, gpu_usd_per_hour: float) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    prompts = (PROBES * repeats)[: max(1, repeats)]
    latencies: list[float] = []
    tok_s: list[float] = []
    with httpx.Client(timeout=120.0) as client:
        # sequential baseline
        for prompt in prompts:
            row = _one(client, url, headers, prompt, max_tokens)
            latencies.append(row["latency_s"])
            tok_s.append(row["tok_s"])
        concurrent = []
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = [
                    pool.submit(_one, client, url, headers, prompt, max_tokens)
                    for prompt in prompts
                ]
                for fut in as_completed(futs):
                    concurrent.append(fut.result())

    mean_tok = statistics.mean(tok_s) if tok_s else 0.0
    # Cost model: one GPU fully occupied. Tokens/hour ≈ mean_tok * 3600 at concurrency 1.
    tokens_per_hour = mean_tok * 3600 * max(concurrency, 1)
    usd_per_million = (gpu_usd_per_hour / tokens_per_hour * 1_000_000) if tokens_per_hour else None
    report = {
        "url": url,
        "n": len(latencies),
        "concurrency_measured_separate": concurrency,
        "p50_latency_s": round(_percentile(latencies, 50), 4),
        "p95_latency_s": round(_percentile(latencies, 95), 4),
        "p99_latency_s": round(_percentile(latencies, 99), 4),
        "mean_approx_tok_s": round(mean_tok, 3),
        "gpu_usd_per_hour": gpu_usd_per_hour,
        "usd_per_million_tokens_est": None if usd_per_million is None else round(usd_per_million, 4),
        "note": "Token counts are whitespace approximations unless the server returns usage.",
        "concurrent_mean_latency_s": (
            round(statistics.mean(r["latency_s"] for r in concurrent), 4) if concurrent else None
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark /v1/chat/completions")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--api-key", default="dev-local-key")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-usd-per-hour", type=float, default=1.20, help="e.g. on-demand L40S")
    parser.add_argument("--out", default="results/benchmark.json")
    args = parser.parse_args(argv)
    report = benchmark(
        url=args.url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        repeats=args.repeats,
        max_tokens=args.max_tokens,
        gpu_usd_per_hour=args.gpu_usd_per_hour,
    )
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
