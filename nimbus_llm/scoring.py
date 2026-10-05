"""Deterministic fact-coverage scoring. No judge model required."""

from __future__ import annotations

from typing import Any


def contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def score_example(prediction: str, row: dict[str, Any]) -> dict[str, Any]:
    must = list(row.get("must_include") or [])
    banned = list(row.get("must_not_include") or [])
    hits = [item for item in must if contains(prediction, item)]
    misses = [item for item in must if item not in hits]
    leaks = [item for item in banned if contains(prediction, item)]
    coverage = 1.0 if not must else len(hits) / len(must)
    clean = 1.0 if not leaks else 0.0
    passed = coverage >= 1.0 and clean >= 1.0
    return {
        "id": row.get("id"),
        "topic": row.get("topic"),
        "fact_id": row.get("fact_id"),
        "coverage": round(coverage, 4),
        "clean": clean,
        "pass": passed,
        "hits": hits,
        "misses": misses,
        "leaks": leaks,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "fact_coverage": 0.0, "pass_rate": 0.0, "clean_rate": 0.0}
    n = len(rows)
    return {
        "n": n,
        "fact_coverage": round(sum(r["coverage"] for r in rows) / n, 4),
        "pass_rate": round(sum(1 for r in rows if r["pass"]) / n, 4),
        "clean_rate": round(sum(r["clean"] for r in rows) / n, 4),
    }


def pairwise_win(base: dict[str, Any], adapter: dict[str, Any]) -> str:
    """Adapter wins if it covers more facts without extra leaks; else base; else tie."""
    if adapter["clean"] < base["clean"]:
        return "base"
    if adapter["clean"] > base["clean"]:
        return "adapter"
    if adapter["coverage"] > base["coverage"]:
        return "adapter"
    if adapter["coverage"] < base["coverage"]:
        return "base"
    return "tie"
