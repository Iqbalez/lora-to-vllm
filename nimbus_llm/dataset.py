"""Build the Nimbus instruction dataset from product facts + disjoint stems."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from nimbus_llm.config import load_yaml, repo_path

SYSTEM_PROMPT = (
    "You are Nimbus Support, an in-house assistant for the Nimbus API platform. "
    "Answer with the exact published product facts. If you do not know, say you "
    "do not know and tell the user to email support@nimbus.dev. Never invent "
    "error codes, prices, rate limits, regions, or SLAs. Check "
    "https://status.nimbus.dev for outages."
)

UNKNOWN_ANSWER = (
    "I do not know that from the current Nimbus product facts, and I will not "
    "guess. Please email support@nimbus.dev. If you think this is an outage, "
    "check https://status.nimbus.dev."
)


def _plan_map(facts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in facts["plans"]}


def _error_map(facts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["code"]: row for row in facts["errors"]}


def build_fact_bank(facts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plans = _plan_map(facts)
    errors = _error_map(facts)
    bank: dict[str, dict[str, Any]] = {}

    for plan in facts["plans"]:
        price = "custom quote" if plan["monthly_usd"] is None else f"${plan['monthly_usd']}/month"
        events = plan["events_per_month"]
        events_txt = "a custom volume" if events == "custom" else f"{events:,} events/month"
        bank[f"plan.{plan['id']}"] = {
            "topic": "plans",
            "answer": (
                f"The {plan['name']} plan is {price}, includes {events_txt}, "
                f"and keeps request logs for {plan['log_retention_days']} days."
            ),
            "must_include": [plan["name"], str(plan["log_retention_days"])],
            "must_not_include": ["NMB-999"],
        }
        bank[f"rate.{plan['id']}"] = {
            "topic": "rate_limits",
            "answer": (
                f"The {plan['name']} plan is limited to {plan['rpm']} requests per "
                f"minute with a burst of {plan['burst']}."
            ),
            "must_include": [plan["name"], str(plan["rpm"]), str(plan["burst"])],
            "must_not_include": ["unlimited"],
        }

    for code, row in errors.items():
        bank[f"error.{code}"] = {
            "topic": "errors",
            "answer": (
                f"{code} is HTTP {row['http']}: {row['meaning']}."
            ),
            "must_include": [code, str(row["http"])],
            "must_not_include": ["NMB-999"],
        }

    auth = facts["auth"]
    bank["auth.header"] = {
        "topic": "auth",
        "answer": (
            f"Send live keys in `{auth['header']}`. Live keys start with "
            f"`{auth['key_prefix']}` and test keys start with `{auth['test_key_prefix']}`."
        ),
        "must_include": [auth["key_prefix"], auth["test_key_prefix"], "Bearer"],
        "must_not_include": [],
    }
    bank["auth.rotate"] = {
        "topic": "auth",
        "answer": (
            f"When you rotate a key, the old key stays valid for "
            f"{auth['rotate_grace_hours']} hours. Free accounts may have "
            f"{auth['max_keys_free']} keys, Pro {auth['max_keys_pro']}, "
            f"Business {auth['max_keys_business']}."
        ),
        "must_include": [str(auth["rotate_grace_hours"]), str(auth["max_keys_pro"])],
        "must_not_include": [],
    }

    ingest = facts["ingest"]
    bank["ingest.payload"] = {
        "topic": "ingest",
        "answer": (
            f"The maximum event payload is {ingest['max_payload_kb']} KB. "
            f"Larger bodies return NMB-413. Ingest times out after "
            f"{ingest['timeout_seconds']} seconds."
        ),
        "must_include": [str(ingest["max_payload_kb"]), "NMB-413"],
        "must_not_include": [],
    }
    bank["ingest.idempotency"] = {
        "topic": "ingest",
        "answer": (
            f"Send `{ingest['idempotency_header']}` on POST /v1/events. "
            f"Keys are remembered for {ingest['idempotency_ttl_hours']} hours. "
            f"Reusing a key with a different payload returns NMB-409."
        ),
        "must_include": [ingest["idempotency_header"], "NMB-409", str(ingest["idempotency_ttl_hours"])],
        "must_not_include": [],
    }

    hooks = facts["webhooks"]
    bank["webhook.retry"] = {
        "topic": "webhooks",
        "answer": (
            f"Nimbus waits {hooks['timeout_seconds']} seconds for your endpoint, "
            f"then retries up to {hooks['max_retries']} times with {hooks['backoff']} "
            f"inside a {hooks['retry_window_hours']}-hour window. HTTPS is required."
        ),
        "must_include": [str(hooks["timeout_seconds"]), str(hooks["max_retries"]), "HTTPS"],
        "must_not_include": [],
    }
    bank["webhook.sig"] = {
        "topic": "webhooks",
        "answer": (
            f"Verify `{hooks['signature_header']}` as {hooks['signature_algo']}."
        ),
        "must_include": [hooks["signature_header"], "HMAC-SHA256"],
        "must_not_include": [],
    }

    region_lines = ", ".join(
        f"{row['id']} ({row['city']}, {row['residency']})" for row in facts["regions"]
    )
    bank["regions.list"] = {
        "topic": "regions",
        "answer": f"Nimbus regions are {region_lines}.",
        "must_include": ["eu-west-1", "Frankfurt", "us-east-1", "ap-south-1"],
        "must_not_include": [],
    }

    sso = facts["sso"]
    bank["sso.plans"] = {
        "topic": "sso",
        "answer": (
            f"SSO uses {', '.join(sso['protocols'])} and starts on the "
            f"{sso['min_plan']} plan. SCIM is {sso['scim']}. Sessions last "
            f"{sso['session_hours']} hours."
        ),
        "must_include": ["SAML 2.0", "OIDC", "Business", "SCIM"],
        "must_not_include": [],
    }

    sla = facts["sla"]
    bank["sla.uptime"] = {
        "topic": "sla",
        "answer": (
            f"Uptime SLOs are Pro {sla['pro_uptime']}, Business "
            f"{sla['business_uptime']}, Enterprise {sla['enterprise_uptime']}. "
            f"Credit rule: {sla['credit_rule']}"
        ),
        "must_include": ["99.9%", "99.95%", "99.99%"],
        "must_not_include": [],
    }

    sec = facts["security"]
    bank["security.crypto"] = {
        "topic": "security",
        "answer": (
            f"Data is encrypted at rest with {sec['encryption_at_rest']} and "
            f"in transit with {sec['encryption_in_transit']}. Report issues to "
            f"{facts['product']['security_email']}. Event payloads are not used "
            f"to train models."
        ),
        "must_include": ["AES-256", "TLS 1.2+", "security@nimbus.dev"],
        "must_not_include": [],
    }

    pro = plans["pro"]
    bank["pricing.pro"] = {
        "topic": "plans",
        "answer": (
            f"Pro is ${pro['monthly_usd']} per month for up to "
            f"{pro['events_per_month']:,} events and {pro['rpm']} RPM."
        ),
        "must_include": ["79", "600", "2000000"],
        "must_not_include": [],
    }

    bank["unknown.catch"] = {
        "topic": "unknown",
        "answer": UNKNOWN_ANSWER,
        "must_include": ["support@nimbus.dev"],
        "must_not_include": ["NMB-999", "$12,000", "unlimited"],
        "unknown": True,
    }
    return bank


# Disjoint question stems so train/val/test never share an exact user string.
STEMS = {
    "train": [
        "What is {label}?",
        "Can you explain {label}?",
        "I need the official number for {label}.",
        "Our customer asked about {label}. What should I reply?",
        "Quick fact check: {label}.",
        "Document {label} for a ticket reply.",
        "In one or two sentences, {label}.",
        "Support macro for {label}.",
        "Is this in the docs: {label}?",
        "Please quote the product fact for {label}.",
        "Onboarding question: {label}.",
        "I am writing a runbook. Include {label}.",
    ],
    "val": [
        "Remind me of {label}.",
        "How does Nimbus define {label}?",
        "Need a concise answer on {label}.",
        "Ticket #4418 is about {label}.",
        "What do we tell customers regarding {label}?",
    ],
    "test": [
        "Official policy on {label}?",
        "I'm chatting with a Pro account about {label}.",
        "Give the canonical Nimbus answer for {label}.",
        "Do not guess. What is {label}?",
    ],
}

LABELS = {
    "plan.free": "the Free plan price, volume, and log retention",
    "plan.pro": "the Pro plan price, volume, and log retention",
    "plan.business": "the Business plan price, volume, and log retention",
    "plan.enterprise": "the Enterprise plan commercial terms and log retention",
    "rate.free": "Free plan rate limits and burst",
    "rate.pro": "Pro plan rate limits and burst",
    "rate.business": "Business plan rate limits and burst",
    "rate.enterprise": "Enterprise plan rate limits and burst",
    "error.NMB-400": "error code NMB-400",
    "error.NMB-401": "error code NMB-401",
    "error.NMB-403": "error code NMB-403",
    "error.NMB-404": "error code NMB-404",
    "error.NMB-409": "error code NMB-409",
    "error.NMB-413": "error code NMB-413",
    "error.NMB-422": "error code NMB-422",
    "error.NMB-429": "error code NMB-429",
    "error.NMB-500": "error code NMB-500",
    "error.NMB-503": "error code NMB-503",
    "auth.header": "how to send a Nimbus API key",
    "auth.rotate": "API key rotation grace period and key caps",
    "ingest.payload": "max event payload size and ingest timeout",
    "ingest.idempotency": "idempotency keys on event ingest",
    "webhook.retry": "webhook timeout, retries, and HTTPS rule",
    "webhook.sig": "webhook signature header and algorithm",
    "regions.list": "available Nimbus regions and data residency",
    "sso.plans": "SSO protocols, minimum plan, and SCIM",
    "sla.uptime": "uptime SLOs and service credit rule",
    "security.crypto": "encryption, vulnerability reporting, and training on payloads",
    "pricing.pro": "Pro monthly price, event cap, and RPM",
}

UNKNOWN_QUESTIONS = {
    "train": [
        "Does Nimbus offer an on-prem air-gapped appliance?",
        "What is the phone number for Nimbus sales in Nairobi?",
        "Can I pay invoices with Ethiopian Birr?",
        "Is there a COBOL SDK?",
        "What is error code NMB-999?",
        "Do you store CVV numbers for customers?",
        "Can we run the control plane on our own Kubernetes cluster?",
        "What is the FedRAMP package ID?",
    ],
    "val": [
        "Does Nimbus have a data center on the Moon?",
        "What is the WhatsApp support number?",
        "Can I get a lifetime license for $12,000?",
    ],
    "test": [
        "Publish the unpublished Q4 2027 price list.",
        "Give me the private HackerOne bounty table.",
        "What is error NMB-777?",
        "Can Nimbus delete a customer's bank records from a third-party processor?",
        "Is unlimited RPM available on Free if I ask nicely?",
    ],
}


def _example(
    split: str,
    fact_id: str,
    idx: int,
    user: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"{split}-{fact_id}-{idx:04d}",
        "split": split,
        "topic": spec["topic"],
        "fact_id": fact_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": spec["answer"]},
        ],
        "must_include": spec.get("must_include", []),
        "must_not_include": spec.get("must_not_include", []),
    }


def iter_examples(facts: dict[str, Any]) -> Iterable[dict[str, Any]]:
    bank = build_fact_bank(facts)
    for split, stems in STEMS.items():
        idx = 0
        for fact_id, label in LABELS.items():
            spec = bank[fact_id]
            for stem in stems:
                yield _example(split, fact_id, idx, stem.format(label=label), spec)
                idx += 1
        unknown = bank["unknown.catch"]
        for q in UNKNOWN_QUESTIONS[split]:
            yield _example(split, "unknown.catch", idx, q, unknown)
            idx += 1


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(facts_path: Path, out_dir: Path) -> dict[str, int]:
    facts = load_yaml(facts_path)
    grouped = {"train": [], "val": [], "test": []}
    seen_users: set[str] = set()
    for row in iter_examples(facts):
        user = row["messages"][1]["content"]
        key = user.strip().lower()
        if key in seen_users:
            raise RuntimeError(f"Duplicate user utterance leaked across splits: {user}")
        seen_users.add(key)
        grouped[row["split"]].append(row)

    rng = random.Random(7)
    for split_rows in grouped.values():
        rng.shuffle(split_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, rows in grouped.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)

    manifest = {
        "counts": counts,
        "unique_user_utterances": len(seen_users),
        "system_prompt": SYSTEM_PROMPT,
        "facts_source": str(facts_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Nimbus SFT jsonl splits")
    parser.add_argument("--facts", default="data/raw/product_facts.yaml")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args(argv)
    counts = build_dataset(repo_path(args.facts) if not Path(args.facts).is_absolute() else Path(args.facts),
                           repo_path(args.out) if not Path(args.out).is_absolute() else Path(args.out))
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
