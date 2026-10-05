# Nimbus support dataset

This dataset is original. It is generated from `data/raw/product_facts.yaml`
and paraphrase templates in `nimbus_llm/dataset.py`. Nothing is scraped from
vendor docs.

## Goal

Teach a small open-source chat model the **Nimbus API** support facts:
plans, rate limits, error codes, webhooks, regions, SSO, SLA, and the
"I don't know" policy.

## Splits

| Split | Role | Leakage rule |
|---|---|---|
| `train.jsonl` | SFT | All product facts, many phrasings |
| `val.jsonl` | Early stopping / LR sanity | Same facts, **disjoint question stems** |
| `test.jsonl` | Held-out scoring | Same facts, **held-out stems** + trap questions |

Train, val, and test never share an exact user utterance. They *do* share
facts on purpose: this is a closed-domain assistant, so eval measures
whether the model recites the right numbers under new wording.

Trap items (`topic: unknown`) have **no** supporting fact. The correct
behavior is to refuse and point to `support@nimbus.dev`. Inventing a
code such as `NMB-999` is a hallucination fail.

## Record schema

```json
{
  "id": "train-rate.pro-0007",
  "split": "train",
  "topic": "rate_limits",
  "fact_id": "rate.pro",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "must_include": ["600", "Pro"],
  "must_not_include": ["NMB-999"]
}
```

`must_include` / `must_not_include` are the eval rubric. Do not tune them
on test outputs.

## Regeneration

```bash
python -m nimbus_llm dataset --out data/processed
```

Fit, select, and threshold only on train/val. Never read test labels
while changing training hyperparameters except as a frozen final report.
