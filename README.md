# Nimbus Support LLM: LoRA → eval → AWQ → vLLM → FastAPI

End-to-end production path for a **domain support assistant**:

1. Build and validate an instruction dataset  
2. Supervised fine-tune an open-source chat model with **LoRA / QLoRA** (Hugging Face **Transformers + PEFT + TRL**)  
3. Score the adapter against the **base model** with a frozen fact-coverage rubric  
4. Merge the adapter and **AWQ-INT4** quantise  
5. Serve with **vLLM** (continuous batching, prefix caching) behind a **FastAPI** OpenAI-compatible API  
6. Benchmark latency, throughput, and estimated **$ / 1M tokens**

This is the portfolio repo that matches an LLM fine-tune + inference-optimisation job: dataset, training, eval, quantisation, serving, cost.

```text
dataset jsonl  →  QLoRA SFT  →  held-out eval vs base
                              →  merge + AWQ-INT4
                              →  vLLM + FastAPI + Docker
                              →  p50/p95/p99 and $/1M tokens
```

## Why these hyperparameters

| Choice | Default | Why |
|---|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` | Small enough to QLoRA on 24 GB, large enough to learn closed-domain facts |
| LoRA rank / alpha | 16 / 32 | Rank 16 on all attention + MLP projections is the usual quality/VRAM trade for 3B SFT. Alpha = 2r. |
| QLoRA 4-bit NF4 | on | Keeps the 3B + 2048 ctx + batch 2×8 inside 24 GB. Turn off on 80 GB. |
| Mixed precision | BF16 | Stable on Ampere+. FP16 fallback is automatic if BF16 is missing. |
| Effective batch | 16 | `per_device=2` × `grad_accum=8`. Same optimiser step as batch 16 without the VRAM. |
| Gradient checkpointing | on | Recompute activations, cut VRAM, slightly slower steps. |
| Eval | frozen fact strings | No LLM-as-judge. If the answer must contain `600` and `Pro`, it either does or it doesn't. |

## Dataset

Original Nimbus API facts (plans, RPM, error codes, webhooks, regions, SSO, SLA). Train / val / test use the **same facts** and **disjoint question stems**, so eval measures new wording, not memorised prompts. Trap questions have no fact; inventing `NMB-999` is a fail.

```bash
python -m nimbus_llm dataset --out data/processed
```

See `data/DATA.md`. Generated counts: **train 356 / val 148 / test 121**.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch   # CUDA wheel on the GPU box, CPU wheel for smoke
```

GPU extras (Linux + NVIDIA only):

```bash
pip install -r requirements-gpu.txt
```

## CPU smoke (this laptop, no NVIDIA)

Proves the **wiring**. Do not quote tiny-gpt2 scores as model quality.

```powershell
pip install -e ".[dev]"
python -m nimbus_llm dataset --out data/processed
python -m pytest tests/test_dataset.py tests/test_api.py
python -m nimbus_llm train --config configs/train_smoke.yaml
python -m nimbus_llm evaluate --base sshleifer/tiny-gpt2 --adapter outputs/smoke-tiny-gpt2/adapter --limit 8 --out results/eval_smoke.json --max-new-tokens 32
python -m nimbus_llm serve --backend transformers --model sshleifer/tiny-gpt2 --adapter outputs/smoke-tiny-gpt2/adapter
```

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/chat/completions ^
  -H "Authorization: Bearer dev-local-key" ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Official policy on Pro plan rate limits and burst?\"}]}"
```

| Smoke check | Result |
|---|---|
| Dataset leak test | pass (356 / 148 / 121, disjoint user strings) |
| FastAPI `/health` + `/v1/chat/completions` | pass (`tests/test_api.py`) |
| LoRA train 8 steps on `tiny-gpt2` CPU | **done** — train loss 10.75 (untrained tiny net; wiring only) |
| Base vs adapter fact coverage (8 test rows) | 0.0 vs 0.0, 8 ties — expected; this model has no language ability |

## GPU path (the actual portfolio numbers)

Hardware target: **24 GB** (A10 / RTX A5000 / L40S). A100/H100 also fine.

```bash
bash scripts/gpu_train.sh
```

That script:

1. Rebuilds jsonl  
2. QLoRA-tunes Qwen2.5-3B-Instruct (`configs/train_qwen3b.yaml`)  
3. Scores **base vs adapter** on all held-out items  
4. Merges LoRA and writes **AWQ INT4** to `outputs/awq-int4`

Serve:

```bash
docker compose -f docker/docker-compose.yml up --build
python -m nimbus_llm benchmark --url http://127.0.0.1:8000/v1/chat/completions --concurrency 8 --repeats 32 --gpu-usd-per-hour 1.20
```

vLLM flags in compose: `--quantization awq`, `--max-model-len 4096`, `--enable-prefix-caching`, `--gpu-memory-utilization 0.90`. FastAPI in front adds an API key, `/health`, and `latency_ms`.

### Fill after the GPU run

| Setup | Fact coverage | Pass rate | Win vs base | tok/s (approx) | p95 latency | VRAM | $ / 1M tok @ $1.20/hr |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct base BF16 | _run eval_ |  | — |  |  | ~8 GB |  |
| + LoRA adapter BF16 |  |  |  |  |  |  |  |
| Merged AWQ-INT4 + vLLM |  |  |  |  |  |  |  |

Paste numbers from `results/eval.json` and `results/benchmark.json`. Do not change training after looking at test.

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | backend, model, adapter |
| POST | `/v1/chat/completions` | OpenAI-shaped body, `Authorization: Bearer <NIMBUS_API_KEY>` |

Backends:

- `transformers` — local PEFT adapter, good for debug and Windows  
- `vllm` — proxies to a vLLM OpenAI server (Docker)  
- `scripted` — unit tests

## Repo map

```text
nimbus_llm/          dataset, train, evaluate, quantize, serve, benchmark
configs/             train_qwen3b.yaml, train_smoke.yaml, serve.yaml
data/raw/            product_facts.yaml (source of truth)
data/processed/      generated jsonl (reproducible)
docker/              FastAPI image + vLLM compose
scripts/             smoke.ps1 / smoke.sh / gpu_train.sh
tests/               schema + API tests (no GPU, no 3B download)
```

## What this is not

- Not a RAG demo. Retrieval is a later extra in front of the same API.  
- Not a 70B flex. The 3B path is the one you can actually own, measure, and serve.  
- Not trained on the hidden test stems. Hyperparameters are locked in `configs/train_qwen3b.yaml`.
