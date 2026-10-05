"""Evaluate a base checkpoint vs an optional LoRA adapter on held-out stems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nimbus_llm.chat import apply_template, ensure_pad_token, prompt_messages
from nimbus_llm.config import repo_path
from nimbus_llm.dataset import read_jsonl
from nimbus_llm.scoring import aggregate, pairwise_win, score_example


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_causal(model_name: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_one(tokenizer, model, messages: list[dict[str, str]], max_new_tokens: int = 128) -> str:
    prompt = apply_template(tokenizer, prompt_messages(messages), add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        device = model.device
    except Exception:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.split("<|im_end|>")[0].strip()


def run_split(tokenizer, model, rows: list[dict[str, Any]], max_new_tokens: int, limit: int | None) -> list[dict[str, Any]]:
    selected = rows[:limit] if limit else rows
    results = []
    for row in selected:
        pred = generate_one(tokenizer, model, row["messages"], max_new_tokens=max_new_tokens)
        scored = score_example(pred, row)
        scored["prediction"] = pred
        results.append(scored)
    return results


def evaluate(
    base_model: str,
    test_path: Path,
    adapter_path: str | None,
    out_path: Path,
    max_new_tokens: int,
    limit: int | None,
) -> dict[str, Any]:
    rows = read_jsonl(test_path)
    tokenizer, base = load_causal(base_model, adapter_path=None)
    base_rows = run_split(tokenizer, base, rows, max_new_tokens, limit)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_rows = None
    wins = {"adapter": 0, "base": 0, "tie": 0}
    if adapter_path:
        tokenizer, adapted = load_causal(base_model, adapter_path=adapter_path)
        adapter_rows = run_split(tokenizer, adapted, rows, max_new_tokens, limit)
        for b, a in zip(base_rows, adapter_rows):
            wins[pairwise_win(b, a)] += 1
        del adapted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "test_path": str(test_path),
        "limit": limit,
        "base": aggregate(base_rows),
        "adapter": aggregate(adapter_rows) if adapter_rows is not None else None,
        "pairwise": wins if adapter_rows is not None else None,
        "base_examples": base_rows,
        "adapter_examples": adapter_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    printable = {k: report[k] for k in ("base_model", "adapter_path", "base", "adapter", "pairwise")}
    print(json.dumps(printable, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score base vs adapter on held-out Nimbus facts")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--test", default="data/processed/test.jsonl")
    parser.add_argument("--out", default="results/eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = parser.parse_args(argv)
    evaluate(
        base_model=args.base,
        test_path=repo_path(args.test) if not Path(args.test).is_absolute() else Path(args.test),
        adapter_path=args.adapter or None,
        out_path=repo_path(args.out) if not Path(args.out).is_absolute() else Path(args.out),
        max_new_tokens=args.max_new_tokens,
        limit=args.limit or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
