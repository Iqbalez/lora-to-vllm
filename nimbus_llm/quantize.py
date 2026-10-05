"""Merge a LoRA adapter into the base model, then AWQ-INT4 quantise (GPU/Linux)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nimbus_llm.chat import ensure_pad_token
from nimbus_llm.config import repo_path


def merge_adapter(base_model: str, adapter_path: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "merge_meta.json").write_text(
        json.dumps({"base_model": base_model, "adapter_path": adapter_path}, indent=2),
        encoding="utf-8",
    )
    return out_dir


def quantize_awq(merged_dir: Path, quant_dir: Path, calib_path: Path, max_calib: int = 64) -> Path:
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "autoawq is not installed. Use a Linux NVIDIA box or the GPU Docker "
            "path: pip install autoawq"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("AWQ quantisation needs a CUDA GPU.")

    from nimbus_llm.dataset import read_jsonl

    rows = read_jsonl(calib_path)[:max_calib]
    calib = [row["messages"][-1]["content"] for row in rows]
    quant_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(str(merged_dir), low_cpu_mem_usage=True)
    model.quantize(
        tokenizer,
        quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
        calib_data=calib,
    )
    model.save_quantized(str(quant_dir))
    tokenizer.save_pretrained(quant_dir)
    (quant_dir / "quant_meta.json").write_text(
        json.dumps({"method": "awq", "w_bit": 4, "source": str(merged_dir)}, indent=2),
        encoding="utf-8",
    )
    return quant_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge LoRA then AWQ-INT4")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--merged-out", default="outputs/merged")
    parser.add_argument("--quant-out", default="outputs/awq-int4")
    parser.add_argument("--calib", default="data/processed/train.jsonl")
    parser.add_argument("--skip-quant", action="store_true")
    args = parser.parse_args(argv)
    merged = Path(args.merged_out)
    if not merged.is_absolute():
        merged = repo_path(args.merged_out)
    quant = Path(args.quant_out)
    if not quant.is_absolute():
        quant = repo_path(args.quant_out)
    calib = Path(args.calib)
    if not calib.is_absolute():
        calib = repo_path(args.calib)
    merge_adapter(args.base, args.adapter, merged)
    if not args.skip_quant:
        quantize_awq(merged, quant, calib)
    print(json.dumps({"merged": str(merged), "quant": str(quant), "skipped_quant": args.skip_quant}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
