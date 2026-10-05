"""SFT + LoRA/QLoRA trainer wrapping Hugging Face TRL + PEFT."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

from nimbus_llm.chat import apply_template, ensure_pad_token
from nimbus_llm.config import load_yaml, repo_path
from nimbus_llm.dataset import read_jsonl


def _precision(name: str, cuda: bool) -> dict[str, bool]:
    if not cuda:
        return {"bf16": False, "fp16": False}
    if name == "bf16" and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    if name in {"fp16", "bf16"}:
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def _load_model(model_name: str, load_in_4bit: bool):
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=kwargs["torch_dtype"],
            )
    else:
        kwargs["torch_dtype"] = torch.float32
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def _text_dataset(path: Path, tokenizer) -> Dataset:
    rows = read_jsonl(path)
    texts = [apply_template(tokenizer, row["messages"], add_generation_prompt=False) for row in rows]
    return Dataset.from_dict({"text": texts})


def _training_args_class():
    try:
        from trl import SFTConfig
        return SFTConfig
    except ImportError:
        from transformers import TrainingArguments
        return TrainingArguments


def _build_args(cfg: dict[str, Any], train_n: int):
    cuda = torch.cuda.is_available()
    t = cfg["training"]
    mixed = _precision(str(t.get("mixed_precision", "fp32")), cuda)
    output_dir = repo_path(cfg["output_dir"]) if not Path(cfg["output_dir"]).is_absolute() else Path(cfg["output_dir"])
    report_to = ["wandb"] if os.environ.get("WANDB_API_KEY") else []
    max_steps = int(t.get("max_steps") or 0)
    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        num_train_epochs=float(t["epochs"]),
        warmup_ratio=float(t.get("warmup_ratio") or 0.0),
        weight_decay=float(t.get("weight_decay") or 0.0),
        max_grad_norm=float(t.get("max_grad_norm") or 1.0),
        logging_steps=int(t.get("logging_steps") or 10),
        eval_strategy="steps" if t.get("eval_steps") else "no",
        eval_steps=int(t.get("eval_steps") or 500),
        save_steps=int(t.get("save_steps") or 500),
        save_total_limit=2,
        lr_scheduler_type=str(t.get("lr_scheduler_type") or "cosine"),
        gradient_checkpointing=bool(t.get("gradient_checkpointing")),
        report_to=report_to,
        seed=int(cfg.get("seed") or 7),
        bf16=mixed["bf16"],
        fp16=mixed["fp16"],
        remove_unused_columns=False,
        logging_first_step=True,
    )
    if max_steps > 0:
        kwargs["max_steps"] = max_steps
    cls = _training_args_class()
    sig = inspect.signature(cls.__init__)
    if "max_length" in sig.parameters:
        kwargs["max_length"] = int(cfg["max_seq_length"])
    elif "max_seq_length" in sig.parameters:
        kwargs["max_seq_length"] = int(cfg["max_seq_length"])
    if "dataset_text_field" in sig.parameters:
        kwargs["dataset_text_field"] = "text"
    if "packing" in sig.parameters:
        kwargs["packing"] = bool(t.get("packing", False))
    if "eval_strategy" not in sig.parameters and "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    # Drop kwargs the installed class does not accept.
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters or k == "output_dir"}
    # TrainingArguments often accepts via **kwargs; prefer full set when possible.
    try:
        return cls(**kwargs), output_dir, train_n
    except TypeError:
        return cls(**allowed), output_dir, train_n


def _make_trainer(model, tokenizer, train_ds, val_ds, args, peft_config, max_seq_length: int):
    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in sig.parameters:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sig.parameters:
        kwargs["max_seq_length"] = max_seq_length
    if "formatting_func" in sig.parameters and "dataset_text_field" not in sig.parameters:
        kwargs["formatting_func"] = lambda example: example["text"]
    return SFTTrainer(**kwargs)


def train(config_path: str) -> Path:
    cfg = load_yaml(config_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    ensure_pad_token(tokenizer)
    model = _load_model(cfg["model_name"], bool(cfg["training"].get("load_in_4bit")))
    if bool(cfg["training"].get("gradient_checkpointing")):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout") or 0.0),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
    )

    train_path = repo_path(cfg["data"]["train"])
    val_path = repo_path(cfg["data"]["val"])
    train_ds = _text_dataset(train_path, tokenizer)
    val_ds = _text_dataset(val_path, tokenizer)
    args, output_dir, train_n = _build_args(cfg, len(train_ds))
    trainer = _make_trainer(
        model,
        tokenizer,
        train_ds,
        val_ds,
        args,
        peft_config,
        int(cfg["max_seq_length"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    meta = {
        "base_model": cfg["model_name"],
        "output_dir": str(output_dir),
        "cuda": torch.cuda.is_available(),
        "train_rows": train_n,
        "lora": lora,
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return output_dir / "adapter"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune with LoRA/QLoRA via TRL")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    train(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
