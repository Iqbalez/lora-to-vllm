"""Chat formatting that works for Qwen and for tiny GPT-2 smoke models."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerBase

FALLBACK_SYSTEM = "You are Nimbus Support. Answer with verified product facts only."


def render_chat(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    else:
        parts.append("")
    return "\n".join(parts)


def apply_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return render_chat(messages, add_generation_prompt=add_generation_prompt)


def prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop the gold assistant turn so the model has to generate it."""
    if messages and messages[-1]["role"] == "assistant":
        return messages[:-1]
    if not any(m["role"] == "system" for m in messages):
        return [{"role": "system", "content": FALLBACK_SYSTEM}, *messages]
    return messages


def assistant_only(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"]
    raise ValueError("No assistant message")


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
