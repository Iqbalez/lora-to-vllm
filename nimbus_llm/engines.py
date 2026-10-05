"""Generation backends: in-process Transformers (dev) and remote vLLM (prod)."""

from __future__ import annotations

from typing import Any, Protocol

import httpx


class Engine(Protocol):
    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str: ...


class TransformersEngine:
    def __init__(self, model_name: str, adapter_path: str | None = None):
        from nimbus_llm.evaluate import generate_one, load_causal

        self.tokenizer, self.model = load_causal(model_name, adapter_path=adapter_path)
        self._generate_one = generate_one

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        del temperature  # greedy on the transformers path for reproducibility
        return self._generate_one(self.tokenizer, self.model, messages, max_new_tokens=max_tokens)


class VLLMEngine:
    """Talks to a vLLM OpenAI-compatible server (Docker or TGI-style)."""

    def __init__(self, base_url: str, model_name: str, api_key: str = "not-needed"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]


class ScriptedEngine:
    """Deterministic engine for API unit tests. Not used in production."""

    def __init__(self, reply: str = "The Pro plan is limited to 600 requests per minute with a burst of 1200."):
        self.reply = reply

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        del messages, max_tokens, temperature
        return self.reply


def build_engine(backend: str, model_name: str, adapter_path: str | None, vllm_base_url: str) -> Any:
    backend = backend.lower()
    if backend == "transformers":
        return TransformersEngine(model_name, adapter_path=adapter_path or None)
    if backend == "vllm":
        return VLLMEngine(vllm_base_url, model_name)
    if backend == "scripted":
        return ScriptedEngine()
    raise ValueError(f"Unknown backend {backend}. Use transformers, vllm, or scripted.")
