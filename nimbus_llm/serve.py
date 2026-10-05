"""Production-shaped FastAPI wrapper around Transformers or vLLM."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from uvicorn import run as uvicorn_run

from nimbus_llm.config import load_yaml
from nimbus_llm.engines import Engine, build_engine

app = FastAPI(title="Nimbus Support LLM", version="0.1.0")
_engine: Engine | None = None
_settings: dict[str, Any] = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class Health(BaseModel):
    status: str
    backend: str
    model: str
    adapter: str


def _expected_key() -> str:
    return str(_settings.get("api_key") or os.environ.get("NIMBUS_API_KEY") or "")


def require_key(authorization: str | None = Header(default=None)) -> None:
    expected = _expected_key()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(
        status="ok",
        backend=str(_settings.get("backend")),
        model=str(_settings.get("model_name")),
        adapter=str(_settings.get("adapter_path") or ""),
    )


@app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
def chat(req: ChatRequest) -> dict[str, Any]:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    started = time.perf_counter()
    messages = [m.model_dump() for m in req.messages]
    max_tokens = req.max_tokens or int(_settings.get("max_new_tokens") or 256)
    temperature = req.temperature if req.temperature is not None else float(_settings.get("temperature") or 0.1)
    try:
        content = _engine.generate(messages, max_tokens=max_tokens, temperature=temperature)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"upstream generate failed: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "id": f"nimbus-{elapsed_ms}",
        "object": "chat.completion",
        "model": req.model or _settings.get("model_name"),
        "backend": _settings.get("backend"),
        "latency_ms": elapsed_ms,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }


def configure(settings: dict[str, Any], engine: Engine | None = None) -> FastAPI:
    global _engine, _settings
    _settings = dict(settings)
    _engine = engine or build_engine(
        backend=str(settings.get("backend") or "transformers"),
        model_name=str(settings.get("model_name")),
        adapter_path=str(settings.get("adapter_path") or "") or None,
        vllm_base_url=str(settings.get("vllm_base_url") or "http://127.0.0.1:8001/v1"),
    )
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Nimbus via FastAPI")
    parser.add_argument("--config", default="configs/serve.yaml")
    parser.add_argument("--backend", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--vllm-url", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    settings = load_yaml(args.config)
    if args.backend:
        settings["backend"] = args.backend
    if args.model:
        settings["model_name"] = args.model
    if args.adapter:
        settings["adapter_path"] = args.adapter
    if args.vllm_url:
        settings["vllm_base_url"] = args.vllm_url
    host = args.host or str(settings.get("host") or "0.0.0.0")
    port = args.port or int(settings.get("port") or 8000)
    settings["backend"] = os.environ.get("NIMBUS_BACKEND", settings.get("backend"))
    settings["model_name"] = os.environ.get("NIMBUS_MODEL", settings.get("model_name"))
    settings["adapter_path"] = os.environ.get("NIMBUS_ADAPTER", settings.get("adapter_path") or "")
    configure(settings)
    uvicorn_run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
