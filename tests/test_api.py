from fastapi.testclient import TestClient

from nimbus_llm.engines import ScriptedEngine
from nimbus_llm.serve import configure


def test_health_and_chat_with_scripted_engine():
    app = configure(
        {
            "backend": "scripted",
            "model_name": "scripted",
            "adapter_path": "",
            "api_key": "test-key",
            "max_new_tokens": 32,
            "temperature": 0.0,
            "vllm_base_url": "http://127.0.0.1:9/v1",
        },
        engine=ScriptedEngine("The Pro plan is limited to 600 requests per minute with a burst of 1200."),
    )
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    denied = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "rate limit?"}]},
    )
    assert denied.status_code == 401

    ok = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={"messages": [{"role": "user", "content": "rate limit?"}]},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert "600" in body["choices"][0]["message"]["content"]
    assert body["object"] == "chat.completion"
    assert "latency_ms" in body
