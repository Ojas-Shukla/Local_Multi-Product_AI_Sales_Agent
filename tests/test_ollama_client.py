import httpx
import pytest

from sellbird.local_ai.ollama_client import (
    OllamaClient,
    OllamaModelNotFound,
    OllamaSettings,
    OllamaUnavailable,
)


@pytest.mark.asyncio
async def test_generate_uses_local_ollama_and_returns_content():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["payload"] = request.read().decode()
        return httpx.Response(200, json={"message": {"content": "Local answer"}})

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    client = OllamaClient(
        OllamaSettings(base_url="http://127.0.0.1:11434", model="qwen3:1.7b"),
        http_client=http,
    )

    try:
        answer = await client.generate("Explain this", system="You are local", expect_json=True)
    finally:
        await http.aclose()

    assert answer == "Local answer"
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert '"model":"qwen3:1.7b"' in seen["payload"]
    assert '"format":"json"' in seen["payload"]


@pytest.mark.asyncio
async def test_missing_model_has_pull_instruction():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "other:latest"}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OllamaClient(
        OllamaSettings(base_url="http://127.0.0.1:11434", model="qwen3:1.7b"),
        http_client=http,
    )
    try:
        with pytest.raises(OllamaModelNotFound, match="ollama pull qwen3:1.7b"):
            await client.require_model()
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_server_unavailable_does_not_fallback():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OllamaClient(
        OllamaSettings(base_url="http://127.0.0.1:11434", model="qwen3:1.7b"),
        http_client=http,
    )
    try:
        with pytest.raises(OllamaUnavailable, match="will not switch to a cloud model"):
            await client.generate("test")
    finally:
        await http.aclose()


def test_settings_reject_invalid_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "zero")
    with pytest.raises(ValueError, match="must be a number"):
        OllamaSettings.from_env()
