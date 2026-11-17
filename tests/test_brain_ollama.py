import pytest
from pydantic import ValidationError

from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.local_ai.ollama_client import OllamaModelNotFound
from sellbird.state import StateManager


class FakeState:
    def __init__(self):
        self.calls = 0
        self.events = []
        self.settings = {}

    async def increment_usage(self, provider):
        assert provider == "ollama"
        self.calls += 1

    async def record_usage_event(self, **kwargs):
        self.events.append(kwargs)

    async def set_setting(self, key, value):
        self.settings[key] = value

    async def get_setting(self, key):
        return self.settings.get(key, "")


class FakeOllama:
    def __init__(self, answer="local qwen response"):
        self.settings = type("Settings", (), {
            "base_url": "http://127.0.0.1:11434", "model": "qwen3:1.7b"
        })()
        self.answer = answer
        self.requests = []

    async def generate(self, prompt, *, expect_json=False):
        self.requests.append((prompt, expect_json))
        return self.answer

    async def require_model(self):
        return None


class MissingModel(FakeOllama):
    async def require_model(self):
        raise OllamaModelNotFound("Model 'qwen3:1.7b' is not installed. Run: ollama pull qwen3:1.7b")


@pytest.mark.asyncio
async def test_brain_routes_generation_to_local_ollama_and_records_usage():
    state = FakeState()
    brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b"))
    local = FakeOllama()
    brain.ollama = local

    answer = await brain.think("Write a short business summary", agent="analyst", task="profile")

    assert answer == "local qwen response"
    assert local.requests == [("Write a short business summary", False)]
    assert state.calls == 1
    assert state.events[0]["provider"] == "ollama"
    assert state.events[0]["model"] == "qwen3:1.7b"
    assert state.events[0]["source"] == "ollama_local"


def test_brain_config_rejects_cloud_llm_providers():
    for provider in ("claude", "codex", "openai", "gemini"):
        with pytest.raises(ValidationError):
            BrainConfig(provider=provider)


@pytest.mark.asyncio
async def test_brain_preflight_checks_local_model_only():
    brain = Brain(FakeState(), BrainConfig(provider="ollama", model="qwen3:1.7b"))
    local = FakeOllama()
    brain.ollama = local

    assert await brain.preflight() == ""


@pytest.mark.asyncio
async def test_missing_local_model_pauses_brain_without_cloud_fallback():
    state = FakeState()
    brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b"))
    brain.ollama = MissingModel()

    problem = await brain.preflight()

    assert "ollama pull qwen3:1.7b" in problem
    assert '"status": "paused"' in state.settings["brain_status"]


@pytest.mark.asyncio
async def test_ollama_usage_counter_migration(tmp_path):
    state = StateManager(str(tmp_path / "phase2.sqlite3"))
    await state.init_db()

    await state.increment_usage(provider="ollama")

    assert await state.get_usage_today(provider="ollama") == 1
