import pytest

from sellbird.state import StateManager


@pytest.mark.asyncio
async def test_local_ai_queue_persists_and_updates(tmp_path):
    state = StateManager(str(tmp_path / "queue.sqlite3"))
    await state.init_db()

    task_id = await state.enqueue_local_ai_task(
        "Summarize this company", agent="analyst", task="profile", expect_json=True
    )
    pending = await state.get_pending_local_ai_tasks()
    assert len(pending) == 1
    assert pending[0]["id"] == task_id
    assert pending[0]["prompt"] == "Summarize this company"
    assert pending[0]["expect_json"] == 1

    await state.update_local_ai_task(task_id, status="processing")
    await state.update_local_ai_task(task_id, status="completed", result_text='{"summary":"ok"}')
    assert await state.get_pending_local_ai_tasks() == []

    async with state._connect() as db:
        async with db.execute(
            "SELECT status, attempts, result_text FROM local_ai_task_queue WHERE id = ?", (task_id,)
        ) as cursor:
            row = await cursor.fetchone()
    assert tuple(row) == ("completed", 1, '{"summary":"ok"}')


@pytest.mark.asyncio
async def test_queue_rejects_invalid_status(tmp_path):
    state = StateManager(str(tmp_path / "queue.sqlite3"))
    await state.init_db()
    task_id = await state.enqueue_local_ai_task("test")
    with pytest.raises(ValueError, match="invalid local AI task status"):
        await state.update_local_ai_task(task_id, status="cloud_retry")


@pytest.mark.asyncio
async def test_brain_queues_on_outage_and_replays_after_ollama_returns(tmp_path):
    from sellbird.brain import Brain
    from sellbird.config import BrainConfig
    from sellbird.local_ai.ollama_client import OllamaUnavailable

    state = StateManager(str(tmp_path / "brain-queue.sqlite3"))
    await state.init_db()
    brain = Brain(state, BrainConfig(provider="ollama", model="qwen3:1.7b"))

    class RecoveringOllama:
        def __init__(self):
            self.settings = type("Settings", (), {
                "base_url": "http://127.0.0.1:11434", "model": "qwen3:1.7b"
            })()
            self.online = False

        async def generate(self, prompt, *, expect_json=False):
            if not self.online:
                raise OllamaUnavailable("Ollama offline")
            return "replayed local answer"

        async def require_model(self):
            if not self.online:
                raise OllamaUnavailable("Ollama offline")

    local = RecoveringOllama()
    brain.ollama = local
    answer = await brain.think("Do this later", agent="test", task="queued")
    assert answer == ""
    queued = await state.get_pending_local_ai_tasks()
    assert len(queued) == 1
    assert queued[0]["prompt"] == "Do this later"

    local.online = True
    result = await brain.process_queued_tasks()
    assert result["completed"] == 1
    assert result["pending"] == 0

    async with state._connect() as db:
        async with db.execute(
            "SELECT status, result_text FROM local_ai_task_queue WHERE id = ?", (queued[0]["id"],)
        ) as cursor:
            row = await cursor.fetchone()
    assert tuple(row) == ("completed", "replayed local answer")
