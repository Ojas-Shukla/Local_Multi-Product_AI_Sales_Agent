"""Tests for usage tracking: schema, pricing, result parsing, reconciliation, quota."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from sellbird.brain import Brain
from sellbird.state import StateManager
from sellbird.integrations.quota import _parse_window, _token_from_json_blob


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        sm = StateManager(db_path)
        await sm.init_db()
        yield sm


# ── usage_events table + aggregations ──


@pytest.mark.asyncio
async def test_record_usage_event_and_totals(state):
    ok = await state.record_usage_event(
        agent="writer", task="write_sequence", session_id="sess-1",
        model="claude-opus-4-8", input_tokens=100, output_tokens=500,
        cache_read_tokens=2000, cache_creation_tokens=300,
        cost_usd=0.05, duration_ms=1200, num_turns=1,
    )
    assert ok

    totals = await state.usage_totals()
    assert totals["today"]["calls"] == 1
    assert totals["today"]["output_tokens"] == 500
    assert totals["today"]["cost_usd"] == pytest.approx(0.05)
    assert totals["week"]["input_tokens"] == 100


@pytest.mark.asyncio
async def test_usage_by_agent_groups_and_counts_sessions(state):
    # Two model-rows from the SAME call (same session) → one "call"
    await state.record_usage_event(
        agent="scout", session_id="sess-a", model="claude-opus-4-8",
        output_tokens=10, cost_usd=0.01,
    )
    await state.record_usage_event(
        agent="scout", session_id="sess-a", model="claude-haiku-4-5",
        output_tokens=5, cost_usd=0.001,
    )
    await state.record_usage_event(
        agent="handler", session_id="sess-b", model="claude-opus-4-8",
        output_tokens=20, cost_usd=0.02,
    )

    rows = await state.usage_by_agent(days=7)
    by_name = {r["agent"]: r for r in rows}
    assert by_name["scout"]["calls"] == 1
    assert by_name["scout"]["output_tokens"] == 15
    assert by_name["handler"]["calls"] == 1


@pytest.mark.asyncio
async def test_request_key_dedup(state):
    first = await state.record_usage_event(
        request_key="msg_1:req_1", model="claude-opus-4-8",
        output_tokens=50, source="transcript",
    )
    second = await state.record_usage_event(
        request_key="msg_1:req_1", model="claude-opus-4-8",
        output_tokens=50, source="transcript",
    )
    assert first is True
    assert second is False  # unique index makes reconciliation idempotent

    totals = await state.usage_totals()
    assert totals["today"]["output_tokens"] == 50


# ── Brain result-JSON parsing ──


SAMPLE_RESULT = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "duration_ms": 1428,
    "num_turns": 1,
    "result": "Hello from Claude",
    "session_id": "f362e409-925c-4eb1-ba98-60484bb2dcf5",
    "total_cost_usd": 0.0182948,
    "usage": {"input_tokens": 10, "output_tokens": 51},
    "modelUsage": {
        "claude-haiku-4-5-20251001": {
            "inputTokens": 10, "outputTokens": 51,
            "cacheReadInputTokens": 17198, "cacheCreationInputTokens": 8155,
            "costUSD": 0.0182948,
        }
    },
}


@pytest.mark.asyncio
async def test_brain_records_local_ollama_usage(state):
    brain = Brain(state)

    async def fake_generate(prompt, *, expect_json=False):
        return "Local Qwen response"

    brain.ollama.generate = fake_generate

    response = await brain.think(
        "Test local inference",
        agent="scout",
        task="score",
    )

    assert response == "Local Qwen response"
    assert await state.get_usage_today(provider="ollama") == 1

    rows = await state.usage_by_agent(days=1)
    assert len(rows) == 1
    assert rows[0]["agent"] == "scout"
    assert rows[0]["calls"] == 1

    models = await state.usage_by_model(days=1)
    assert models[0]["model"] == "qwen3:1.7b"

@pytest.mark.asyncio
async def test_usage_is_sellbird_only(state):
    """Sanity: the ledger only contains rows Sellbird's Brain wrote. There is
    no transcript scan to pull in other projects' Claude sessions."""
    import sellbird.usage as usage_mod
    assert not hasattr(usage_mod, "reconcile_transcripts")
    assert not hasattr(usage_mod, "parse_transcript_events")


# ── quota client helpers ──


def test_token_from_json_blob():
    blob = json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-abc"}})
    assert _token_from_json_blob(blob) == "sk-ant-oat01-abc"
    assert _token_from_json_blob("not json") is None
    assert _token_from_json_blob(json.dumps({"other": 1})) is None


def test_parse_window():
    parsed = _parse_window({"utilization": 42.5, "resets_at": "2026-08-29T01:00:00Z"})
    assert parsed == {"utilization": 42.5, "resets_at": "2026-08-29T01:00:00Z"}
    assert _parse_window({"utilization": "bad"}) is None
    assert _parse_window(None) is None
    assert _parse_window({"utilization": 0})["utilization"] == 0.0
