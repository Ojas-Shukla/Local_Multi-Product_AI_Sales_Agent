"""End-to-end: the real heartbeat, agents, and Brain, using a local Ollama
fixture instead of requiring a running model server.

Everything outside Sellbird is a fixture: a disposable database and config,
a capturing mail transport (nothing is ever sent over the network), a
stubbed Scout and profiler (they would reach the internet), and a mocked
Ollama model. These tests check what an operator would see across whole
heartbeat cycles: what is queued or sent, what is consumed, what is counted,
and what happens when local inference can't be used.
"""

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

import sellbird.brain as brain_mod
import sellbird.main as main_mod
from sellbird.config import EnvConfig, SellbirdConfig
from sellbird.integrations.mail_provider import InboundMessage
from sellbird.models.conversation import Conversation, Message
from sellbird.local_ai.ollama_client import OllamaUnavailable
from sellbird.state import StateManager
from tests.test_outbox_native import FakeProvider, _now_iso, seed_prospect

FIXTURES = Path(__file__).with_name("fixtures")

BASE = {
    "persona": {"name": "Sellbird", "company": "Acme Tools", "role": "BD",
                "email": "sellbird@acme.example", "linkedin": "", "tone": "direct"},
    "product": {"name": "Widget", "description": "Widgets for teams", "pricing": "$10",
                "key_benefits": ["fast"], "objection_responses": {}},
    "icp": {"industries": ["software"], "company_size": "10-50", "titles": ["VP"],
            "geography": ["US"]},
    "channels": {"email": {"provider": "smtp", "require_approval": True}},
}

class Harness:
    """One disposable Sellbird install wired to local inference fixtures."""

    def __init__(self, monkeypatch, tmp_path, provider="ollama", model="qwen3:1.7b", db=None):
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.scenario = "smart"
        self.model_calls = 0
        self.fail_call = None
        self.state = StateManager(str(db or tmp_path / "sellbird.db"))
        self.mail = FakeProvider()
        self.cycles = 0
        self.stop_after = 1
        self.between = {}
        self.scout_runs = 0
        self.set_config(provider, model)
        # Gmail integration tests use a mocked local Ollama installation.
        # Keep Brain.preflight() active, but don't require a real Ollama server.
        if provider == "ollama":
            from sellbird.local_ai.ollama_client import (
                OllamaClient,
                OllamaError,
                OllamaModelNotFound,
            )

            async def mock_require_model(client):
                if self.scenario == "server_down":
                    raise OllamaUnavailable("Cannot connect to local Ollama server")
                if client.settings.model == "missing-model":
                    raise OllamaModelNotFound(
                        f"Model {client.settings.model!r} is not installed"
                    )

            async def mock_generate(
                client,
                prompt,
                *,
                system=None,
                temperature=0.2,
                expect_json=False,
            ):
                self.model_calls += 1
                if self.fail_call == self.model_calls:
                    raise OllamaUnavailable("Synthetic local Ollama outage")
                if self.scenario == "server_down":
                    raise OllamaUnavailable("Cannot connect to local Ollama server")
                prompt_lower = prompt.lower()

                # Simulate an empty model response for the human-escalation test.
                if self.scenario == "commentary_only":
                    if "classify this email reply" in prompt_lower:
                        return "interested"
                    raise OllamaError("Ollama returned an empty model response")

                # Reply classification.
                if "classify this email reply" in prompt_lower:
                    return "interested"

                # Writer: three-email sequence.
                if expect_json and (
                    "3-email cold outreach sequence" in prompt_lower
                    or "return only a json array" in prompt_lower
                ):
                    return json.dumps([
                        {
                            "step": 1,
                            "subject": "Improving team workflows",
                            "body": (
                                "Hi Jane,\\n\\n"
                                "I noticed your work at Acme Tools. "
                                "Would improving team workflows be a priority?\\n\\n"
                                "Best,\\nSellbird"
                            ),
                            "delay_days": 0,
                        },
                        {
                            "step": 2,
                            "subject": "Following up",
                            "body": (
                                "Hi Jane,\\n\\n"
                                "I wanted to follow up on my earlier note. "
                                "Would a short conversation be useful?\\n\\n"
                                "Best,\\nSellbird"
                            ),
                            "delay_days": 3,
                        },
                        {
                            "step": 3,
                            "subject": "One last note",
                            "body": (
                                "Hi Jane,\\n\\n"
                                "Would it be worth revisiting this later?\\n\\n"
                                "Best,\\nSellbird"
                            ),
                            "delay_days": 4,
                        },
                    ])

                # Writer: individual personalised email.
                if expect_json:
                    return json.dumps({
                        "subject": "Improving team workflows",
                        "body": (
                            "Hi Jane,\\n\\n"
                            "I noticed your work at Acme Tools. "
                            "Would improving team workflows be a priority?"
                        ),
                    })

                # Handler: draft a reply to an interested prospect.
                return (
                    "Happy to help. Our pricing starts at $10. "
                    "Would you like me to share the details?"
                )

            monkeypatch.setattr(
                OllamaClient,
                "require_model",
                mock_require_model,
            )
            monkeypatch.setattr(
                OllamaClient,
                "generate",
                mock_generate,
            )

        import sellbird.agents.handler as handler_mod
        import sellbird.agents.scout as scout_mod
        import sellbird.agents.sender as sender_mod

        monkeypatch.setattr(handler_mod, "get_mail_provider", lambda c, e: self.mail)
        monkeypatch.setattr(sender_mod, "get_mail_provider", lambda c, e: self.mail)

        async def scout_run(_self):
            self.scout_runs += 1

        async def no_profile(state, limit=25, **_):
            return None

        monkeypatch.setattr(scout_mod.Scout, "run", scout_run)
        monkeypatch.setattr(main_mod, "run_profile_stage", no_profile)
        monkeypatch.setattr(main_mod, "load_config", lambda: self.config)
        monkeypatch.setattr(main_mod, "load_env", lambda: EnvConfig())
        monkeypatch.setattr(main_mod, "StateManager", lambda: self.state)
        monkeypatch.setattr(main_mod, "in_quiet_hours", lambda config: False)

        async def one_cycle_sleep(seconds, stop_event):
            self.cycles += 1
            step = self.between.get(self.cycles)
            if step is not None:
                result = step()
                if asyncio.iscoroutine(result):
                    await result
            return self.cycles >= self.stop_after

        monkeypatch.setattr(main_mod, "_interruptible_sleep", one_cycle_sleep)
        # Sender: no inter-send pacing sleeps.
        real_init = sender_mod.Sender.__init__

        def init_no_pacing(s, *a, **k):
            real_init(s, *a, **k)
            s.send_pacing = False

        monkeypatch.setattr(sender_mod.Sender, "__init__", init_no_pacing)

    def set_config(self, provider, model=""):
        brain = {"provider": provider}
        if model:
            brain["model"] = model
        self.config = SellbirdConfig(**BASE, brain=brain)

    async def run_cycles(self, n=1, between=None):
        """Run one heartbeat (one Sellbird process) for ``n`` cycles.
        ``between[k]`` runs after cycle k, inside the same process."""
        self.cycles = 0
        self.stop_after = n
        self.between = between or {}
        await asyncio.wait_for(main_mod.heartbeat(asyncio.Event()), 120)

    async def status(self):
        return brain_mod.parse_brain_status(await self.state.get_setting(brain_mod.BRAIN_STATUS_KEY))

    def fail_from_call(self, n):
        """Fail the selected model request with a local Ollama outage."""
        self.fail_call = n

    def recover(self):
        """Restore the local Ollama fixture for the next heartbeat cycle."""
        self.scenario = "smart"
        self.fail_call = None


@pytest.fixture
def sellbird(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


async def _open_thread(state, email="jane@acme.com"):
    pid = await seed_prospect(state, email=email, status="contacted")
    convo = Conversation(id="", prospect_id=pid, channel="email",
                         thread=[Message(sender="sellbird", content="Hi Jane, quick question?")])
    convo.id = await state.add_conversation(convo)
    return pid, convo.id


def _reply(body, provider_id="in-1", email="jane@acme.com"):
    return InboundMessage(provider_id=provider_id, from_email=email,
                          subject="Re: quick question", body=body, message_id=f"<{provider_id}@x>")


async def _approved_item(state, email="bob@beta.com"):
    pid = await seed_prospect(state, email=email, status="contacted")
    return await state.add_outbox_item(
        prospect_id=pid, to_email=email, subject="following up",
        body="Hi Bob, any thoughts on this?", send_at=_now_iso(), status="approved",
        campaign_id="c-1", step=2,
    )


# ── healthy operation ──


@pytest.mark.asyncio
async def test_reply_is_classified_answered_for_review_and_never_reprocessed(sellbird):
    await sellbird.state.init_db()
    pid, convo_id = await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Thanks. Could you share pricing details?")]

    await sellbird.run_cycles(2)  # the second cycle sees the same inbound message

    convo = await sellbird.state.get_conversation(convo_id)
    assert convo.intent == "interested"
    assert [m.sender for m in convo.thread] == ["sellbird", "prospect"]
    replies = await sellbird.state.get_outbox(status="pending_review")
    assert len(replies) == 1 and replies[0]["kind"] == "reply"
    assert replies[0]["body"].startswith("Happy to help")
    assert sellbird.mail.sent == []  # approval required: nothing leaves
    assert await sellbird.state.is_reply_processed("in-1")
    # Classification and reply drafting are both attributed to local Ollama.
    assert sellbird.model_calls == 2
    assert await sellbird.state.get_usage_today("ollama") == 2
    rows = await sellbird.state.usage_by_provider(days=1)
    assert [(r["provider"], r["calls"]) for r in rows] == [("ollama", 2)]
    assert (await sellbird.status()) == {}  # never paused


@pytest.mark.asyncio
async def test_approved_email_is_delivered_once_across_cycles(sellbird):
    await sellbird.state.init_db()
    await _approved_item(sellbird.state)
    await sellbird.run_cycles(3)
    assert [m["to"] for m in sellbird.mail.sent] == ["bob@beta.com"]
    assert len(await sellbird.state.get_outbox(status="sent")) == 1
    assert await sellbird.state.get_usage_today("ollama") == 0  # sending needs no model


@pytest.mark.asyncio
async def test_writer_drafts_through_ollama_and_waits_for_approval(sellbird):
    await sellbird.state.init_db()
    await seed_prospect(sellbird.state, email="jane@acme.com", status="new")
    # One cycle: the writer drafts, then (send phase, after it) the sender
    # stages the drafts for approval. Nothing is sent.
    await sellbird.run_cycles(1)
    drafts = await sellbird.state.get_outbox(status="pending_review")
    assert drafts and all(d["to_email"] == "jane@acme.com" for d in drafts)
    assert sellbird.mail.sent == []
    assert await sellbird.state.get_usage_today("ollama") >= 1
    assert sellbird.scout_runs == 0


# ── the provider can't be used: pause, touch nothing, recover ──


@pytest.mark.asyncio
async def test_unavailable_ollama_pauses_everything_and_recovers(sellbird):
    await sellbird.state.init_db()
    _, convo_id = await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    await _approved_item(sellbird.state)
    sellbird.scenario = "server_down"

    await sellbird.run_cycles(2)
    # Preflight keeps mailbox handling and sending paused while Ollama is down.
    assert not await sellbird.state.is_reply_processed("in-1")
    assert sellbird.mail.sent == []
    assert sellbird.model_calls == 0
    status = await sellbird.status()
    assert status["status"] == "paused" and "Ollama" in status["reason"]
    assert len((await sellbird.state.get_conversation(convo_id)).thread) == 1

    sellbird.recover()
    await sellbird.run_cycles(1)
    assert await sellbird.state.is_reply_processed("in-1")
    assert [m["to"] for m in sellbird.mail.sent] == ["bob@beta.com"]
    assert (await sellbird.status())["status"] == "ok"


@pytest.mark.asyncio
async def test_restored_ollama_is_picked_up_without_a_restart(sellbird):
    await sellbird.state.init_db()
    sellbird.scenario = "server_down"
    await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    seen = {}

    async def still_paused():
        seen["processed"] = await sellbird.state.is_reply_processed("in-1")
        seen["calls"] = sellbird.model_calls

    def restore_ollama():
        sellbird.recover()

    await sellbird.run_cycles(3, between={1: still_paused, 2: restore_ollama})
    assert seen == {"processed": False, "calls": 0}
    assert await sellbird.state.is_reply_processed("in-1")
    assert (await sellbird.status())["status"] == "ok"


@pytest.mark.asyncio
async def test_ollama_outage_mid_cycle_leaves_replies_for_later(sellbird):
    await sellbird.state.init_db()
    _, convo_id = await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?"),
                           _reply("Another question", provider_id="in-2")]
    sellbird.fail_from_call(1)

    async def check_paused():
        assert not await sellbird.state.is_reply_processed("in-1")
        assert not await sellbird.state.is_reply_processed("in-2")
        convo = await sellbird.state.get_conversation(convo_id)
        assert convo.status == "open" and len(convo.thread) == 1  # not escalated
        assert await sellbird.state.get_outbox() == []
        assert (await sellbird.status())["status"] == "paused"

    async def pause_then_keep_down():
        await check_paused()
        sellbird.scenario = "server_down"

    await sellbird.run_cycles(3, between={
        1: pause_then_keep_down,
        2: sellbird.recover,
    })
    # Cycle 2 stays paused; cycle 3 handles both replies exactly once.
    assert await sellbird.state.is_reply_processed("in-1")
    assert await sellbird.state.is_reply_processed("in-2")
    assert (await sellbird.status())["status"] == "ok"


@pytest.mark.asyncio
async def test_rejected_model_pauses_with_an_actionable_reason(monkeypatch, tmp_path):
    sellbird = Harness(monkeypatch, tmp_path, model="missing-model")  # absent from catalog
    await sellbird.state.init_db()
    await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    await sellbird.run_cycles(1)
    status = await sellbird.status()
    assert status["status"] == "paused" and "missing-model" in status["reason"]
    assert not await sellbird.state.is_reply_processed("in-1")
    assert sellbird.model_calls == 0


@pytest.mark.asyncio
async def test_ollama_outage_holds_reply_until_service_returns(sellbird):
    await sellbird.state.init_db()
    await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    sellbird.scenario = "server_down"
    paused = {}

    async def after_first():
        paused["calls"] = sellbird.model_calls
        paused["processed"] = await sellbird.state.is_reply_processed("in-1")
        sellbird.recover()

    await sellbird.run_cycles(2, between={1: after_first})
    assert paused == {"calls": 0, "processed": False}
    assert await sellbird.state.is_reply_processed("in-1")


@pytest.mark.asyncio
async def test_restart_clears_a_stale_paused_status(sellbird):
    """A pause recorded before a restart must not keep showing once healthy."""
    await sellbird.state.init_db()
    await sellbird.state.set_setting(brain_mod.BRAIN_STATUS_KEY, json.dumps(
        {"provider": "ollama", "status": "paused", "reason": "old", "since": "x"}))
    await sellbird.run_cycles(1)
    assert (await sellbird.status())["status"] == "ok"


@pytest.mark.asyncio
async def test_transient_failure_skips_a_cycle_then_recovers(sellbird):
    await sellbird.state.init_db()
    await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    sellbird.fail_from_call(1)
    await sellbird.run_cycles(1)
    assert not await sellbird.state.is_reply_processed("in-1")
    assert (await sellbird.status())["status"] == "paused"

    sellbird.recover()
    await sellbird.run_cycles(1)
    assert await sellbird.state.is_reply_processed("in-1")
    assert len(await sellbird.state.get_outbox(status="pending_review")) == 1


@pytest.mark.asyncio
async def test_empty_ollama_response_leaves_reply_for_retry(sellbird):
    """An empty Ollama response must not consume or answer the inbound reply."""
    await sellbird.state.init_db()
    _, convo_id = await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    sellbird.scenario = "commentary_only"
    await sellbird.run_cycles(1)
    convo = await sellbird.state.get_conversation(convo_id)
    assert convo.status == "open"
    assert [message.sender for message in convo.thread] == ["sellbird"]
    assert await sellbird.state.get_outbox() == [] and sellbird.mail.sent == []
    assert not await sellbird.state.is_reply_processed("in-1")
    assert await sellbird.state.get_usage_today("ollama") == 1  # only classification succeeded
    assert (await sellbird.status())["status"] == "paused"


# ── existing installs and local provider ──


def _load_pre_codex_db(path: Path):
    """A database written by Sellbird 0.2.1 (before the Codex provider)."""
    con = sqlite3.connect(path)
    con.executescript((FIXTURES / "pre_codex_harvey_0_2_1.sql").read_text())
    con.close()


@pytest.mark.asyncio
async def test_pre_codex_database_migrates_and_runs_on_ollama(monkeypatch, tmp_path):
    db = tmp_path / "old.db"
    _load_pre_codex_db(db)
    sellbird = Harness(
        monkeypatch,
        tmp_path,
        provider="ollama",
        model="qwen3:1.7b",
        db=db,
    )
    await sellbird.state.init_db()

    rows = await sellbird.state.usage_by_provider(days=36500)
    assert [(r["provider"], r["calls"]) for r in rows] == [("claude", 1)]
    old_prospect = await sellbird.state.get_prospect_by_email("old@legacy.com")
    assert old_prospect is not None and old_prospect.status == "contacted"

    _, convo_id = await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    await sellbird.run_cycles(1)
    assert await sellbird.state.is_reply_processed("in-1")
    rows = {r["provider"]: r["calls"] for r in await sellbird.state.usage_by_provider(days=36500)}
    assert rows == {"claude": 1, "ollama": 2}


@pytest.mark.asyncio
async def test_fixture_transport_is_the_only_mail_path(sellbird):
    """Guard for the tests above: no real SMTP/IMAP/API client is built."""
    await sellbird.state.init_db()
    import smtplib
    import imaplib

    def refuse(*a, **k):
        raise AssertionError("a real mail connection was attempted")

    sellbird.monkeypatch.setattr(smtplib, "SMTP", refuse)
    sellbird.monkeypatch.setattr(smtplib, "SMTP_SSL", refuse)
    sellbird.monkeypatch.setattr(imaplib, "IMAP4_SSL", refuse)
    await _approved_item(sellbird.state)
    await sellbird.run_cycles(1)
    assert len(sellbird.mail.sent) == 1


class _DownBrain:
    """A brain whose local provider just became unavailable mid-cycle."""
    paused_reason = "Cannot connect to local Ollama server."

    def load_skills_for_agent(self, *_):
        return ""

    def load_prompt(self, *_a, **_k):
        return ""

    async def think(self, *a, **k):
        return ""


@pytest.mark.asyncio
async def test_instantly_path_leaves_the_reply_unprocessed_when_the_brain_is_down(tmp_path):
    """Legacy Instantly replies get the same protection as native ones."""
    from sellbird.agents.handler import Handler
    from sellbird.brain import BrainUnavailable
    from sellbird.models.campaign import Campaign

    state = StateManager(str(tmp_path / "i.db"))
    await state.init_db()
    _, convo_id = await _open_thread(state)
    config = SellbirdConfig(**{**BASE, "channels": {"email": {"provider": "instantly"}}})
    handler = Handler(brain=_DownBrain(), state=state, config=config, env=EnvConfig())
    campaign = Campaign(id="c1", name="c", channel="email", sequence=[], prospect_ids=[],
                        status="active", instantly_campaign_id="ic-1")

    class Instantly:
        async def get_replies(self, campaign_id):
            return [{"uuid": "r-1", "lead_email": "jane@acme.com",
                     "body": "Could you share pricing details?"}]

        async def send_reply(self, *a):
            raise AssertionError("must not reply")

    handler.instantly = Instantly()
    with pytest.raises(BrainUnavailable):
        await handler._handle_reply({"uuid": "r-1", "lead_email": "jane@acme.com",
                                     "body": "Could you share pricing details?"}, campaign)
    assert not await state.is_reply_processed("r-1")
    convo = await state.get_conversation(convo_id)
    assert convo.status == "open" and len(convo.thread) == 1

    async def active(status):
        return [campaign]

    state.get_campaigns_by_status = active
    await handler.run()  # the loop stops cleanly instead of escalating
    assert not await state.is_reply_processed("r-1")


# ── outages part-way through a reply, and sends waiting on replies ──


@pytest.mark.asyncio
async def test_outage_while_drafting_the_reply_loses_nothing_and_duplicates_nothing(sellbird):
    """Classification succeeds, the provider fails before the reply draft:
    nothing about the reply may be stored, and the retry after recovery
    must leave exactly one inbound message, one stage change, one reply."""
    await sellbird.state.init_db()
    _, convo_id = await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    sellbird.fail_from_call(2)
    seen = {}

    async def after_outage():
        convo = await sellbird.state.get_conversation(convo_id)
        seen.update(processed=await sellbird.state.is_reply_processed("in-1"),
                    thread=[m.sender for m in convo.thread], stage=convo.stage,
                    status=convo.status, outbox=len(await sellbird.state.get_outbox()),
                    brain=(await sellbird.status()).get("kind"))
        sellbird.recover()

    await sellbird.run_cycles(3, between={1: after_outage})
    assert seen == {"processed": False, "thread": ["sellbird"], "stage": "initial_outreach",
                    "status": "open", "outbox": 0, "brain": "setup"}
    convo = await sellbird.state.get_conversation(convo_id)
    assert [m.sender for m in convo.thread] == ["sellbird", "prospect"]
    assert convo.stage == "engaged" and convo.intent == "interested"
    replies = await sellbird.state.get_outbox(status="pending_review")
    assert len(replies) == 1 and replies[0]["kind"] == "reply"
    assert await sellbird.state.is_reply_processed("in-1")
    assert (await sellbird.status())["status"] == "ok"


@pytest.mark.asyncio
async def test_instantly_reply_survives_an_outage_while_drafting(sellbird):
    """Legacy Instantly path: same guarantee, and the reply is sent once."""
    from sellbird.agents.handler import Handler
    from sellbird.brain import Brain
    from sellbird.models.campaign import Campaign

    await sellbird.state.init_db()
    _, convo_id = await _open_thread(sellbird.state)
    config = SellbirdConfig(**{**BASE, "channels": {"email": {
                                "provider": "instantly", "require_approval": False}},
                             "brain": {"provider": "ollama", "model": "qwen3:1.7b"}})
    brain = Brain(sellbird.state, config.brain)
    handler = Handler(brain=brain, state=sellbird.state, config=config, env=EnvConfig())
    campaign = Campaign(id="c1", name="c", channel="email", sequence=[], prospect_ids=[],
                        status="active", instantly_campaign_id="ic-1")
    sent = []

    class Instantly:
        async def get_replies(self, campaign_id):
            return [{"uuid": "r-1", "lead_email": "jane@acme.com",
                     "body": "Could you share pricing details?"}]

        async def send_reply(self, reply_uuid, body):
            sent.append((reply_uuid, body))
            return {"ok": True}

    async def active(status):
        return [campaign]

    handler.instantly = Instantly()
    sellbird.state.get_campaigns_by_status = active
    sellbird.fail_from_call(2)
    assert await brain.preflight() == ""
    await handler.run()
    convo = await sellbird.state.get_conversation(convo_id)
    assert (sent, len(convo.thread), convo.stage) == ([], 1, "initial_outreach")
    assert not await sellbird.state.is_reply_processed("r-1")

    sellbird.recover()
    assert await brain.preflight() == ""
    await handler.run()
    await handler.run()  # already processed: no second reply
    convo = await sellbird.state.get_conversation(convo_id)
    assert [m.sender for m in convo.thread] == ["sellbird", "prospect", "sellbird"]
    assert convo.stage == "engaged" and len(sent) == 1 and sent[0][0] == "r-1"


@pytest.mark.asyncio
async def test_rejected_model_never_sends_queued_mail(monkeypatch, tmp_path):
    sellbird = Harness(monkeypatch, tmp_path, model="missing-model")  # absent from catalog
    await sellbird.state.init_db()
    await _open_thread(sellbird.state)
    await _approved_item(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    await sellbird.run_cycles(1)
    assert sellbird.mail.sent == []
    status = await sellbird.status()
    assert status["status"] == "paused" and status["kind"] == "setup"
    assert "missing-model" in status["reason"]
    assert not await sellbird.state.is_reply_processed("in-1")
    assert sellbird.model_calls == 0
    # Caught by the heartbeat preflight, before any agent ran: the mailbox
    # wasn't even read (the prospect isn't marked as having replied).
    jane = await sellbird.state.get_prospect_by_email("jane@acme.com")
    assert jane.status == "contacted"


@pytest.mark.asyncio
async def test_outage_during_the_handler_holds_queued_sends_until_recovery(sellbird):
    await sellbird.state.init_db()
    await _open_thread(sellbird.state)
    await _approved_item(sellbird.state)  # due to bob
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    sellbird.fail_from_call(1)  # preflight passes; the classification call fails
    seen = {}

    async def after_outage():
        seen["sent"] = list(sellbird.mail.sent)
        seen["processed"] = await sellbird.state.is_reply_processed("in-1")
        sellbird.recover()

    await sellbird.run_cycles(3, between={1: after_outage})
    assert seen == {"sent": [], "processed": False}
    assert [m["to"] for m in sellbird.mail.sent] == ["bob@beta.com"]  # once, after recovery
    assert await sellbird.state.is_reply_processed("in-1")


@pytest.mark.asyncio
async def test_a_reply_cancels_its_due_follow_up_before_the_send_phase(sellbird):
    """Stop-on-reply beats the outbox drain in the same cycle, even when the
    reply starts a new conversation (none was open before)."""
    await sellbird.state.init_db()
    pid = await seed_prospect(sellbird.state, email="jane@acme.com", status="contacted")
    await sellbird.state.add_outbox_item(
        prospect_id=pid, to_email="jane@acme.com", subject="following up",
        body="Hi Jane, any thoughts?", send_at=_now_iso(), status="approved",
        campaign_id="c-1", step=2)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    await sellbird.run_cycles(1)
    assert sellbird.mail.sent == []
    assert len(await sellbird.state.get_outbox(status="cancelled")) == 1
    assert len(await sellbird.state.get_outbox(status="pending_review")) == 1  # Sellbird's answer
    assert await sellbird.state.is_reply_processed("in-1")


# ── status truthfulness ──


@pytest.mark.asyncio
async def test_ollama_outage_status_survives_restart_and_clears(sellbird):
    await sellbird.state.init_db()
    await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    sellbird.scenario = "server_down"
    await sellbird.run_cycles(2)
    status = await sellbird.status()
    assert status["status"] == "paused" and "Ollama" in status["reason"]

    # A new heartbeat while the server is down remains paused and does no work.
    await sellbird.run_cycles(1)
    assert (await sellbird.status())["status"] == "paused"
    assert not await sellbird.state.is_reply_processed("in-1")

    # Once Ollama is restored, the next heartbeat records recovery and handles it.
    sellbird.recover()
    await sellbird.run_cycles(1)
    assert (await sellbird.status())["status"] == "ok"
    assert await sellbird.state.is_reply_processed("in-1")


@pytest.mark.asyncio
async def test_ollama_recovers_in_process_after_service_returns(sellbird):
    await sellbird.state.init_db()
    await _open_thread(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    sellbird.scenario = "server_down"
    seen = {}

    async def restore_after_pause():
        seen["status"] = (await sellbird.status())["status"]
        seen["processed"] = await sellbird.state.is_reply_processed("in-1")
        sellbird.recover()

    await sellbird.run_cycles(2, between={1: restore_after_pause})
    assert seen == {"status": "paused", "processed": False}
    assert (await sellbird.status())["status"] == "ok"
    assert await sellbird.state.is_reply_processed("in-1")


# ── sending waits for a completed mailbox phase (both providers) ──


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, model", [("ollama", "qwen3:1.7b")])
async def test_mailbox_error_holds_sends_until_replies_can_be_read(monkeypatch, tmp_path,
                                                                   provider, model):
    sellbird = Harness(monkeypatch, tmp_path, provider=provider, model=model)
    await sellbird.state.init_db()
    await _approved_item(sellbird.state)
    real_get_replies = sellbird.mail.get_replies

    async def mailbox_down(limit=50):
        raise OSError("synthetic mailbox unavailable")

    sellbird.mail.get_replies = mailbox_down
    seen = {}

    def after_failure():
        seen["sent"] = list(sellbird.mail.sent)
        sellbird.mail.get_replies = real_get_replies  # the mailbox is back

    await sellbird.run_cycles(3, between={1: after_failure})
    assert seen["sent"] == []
    assert [m["to"] for m in sellbird.mail.sent] == ["bob@beta.com"]  # once, after recovery
    assert len(await sellbird.state.get_outbox(status="sent")) == 1


@pytest.mark.asyncio
async def test_timed_out_reply_handling_holds_sends(sellbird, monkeypatch):
    await sellbird.state.init_db()
    await _approved_item(sellbird.state)
    real_get_replies = sellbird.mail.get_replies

    async def mailbox_hangs(limit=50):
        await asyncio.sleep(30)
        return []

    sellbird.mail.get_replies = mailbox_hangs
    monkeypatch.setattr(main_mod, "CYCLE_TIMEOUT_SECONDS", 0.5)
    seen = {}

    def after_timeout():
        seen["sent"] = list(sellbird.mail.sent)
        sellbird.mail.get_replies = real_get_replies
        monkeypatch.setattr(main_mod, "CYCLE_TIMEOUT_SECONDS", 600)

    await sellbird.run_cycles(2, between={1: after_timeout})
    assert seen["sent"] == []
    assert [m["to"] for m in sellbird.mail.sent] == ["bob@beta.com"]


@pytest.mark.asyncio
async def test_a_message_that_fails_to_process_holds_that_cycles_sends(sellbird, monkeypatch):
    """An error handling one reply (here: storing it) means stop-on-reply may
    not have run for it; the cycle sends nothing, the next one does."""
    await sellbird.state.init_db()
    await _open_thread(sellbird.state)
    await _approved_item(sellbird.state)
    sellbird.mail.inbound = [_reply("Could you share pricing details?")]
    real_update = sellbird.state.update_conversation
    calls = {"n": 0}

    async def flaky_update(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic database hiccup")
        return await real_update(*a, **k)

    sellbird.state.update_conversation = flaky_update
    seen = {}

    def after_first():
        seen["sent"] = list(sellbird.mail.sent)

    await sellbird.run_cycles(2, between={1: after_first})
    assert seen["sent"] == []
    assert [m["to"] for m in sellbird.mail.sent] == ["bob@beta.com"]
