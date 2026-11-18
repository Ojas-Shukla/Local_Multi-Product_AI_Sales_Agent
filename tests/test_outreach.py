from types import SimpleNamespace
import pytest

from sellbird.outreach import generate_personalised_email, classify_prospect_reply


class FakeState:
    def __init__(self):
        self.prospect = SimpleNamespace(id="p-1", company_id="c-1", first_name="Ada",
            last_name="Lovelace", title="Head of Operations", department="Operations",
            company="Example Co", industry="SaaS", company_size="51-200",
            personalization_notes="Hiring an operations manager", email="ada@example.test")
        self.company = SimpleNamespace(name="Example Co", description="Workflow platform",
            industry="SaaS", company_size="51-200", notes="", tech_stack=["PostgreSQL"])
        self.saved_email = None
        self.saved_reply = None

    async def get_prospect(self, pid): return self.prospect if pid == "p-1" else None
    async def get_company(self, cid): return self.company
    async def get_prospect_research(self, pid):
        return {"product_id":"prod-1", "research":{"relevant_facts":["Hiring operations roles"]}}
    async def get_buying_signal_classification(self, pid, product_id):
        return {"classification":{"overall_strength":"moderate","recommended_next_action":"Ask about workflow"}}
    async def save_personalised_email(self, *args): self.saved_email = args
    async def save_reply_classification(self, *args): self.saved_reply = args


class FakeBrain:
    model = "qwen3:1.7b"
    unavailable = ""
    def __init__(self, result): self.result = result; self.prompt = ""; self.kwargs = {}
    async def think_json(self, prompt, **kwargs):
        self.prompt, self.kwargs = prompt, kwargs
        return self.result


@pytest.fixture
def products(monkeypatch):
    import sellbird.sales_intelligence as si
    import sellbird.config as config
    async def fake_products(state, cfg):
        return [{"id":"prod-1", "name":"OpsFlow", "description":"Operations workflow automation",
                 "key_benefits":["Reduce manual handoffs"], "icp":{"department":"Operations"}}]
    monkeypatch.setattr(si, "list_products", fake_products)
    monkeypatch.setattr(config, "load_config", lambda: object())


@pytest.mark.asyncio
async def test_personalised_email_uses_local_context_and_saves_draft(products):
    state = FakeState()
    brain = FakeBrain({"subject":"Operations workflows at Example Co",
        "body":"Hi Ada, I noticed your team is hiring an operations manager. Would a short conversation about reducing manual handoffs be useful?",
        "personalization_basis":["Operations manager hiring"], "evidence_gaps":[]})
    result = await generate_personalised_email(brain, state, "p-1", "prod-1")
    assert result["provider"] == "ollama_local"
    assert result["status"] == "draft_only"
    assert result["model"] == "qwen3:1.7b"
    assert state.saved_email[0:2] == ("p-1", "prod-1")
    assert brain.kwargs["task"] == "generate_personalised_email_local"
    assert "Do not send the email" in brain.prompt
    assert "saved_buying_signals" in brain.prompt


@pytest.mark.asyncio
async def test_reply_classification_uses_supplied_sample_and_persists(products):
    state = FakeState()
    brain = FakeBrain({"intent":"interested", "sentiment":"positive", "buying_readiness":"medium",
        "summary":"The prospect is open to a discussion.", "key_points":["Requests a demo"],
        "recommended_next_action":"Offer two demo slots", "suggested_response":"Happy to arrange that.",
        "confidence":"high"})
    result = await classify_prospect_reply(brain, state, "p-1", "Prospect: Could you show me a demo?", "prod-1")
    assert result["intent"] == "interested"
    assert result["provider"] == "ollama_local"
    assert state.saved_reply[0:2] == ("p-1", "prod-1")
    assert "Could you show me a demo?" in brain.prompt
    assert brain.kwargs["task"] == "classify_reply_local"


@pytest.mark.asyncio
async def test_reply_requires_sample_conversation(products):
    with pytest.raises(ValueError, match="Paste a sample"):
        await classify_prospect_reply(FakeBrain({}), FakeState(), "p-1", "  ")


@pytest.mark.asyncio
async def test_unknown_reply_intent_is_rejected(products):
    result = {"intent":"invented_intent"}
    with pytest.raises(RuntimeError, match="valid reply classification"):
        await classify_prospect_reply(FakeBrain(result), FakeState(), "p-1", "Hello")
