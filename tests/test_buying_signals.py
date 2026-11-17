from types import SimpleNamespace

import pytest

from sellbird.buying_signals import classify_buying_signals


class FakeState:
    def __init__(self):
        self.prospect = SimpleNamespace(
            id="p-1", company_id="c-1", first_name="Ada", last_name="Lovelace",
            title="Head of Operations", seniority="director", department="operations",
            email="ada@example.test", linkedin_url="", source="imported_csv",
            source_url="file://prospects.csv", company="Example Co", industry="SaaS",
            company_size="51-200", personalization_notes="Uses a CRM",
        )
        self.company = SimpleNamespace(
            name="Example Co", domain="example.test", website="https://example.test",
            description="Workflow software company", industry="SaaS", company_size="51-200",
            location="London", source="imported_csv", source_url="file://companies.csv",
            notes="", tech_stack=["PostgreSQL"], signals=[],
        )
        self.saved = None

    async def get_prospect(self, prospect_id):
        return self.prospect if prospect_id == "p-1" else None

    async def get_company(self, company_id):
        return self.company

    async def get_observations(self, company_id="", limit=100):
        return [{"signal_code": "HIRING_ROLE", "value_text": "Operations Manager",
                 "value_num": None, "evidence_url": "https://example.test/careers",
                 "observed_at": "2026-11-01"}]

    async def get_prospect_research(self, prospect_id):
        return {"research": {"potential_relevance": ["Operations team may be growing"]}}

    async def save_buying_signal_classification(self, *args):
        self.saved = args


class FakeBrain:
    model = "qwen3:1.7b"
    unavailable = ""

    def __init__(self, result):
        self.result = result
        self.prompt = ""
        self.kwargs = {}

    async def think_json(self, prompt, **kwargs):
        self.prompt, self.kwargs = prompt, kwargs
        return self.result


def model_result():
    return {
        "overall_strength": "moderate",
        "signals": [{"type": "hiring", "strength": "moderate",
                     "evidence": "Operations Manager", "evidence_date": "fake-date",
                     "rationale": "A role in operations may indicate team growth."}],
        "buying_stage": "problem_aware", "urgency": "medium",
        "product_relevance": "Potential fit for operations workflows.",
        "recommended_next_action": "Ask about current workflow bottlenecks.",
        "confidence": "medium", "evidence_gaps": ["Budget is unknown."],
    }


@pytest.mark.asyncio
async def test_classification_uses_local_data_and_persists_with_ollama_provider(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return []

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())
    state, brain = FakeState(), FakeBrain(model_result())
    result = await classify_buying_signals(brain, state, "p-1")
    assert result["overall_strength"] == "moderate"
    assert result["provider"] == "ollama_local"
    assert result["model"] == "qwen3:1.7b"
    assert result["signals"][0]["evidence_date"] == "2026-11-01"
    assert result["signals"][0]["evidence_source"] == "https://example.test/careers"
    assert state.saved[0:2] == ("p-1", "")
    assert brain.kwargs["task"] == "classify_buying_signals_local_data"
    assert "Do not browse" in brain.prompt
    assert "Previously downloaded local prospect/company records" in brain.prompt


@pytest.mark.asyncio
async def test_unknown_prospect_is_rejected():
    state = FakeState()
    state.prospect = None
    with pytest.raises(ValueError, match="not found"):
        await classify_buying_signals(FakeBrain(model_result()), state, "missing")


@pytest.mark.asyncio
async def test_unknown_product_is_rejected(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return []

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())
    with pytest.raises(ValueError, match="product was not found"):
        await classify_buying_signals(FakeBrain(model_result()), FakeState(), "p-1", "missing-product")


@pytest.mark.asyncio
async def test_invalid_model_fields_are_sanitized_and_fake_dates_are_not_trusted(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return []

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())
    result = model_result()
    result["overall_strength"] = "certain"
    result["confidence"] = "absolute"
    result["signals"].append({"type": "invented_category", "strength": "certain",
                              "evidence": "fabricated fact"})
    result["signals"][0]["evidence"] = "not present in source records"
    state = FakeState()
    classified = await classify_buying_signals(FakeBrain(result), state, "p-1")
    assert classified["overall_strength"] == "no_signal"
    assert classified["confidence"] == "low"
    assert classified["signals"] == []


@pytest.mark.asyncio
async def test_malformed_model_response_is_rejected(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return []

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())
    with pytest.raises(RuntimeError, match="valid buying-signal"):
        await classify_buying_signals(FakeBrain({"overall_strength": "strong"}), FakeState(), "p-1")


@pytest.mark.asyncio
async def test_ollama_unavailable_error_is_not_swallowed(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return []

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())
    brain = FakeBrain({})
    brain.unavailable = "Ollama is offline"
    with pytest.raises(RuntimeError, match="Ollama is unavailable"):
        await classify_buying_signals(brain, FakeState(), "p-1")


@pytest.mark.asyncio
async def test_selected_product_icp_is_included_in_local_prompt(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return [{"id": "prod-1", "name": "Ops Copilot", "description": "Workflow automation",
                 "key_benefits": ["Reduce manual work"],
                 "icp": {"industries": ["SaaS"], "titles": ["COO"],
                         "buying_triggers": ["Hiring operations staff"]}}]

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())
    brain = FakeBrain(model_result())
    result = await classify_buying_signals(brain, FakeState(), "p-1", "prod-1")
    assert result["product_id"] == "prod-1"
    assert "Ops Copilot" in brain.prompt
    assert "Hiring operations staff" in brain.prompt
