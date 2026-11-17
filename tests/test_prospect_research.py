from types import SimpleNamespace

import pytest

from sellbird.prospect_research import build_local_research_context, generate_prospect_research


class FakeState:
    def __init__(self):
        self.saved = None
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

    async def get_prospect(self, prospect_id):
        return self.prospect if prospect_id == "p-1" else None

    async def get_company(self, company_id):
        return self.company

    async def get_observations(self, company_id="", limit=100):
        return [{"signal_code": "HIRING_ROLE", "value_text": "Operations Manager",
                 "value_num": None, "evidence_url": "https://example.test/careers",
                 "observed_at": "2026-11-01"}]

    async def save_prospect_research(self, *args):
        self.saved = args


class FakeBrain:
    model = "qwen3:1.7b"
    unavailable = ""

    async def think_json(self, *args, **kwargs):
        return {
            "company_overview": "Example Co builds workflow software.",
            "contact_context": "Ada is Head of Operations.",
            "relevant_facts": ["The company is listed as SaaS."],
            "evidence": [{"fact": "Hiring Operations Manager", "source": "https://example.test/careers",
                          "observed_at": "2026-11-01"}],
            "potential_relevance": ["An operations role may indicate process growth."],
            "research_gaps": ["Current technology budget is unknown."],
            "suggested_research_questions": ["Which workflow is currently most manual?"],
            "product_fit": "Product not selected.",
            "confidence": "medium",
        }


@pytest.mark.asyncio
async def test_build_context_uses_only_existing_local_records():
    state = FakeState()
    prospect, facts, source_count = await build_local_research_context(state, "p-1")
    assert prospect.id == "p-1"
    assert facts["company"]["name"] == "Example Co"
    assert facts["previously_downloaded_observations"][0]["signal_code"] == "HIRING_ROLE"
    assert source_count > 0


@pytest.mark.asyncio
async def test_unknown_prospect_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        await build_local_research_context(FakeState(), "missing")


@pytest.mark.asyncio
async def test_no_local_data_is_rejected():
    state = FakeState()
    state.prospect = SimpleNamespace(
        id="p-1", company_id="", first_name="", last_name="", title="", seniority="",
        department="", email="", linkedin_url="", source="", source_url="", company="",
        industry="", company_size="", personalization_notes="",
    )
    with pytest.raises(ValueError, match="No previously downloaded"):
        await build_local_research_context(state, "p-1")


@pytest.mark.asyncio
async def test_research_is_generated_and_persisted_with_local_provider(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return []

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())
    state = FakeState()
    result = await generate_prospect_research(FakeBrain(), state, "p-1")
    assert result["provider"] == "ollama_local"
    assert result["model"] == "qwen3:1.7b"
    assert result["source_mode"] == "previously_downloaded_local_data"
    assert result["confidence"] == "medium"
    assert state.saved[0] == "p-1"
    assert state.saved[3] == "qwen3:1.7b"


@pytest.mark.asyncio
async def test_invalid_model_output_is_not_saved(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return []

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())

    class BadBrain(FakeBrain):
        async def think_json(self, *args, **kwargs):
            return {"overview": "not the required schema"}

    state = FakeState()
    with pytest.raises(RuntimeError, match="valid prospect research"):
        await generate_prospect_research(BadBrain(), state, "p-1")
    assert state.saved is None


@pytest.mark.asyncio
async def test_selected_product_context_is_included(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return [{"id": "prod-1", "name": "Workflow Suite", "description": "Automates workflows",
                 "key_benefits": ["Less manual work"],
                 "icp": {"industries": ["SaaS"], "pain_points": ["Manual processes"]}}]

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())
    state = FakeState()
    result = await generate_prospect_research(FakeBrain(), state, "p-1", "prod-1")
    assert result["product_fit"] == "Product not selected."
    assert state.saved[1] == "prod-1"


@pytest.mark.asyncio
async def test_model_cannot_inject_unverified_evidence_urls(monkeypatch):
    import sellbird.sales_intelligence as sales_intelligence
    import sellbird.config as config_module

    async def fake_products(state, config):
        return []

    monkeypatch.setattr(sales_intelligence, "list_products", fake_products)
    monkeypatch.setattr(config_module, "load_config", lambda: object())

    class FakeEvidenceBrain(FakeBrain):
        async def think_json(self, *args, **kwargs):
            result = await super().think_json(*args, **kwargs)
            result["evidence"] = [{"fact": "Invented claim", "source": "https://fake.invalid"}]
            return result

    result = await generate_prospect_research(FakeEvidenceBrain(), FakeState(), "p-1")
    assert all(item["source"] != "https://fake.invalid" for item in result["evidence"])
    assert any("HIRING_ROLE" in item["fact"] for item in result["evidence"])
