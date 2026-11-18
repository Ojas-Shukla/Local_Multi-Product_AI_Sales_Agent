from types import SimpleNamespace
import pytest

from sellbird.prospect_intelligence import generate_problem_statement, score_prospect


class FakeState:
    def __init__(self):
        self.prospect = SimpleNamespace(
            id="p-1", company_id="c-1", first_name="Ada", last_name="Lovelace",
            title="Head of Operations", department="Operations", company="Example Co",
            industry="SaaS", company_size="51-200", personalization_notes="Hiring operations staff",
            email="ada@example.test",
        )
        self.company = SimpleNamespace(
            name="Example Co", description="Workflow software", industry="SaaS",
            company_size="51-200", notes="", tech_stack=["PostgreSQL"],
        )
        self.saved_problem = None
        self.saved_score = None

    async def get_prospect(self, prospect_id):
        return self.prospect if prospect_id == "p-1" else None

    async def get_company(self, company_id):
        return self.company

    async def get_prospect_research(self, prospect_id):
        return {"product_id": "prod-1", "research": {"relevant_facts": ["Hiring operations staff"]}}

    async def get_buying_signal_classification(self, prospect_id, product_id):
        return {"payload": {"overall_strength": "moderate", "recommended_next_action": "Ask about workflows"}}

    async def save_problem_statement(self, *args):
        self.saved_problem = args

    async def save_prospect_ai_score(self, *args):
        self.saved_score = args


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


@pytest.fixture
def products(monkeypatch):
    import sellbird.sales_intelligence as sales
    import sellbird.config as config

    async def fake_products(state, cfg):
        return [{"id": "prod-1", "name": "OpsFlow", "description": "Workflow automation",
                 "key_benefits": ["Reduce manual handoffs"],
                 "icp": {"industries": ["SaaS"], "titles": ["COO"]}}]

    monkeypatch.setattr(sales, "list_products", fake_products)
    monkeypatch.setattr(config, "load_config", lambda: object())


@pytest.mark.asyncio
async def test_problem_statement_uses_product_context_and_persists(products):
    state = FakeState()
    brain = FakeBrain({
        "problem_statement": "The operations team may be coordinating manual handoffs.",
        "supporting_evidence": [{"fact": "Hiring operations staff", "source_context": "Saved research"}],
        "hypothesis": "The team may be scaling its workflows.",
        "product_relevance": "OpsFlow may reduce manual handoffs.",
        "discovery_questions": ["How are handoffs managed today?"],
        "evidence_gaps": ["Current workflow is unknown."], "confidence": "medium",
    })
    result = await generate_problem_statement(brain, state, "p-1", "prod-1")
    assert result["provider"] == "ollama_local"
    assert result["confidence"] == "medium"
    assert state.saved_problem[:2] == ("p-1", "prod-1")
    assert brain.kwargs["task"] == "generate_problem_statement_local"
    assert "OpsFlow" in brain.prompt
    assert "Clearly separate verified evidence from hypotheses" in brain.prompt


@pytest.mark.asyncio
async def test_problem_statement_rejects_empty_model_result(products):
    with pytest.raises(RuntimeError, match="valid problem statement"):
        await generate_problem_statement(FakeBrain({}), FakeState(), "p-1", "prod-1")


@pytest.mark.asyncio
async def test_prospect_score_is_bounded_and_product_specific(products):
    state = FakeState()
    brain = FakeBrain({
        "score": 84, "summary": "Strong product and role fit, with some evidence gaps.",
        "factors": [{"name": "Role fit", "points": 25, "evidence": "Head of Operations",
                     "rationale": "Matches the product buyer profile."}],
        "disqualifiers": [], "evidence_gaps": ["Budget is unknown."], "confidence": "high",
    })
    result = await score_prospect(brain, state, "p-1", "prod-1")
    assert result["score"] == 84
    assert result["tier"] == "high_priority"
    assert result["provider"] == "ollama_local"
    assert state.saved_score[:2] == ("p-1", "prod-1")
    assert brain.kwargs["task"] == "score_prospect_product_fit_local"
    assert "product-specific ICP" in brain.prompt


@pytest.mark.asyncio
async def test_prospect_score_rejects_out_of_range_score(products):
    brain = FakeBrain({"score": 101, "summary": "Invalid score"})
    with pytest.raises(RuntimeError, match="outside the 0–100 range"):
        await score_prospect(brain, FakeState(), "p-1", "prod-1")


@pytest.mark.asyncio
async def test_prospect_score_rejects_invalid_model_response(products):
    with pytest.raises(RuntimeError, match="valid prospect score"):
        await score_prospect(FakeBrain({"score": "high"}), FakeState(), "p-1", "prod-1")
