"""End-to-end integration coverage for the local, product-specific sales workflow.

The fake Brain returns deterministic model-shaped responses. StateManager is real
SQLite, so this test exercises actual migrations, persistence and cross-step reads
without contacting Ollama or any external service.
"""

import pytest
import pytest_asyncio

from sellbird.state import StateManager
from sellbird.models.company import Company
from sellbird.models.prospect import Prospect
from sellbird.prospect_research import generate_prospect_research
from sellbird.buying_signals import classify_buying_signals
from sellbird.prospect_intelligence import generate_problem_statement, score_prospect
from sellbird.outreach import generate_personalised_email, classify_prospect_reply


PRODUCTS = [
    {
        "id": "ops-product",
        "name": "OpsFlow",
        "description": "Workflow automation for operations teams",
        "key_benefits": ["Reduce manual handoffs"],
        "icp": {"industries": ["SaaS"], "titles": ["COO", "Head of Operations"]},
    },
    {
        "id": "sales-product",
        "name": "SalesPulse",
        "description": "Sales pipeline analytics",
        "key_benefits": ["Improve pipeline visibility"],
        "icp": {"industries": ["SaaS"], "titles": ["VP Sales"]},
    },
]


class WorkflowFakeBrain:
    model = "qwen3:1.7b"
    unavailable = ""

    def __init__(self):
        self.tasks = []

    async def think_json(self, prompt, **kwargs):
        task = kwargs.get("task")
        self.tasks.append(task)
        if task == "research_prospect_local_data":
            return {
                "company_overview": "Example Co builds workflow software.",
                "contact_context": "Ada leads operations.",
                "relevant_facts": ["Example Co is a SaaS company."],
                "evidence": [],
                "potential_relevance": ["Operations workflows may be relevant."],
                "research_gaps": ["Current workflow volume is unknown."],
                "suggested_research_questions": ["How are handoffs managed?"],
                "product_fit": "Potential fit for OpsFlow.",
                "confidence": "medium",
            }
        if task == "classify_buying_signals_local_data":
            return {
                "overall_strength": "moderate",
                "signals": [{
                    "type": "technology", "strength": "moderate",
                    "evidence": "Workflow software", "evidence_date": "invented-date",
                    "rationale": "The company description indicates a workflow software focus.",
                }],
                "buying_stage": "problem_aware", "urgency": "medium",
                "product_relevance": "Potential workflow fit.",
                "recommended_next_action": "Ask about current handoffs.",
                "confidence": "medium", "evidence_gaps": ["Budget is unknown."],
            }
        if task == "generate_problem_statement_local":
            return {
                "problem_statement": "The operations team may need to coordinate workflow handoffs.",
                "supporting_evidence": [{
                    "fact": "Head of Operations", "source_context": "Local prospect record",
                }],
                "hypothesis": "Workflow coordination may become harder as teams scale.",
                "product_relevance": "OpsFlow may help structure handoffs.",
                "discovery_questions": ["How are handoffs managed today?"],
                "evidence_gaps": ["Current process is not documented."],
                "confidence": "medium",
            }
        if task == "score_prospect_product_fit_local":
            return {
                "score": 82, "summary": "Strong role and industry alignment; evidence remains limited.",
                "factors": [{"name": "Role fit", "points": 25,
                             "evidence": "Head of Operations",
                             "rationale": "Matches the configured buyer role."}],
                "disqualifiers": [], "evidence_gaps": ["Budget is unknown."],
                "confidence": "medium",
            }
        if task == "generate_personalised_email_local":
            return {
                "subject": "Question about operations handoffs",
                "body": "Hi Ada, I noticed your operations leadership role. How are handoffs managed today?",
                "personalization_basis": ["Head of Operations"],
                "evidence_gaps": ["Specific workflow tooling is unknown."],
            }
        if task == "classify_reply_local":
            return {
                "intent": "interested", "sentiment": "positive",
                "buying_readiness": "medium", "summary": "The prospect asked to learn more.",
                "key_points": ["Requested more information."],
                "recommended_next_action": "Offer a short discovery call.",
                "suggested_response": "Happy to share more details.",
                "confidence": "medium",
            }
        raise AssertionError(f"Unexpected local AI task: {task}")


@pytest_asyncio.fixture
async def workflow_state(tmp_path):
    state = StateManager(str(tmp_path / "workflow.db"))
    await state.init_db()
    return state


@pytest.fixture
def local_products(monkeypatch):
    import sellbird.sales_intelligence as sales
    import sellbird.config as config

    async def fake_list_products(_state, _config):
        return PRODUCTS

    monkeypatch.setattr(sales, "list_products", fake_list_products)
    monkeypatch.setattr(config, "load_config", lambda: object())


@pytest.mark.asyncio
async def test_complete_local_sales_workflow_persists_each_product_step(
    workflow_state, local_products,
):
    state = workflow_state
    company_id = await state.add_company(Company(
        name="Example Co", domain="example.test", description="Workflow software company",
        industry="SaaS", company_size="51-200", source="local_fixture",
    ))
    prospect_id = await state.add_prospect(Prospect(
        first_name="Ada", last_name="Lovelace", email="ada@example.test",
        title="Head of Operations", department="Operations", company_id=company_id,
        company="Example Co", industry="SaaS", company_size="51-200",
        personalization_notes="Leads operations workflows", score=37,
        source="local_fixture",
    ))
    brain = WorkflowFakeBrain()
    product_id = "ops-product"

    research = await generate_prospect_research(brain, state, prospect_id, product_id)
    buying = await classify_buying_signals(brain, state, prospect_id, product_id)
    problem = await generate_problem_statement(brain, state, prospect_id, product_id)
    score = await score_prospect(brain, state, prospect_id, product_id)
    draft = await generate_personalised_email(brain, state, prospect_id, product_id)
    reply = await classify_prospect_reply(
        brain, state, prospect_id,
        "Prospect: This looks useful. Can you send more information?",
        product_id,
    )

    assert research["source_mode"] == "previously_downloaded_local_data"
    assert buying["source_mode"] == "previously_downloaded_local_data"
    assert problem["provider"] == score["provider"] == "ollama_local"
    assert score["score"] == 82 and score["tier"] == "high_priority"
    assert draft["status"] == "draft_only"
    assert reply["intent"] == "interested"

    # All six steps must be present in the real SQLite store for this product.
    assert (await state.get_prospect_research(prospect_id, product_id))["research"]["product_id"] == product_id
    assert (await state.get_buying_signal_classification(prospect_id, product_id))["payload"]["overall_strength"] == "moderate"
    assert (await state.get_problem_statement(prospect_id, product_id))["payload"]["problem_statement"] == problem["problem_statement"]
    assert (await state.get_prospect_ai_score(prospect_id, product_id))["payload"]["score"] == 82
    assert (await state.get_personalised_email(prospect_id, product_id))["payload"]["status"] == "draft_only"
    assert (await state.get_reply_classification(prospect_id, product_id))["payload"]["intent"] == "interested"

    # Existing deterministic Scout score is independent from the AI score.
    saved_prospect = await state.get_prospect(prospect_id)
    assert saved_prospect.score == 37

    # The workflow called exactly the six expected local model tasks.
    assert brain.tasks == [
        "research_prospect_local_data",
        "classify_buying_signals_local_data",
        "generate_problem_statement_local",
        "score_prospect_product_fit_local",
        "generate_personalised_email_local",
        "classify_reply_local",
    ]
    assert all("ollama_local" in item["provider"] for item in [
        research, buying, problem, score, draft, reply,
    ])

    # A different product must not inherit the first product's generated records.
    assert await state.get_prospect_research(prospect_id, "sales-product") is None
    assert await state.get_buying_signal_classification(prospect_id, "sales-product") is None
    assert await state.get_problem_statement(prospect_id, "sales-product") is None
    assert await state.get_prospect_ai_score(prospect_id, "sales-product") is None
    assert await state.get_personalised_email(prospect_id, "sales-product") is None
    assert await state.get_reply_classification(prospect_id, "sales-product") is None
