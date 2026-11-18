"""Dashboard API regression tests for product-specific local AI workflows."""

import asyncio

import pytest
from fastapi.testclient import TestClient

import sellbird.dashboard as dash
from sellbird.models.company import Company
from sellbird.models.prospect import Prospect
from sellbird.state import StateManager

ORIGIN = "http://127.0.0.1:5555"


@pytest.fixture
def ai_api_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "sellbird.db"
    monkeypatch.setattr(dash, "DB_PATH", db_path)
    monkeypatch.setattr(dash, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(dash, "PID_FILE", data_dir / "sellbird.pid")
    monkeypatch.setattr(dash, "LOG_FILE", data_dir / "sellbird.log")
    dash.token_store.configure(data_dir / "dashboard-token")

    state = StateManager(str(db_path))

    async def seed():
        await state.init_db()
        company_id = await state.add_company(Company(
            name="Regression Co", domain="regression.test", industry="SaaS",
            description="A locally stored SaaS company.",
        ))
        prospect_id = await state.add_prospect(Prospect(
            first_name="Ada", last_name="Lovelace", email="ada@regression.test",
            title="Head of Operations", company_id=company_id,
            company="Regression Co", industry="SaaS", score=41,
        ))
        await state.save_buying_signal_classification(
            prospect_id, "product-a", {"overall_strength": "moderate"}, "qwen3:1.7b", 2,
        )
        await state.save_problem_statement(
            prospect_id, "product-a", {"problem_statement": "Product A problem"}, "qwen3:1.7b",
        )
        await state.save_prospect_ai_score(
            prospect_id, "product-a", {"score": 82}, "qwen3:1.7b",
        )
        await state.save_personalised_email(
            prospect_id, "product-a", {"status": "draft_only", "subject": "Hello"}, "qwen3:1.7b",
        )
        await state.save_reply_classification(
            prospect_id, "product-a", {"intent": "interested"}, "qwen3:1.7b",
        )
        return prospect_id

    prospect_id = asyncio.run(seed())
    return {"db": db_path, "state": state, "prospect_id": prospect_id,
            "token": dash.token_store.get()}


@pytest.fixture
def ai_api_client(ai_api_env):
    with TestClient(
        dash.app,
        base_url=ORIGIN,
        headers={"Authorization": f"Bearer {ai_api_env['token']}"},
    ) as client:
        yield client


def test_all_saved_ai_workflow_results_are_available_through_dashboard_api(
    ai_api_client, ai_api_env,
):
    pid = ai_api_env["prospect_id"]
    expected = {
        "buying-signals": ("classification", {"overall_strength": "moderate"}),
        "problem-statement": ("problem_statement", {"problem_statement": "Product A problem"}),
        "score": ("score", {"score": 82}),
        "personalised-email": ("draft", {"status": "draft_only", "subject": "Hello"}),
        "reply-classification": ("classification", {"intent": "interested"}),
    }
    for endpoint, (key, value) in expected.items():
        response = ai_api_client.get(
            f"/api/prospects/{pid}/{endpoint}", params={"product_id": "product-a"},
        )
        assert response.status_code == 200
        assert response.json()[key] == value

    # Research is also product-scoped; an ungenerated product returns no result.
    research = ai_api_client.get(
        f"/api/prospects/{pid}/research", params={"product_id": "product-b"},
    )
    assert research.status_code == 200
    assert research.json()["research"] is None


def test_ai_workflow_dashboard_get_routes_return_404_for_unknown_prospect(ai_api_client):
    for endpoint in (
        "research", "buying-signals", "problem-statement", "score",
        "personalised-email", "reply-classification",
    ):
        response = ai_api_client.get(f"/api/prospects/not-a-real-prospect/{endpoint}")
        assert response.status_code == 404, endpoint


def test_reply_classification_rejects_empty_conversation(ai_api_client, monkeypatch, ai_api_env):
    async def fake_offer_brain():
        class NeverCalledBrain:
            async def think_json(self, *_args, **_kwargs):
                raise AssertionError("The model must not be called for empty input")
        return NeverCalledBrain(), ai_api_env["state"], object()

    monkeypatch.setattr(dash, "_offer_brain", fake_offer_brain)
    import sellbird.outreach as outreach

    async def fake_context(_state, _prospect_id, _product_id):
        return object(), {}

    monkeypatch.setattr(outreach, "_context", fake_context)
    response = ai_api_client.post(
        f"/api/prospects/{ai_api_env['prospect_id']}/reply-classification",
        json={"product_id": "product-a", "conversation_text": "   "},
    )
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_product_a_saved_results_do_not_leak_into_product_b_dashboard_reads(
    ai_api_client, ai_api_env,
):
    pid = ai_api_env["prospect_id"]
    for endpoint, key in (
        ("buying-signals", "classification"),
        ("problem-statement", "problem_statement"),
        ("score", "score"),
        ("personalised-email", "draft"),
        ("reply-classification", "classification"),
    ):
        response = ai_api_client.get(
            f"/api/prospects/{pid}/{endpoint}", params={"product_id": "product-b"},
        )
        assert response.status_code == 200
        assert response.json()[key] is None
