"""Tests for state management (SQLite operations)."""

import asyncio
import tempfile
import os

import pytest
import pytest_asyncio

from sellbird.state import StateManager
from sellbird.models.prospect import Prospect
from sellbird.models.company import Company


@pytest_asyncio.fixture
async def state():
    """Create a StateManager with a temp database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        sm = StateManager(db_path)
        await sm.init_db()
        yield sm


@pytest.mark.asyncio
async def test_add_and_get_company(state):
    company = Company(name="Acme", domain="acme.com", industry="SaaS")
    company_id = await state.add_company(company)
    assert company_id

    fetched = await state.get_company_by_domain("acme.com")
    assert fetched is not None
    assert fetched.name == "Acme"


@pytest.mark.asyncio
async def test_company_exists(state):
    company = Company(name="Acme", domain="acme.com")
    await state.add_company(company)
    assert await state.company_exists("acme.com")
    assert not await state.company_exists("notreal.com")


@pytest.mark.asyncio
async def test_add_and_get_prospect(state):
    p = Prospect(
        first_name="Jane", last_name="Doe", email="jane@acme.com",
        title="VP Sales", status="new",
    )
    prospect_id = await state.add_prospect(p)
    assert prospect_id

    assert await state.prospect_exists(email="jane@acme.com")
    assert not await state.prospect_exists(email="nobody@acme.com")


@pytest.mark.asyncio
async def test_get_prospect_by_email(state):
    p = Prospect(
        first_name="Jane", last_name="Doe", email="jane@acme.com",
        title="VP Sales",
    )
    await state.add_prospect(p)

    found = await state.get_prospect_by_email("jane@acme.com")
    assert found is not None
    assert found.first_name == "Jane"

    not_found = await state.get_prospect_by_email("nobody@acme.com")
    assert not_found is None


@pytest.mark.asyncio
async def test_prospect_status_update(state):
    p = Prospect(first_name="Jane", last_name="Doe", email="jane@acme.com", title="VP")
    pid = await state.add_prospect(p)
    await state.update_prospect_status(pid, "contacted")

    prospects = await state.get_prospects_by_status("contacted")
    assert len(prospects) == 1
    assert prospects[0].status == "contacted"


@pytest.mark.asyncio
async def test_usage_tracking(state):
    """Usage increment should use UPSERT, not insert duplicate rows."""
    await state.increment_usage()
    await state.increment_usage()
    await state.increment_usage()

    count = await state.get_usage_today()
    assert count == 3


@pytest.mark.asyncio
async def test_reply_deduplication(state):
    assert not await state.is_reply_processed("reply-123")
    await state.mark_reply_processed("reply-123")
    assert await state.is_reply_processed("reply-123")
    # Marking again should not error
    await state.mark_reply_processed("reply-123")


@pytest.mark.asyncio
async def test_contacts_for_company(state):
    company = Company(name="Acme", domain="acme.com")
    cid = await state.add_company(company)

    p1 = Prospect(first_name="Jane", last_name="Doe", email="jane@acme.com", title="VP", company_id=cid)
    p2 = Prospect(first_name="John", last_name="Doe", email="john@acme.com", title="Dir", company_id=cid)
    await state.add_prospect(p1)
    await state.add_prospect(p2)

    contacts = await state.get_contacts_for_company(cid)
    assert len(contacts) == 2


@pytest.mark.asyncio
async def test_product_specific_ai_records_persist_and_upsert(state):
    prospect_id = await state.add_prospect(Prospect(
        first_name="Ada", last_name="Lovelace", email="ada@records.test", title="COO",
    ))

    buying = {"overall_strength": "moderate", "signals": []}
    await state.save_buying_signal_classification(
        prospect_id, "product-a", buying, "qwen3:1.7b", 3,
    )
    fetched_buying = await state.get_buying_signal_classification(prospect_id, "product-a")
    assert fetched_buying["payload"] == buying
    assert fetched_buying["provider"] == "ollama_local"
    assert fetched_buying["source_count"] == 3

    email = {"subject": "Hello", "body": "Draft only", "status": "draft_only"}
    await state.save_personalised_email(prospect_id, "product-a", email, "qwen3:1.7b")
    assert (await state.get_personalised_email(prospect_id, "product-a"))["payload"] == email

    reply = {"intent": "interested", "summary": "Asked for details"}
    await state.save_reply_classification(prospect_id, "product-a", reply, "qwen3:1.7b")
    assert (await state.get_reply_classification(prospect_id, "product-a"))["payload"] == reply

    problem = {"problem_statement": "Potential workflow friction"}
    await state.save_problem_statement(prospect_id, "product-a", problem, "qwen3:1.7b")
    assert (await state.get_problem_statement(prospect_id, "product-a"))["payload"] == problem

    score = {"score": 78, "tier": "medium_priority"}
    await state.save_prospect_ai_score(prospect_id, "product-a", score, "qwen3:1.7b")
    assert (await state.get_prospect_ai_score(prospect_id, "product-a"))["payload"] == score

    # The same prospect can have separate results for different products.
    await state.save_prospect_ai_score(
        prospect_id, "product-b", {"score": 35, "tier": "poor_fit"}, "qwen3:1.7b",
    )
    assert (await state.get_prospect_ai_score(prospect_id, "product-a"))["payload"]["score"] == 78
    assert (await state.get_prospect_ai_score(prospect_id, "product-b"))["payload"]["score"] == 35

    # Re-running a workflow replaces the latest result instead of duplicating rows.
    await state.save_prospect_ai_score(
        prospect_id, "product-a", {"score": 82, "tier": "high_priority"}, "qwen3:1.7b",
    )
    assert (await state.get_prospect_ai_score(prospect_id, "product-a"))["payload"]["score"] == 82
