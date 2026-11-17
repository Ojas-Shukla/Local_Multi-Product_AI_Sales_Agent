import json
from types import SimpleNamespace

import pytest

from sellbird.sales_intelligence import (
    BUSINESS_PROFILE_KEY, PRODUCTS_KEY, generate_business_profile,
    generate_product_icp, get_business_profile, list_products, save_product,
)


class FakeState:
    def __init__(self):
        self.settings = {}

    async def get_setting(self, key, default=""):
        return self.settings.get(key, default)

    async def set_setting(self, key, value):
        self.settings[key] = str(value)


class FakeBrain:
    model = "qwen3:1.7b"
    unavailable = ""

    def __init__(self, result):
        self.result = result
        self.prompts = []

    async def think_json(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return self.result


def fake_config():
    return SimpleNamespace(
        persona=SimpleNamespace(company="Acme Systems", role="Founder"),
        product=SimpleNamespace(name="Sales Copilot", description="Local sales workflow tool",
                                pricing="$20/month", key_benefits=["Saves research time"]),
        icp=SimpleNamespace(industries=["SaaS"], company_size="10-100", titles=["CEO"],
                            geography=["India"]),
    )


@pytest.mark.asyncio
async def test_existing_config_product_is_seeded_into_local_catalogue(tmp_path):
    state = FakeState()
    products = await list_products(state, fake_config())
    assert len(products) == 1
    assert products[0]["name"] == "Sales Copilot"
    assert products[0]["icp"]["industries"] == ["SaaS"]
    assert json.loads(await state.get_setting(PRODUCTS_KEY))[0]["id"] == products[0]["id"]


@pytest.mark.asyncio
async def test_can_add_multiple_products_without_overwriting_existing_icp(tmp_path):
    state = FakeState()
    first = (await list_products(state, fake_config()))[0]
    second = await save_product(state, {"name": "Support Bot", "description": "Customer support automation"}, fake_config())
    updated = await save_product(state, {"id": first["id"], "name": "Sales Copilot Plus",
                                         "description": "Updated sales workflow tool"}, fake_config())
    products = await list_products(state, fake_config())
    assert len(products) == 2
    assert updated["icp"]["industries"] == ["SaaS"]
    assert next(p for p in products if p["id"] == second["id"])["name"] == "Support Bot"


@pytest.mark.asyncio
async def test_business_profile_is_generated_by_brain_and_persisted(tmp_path):
    state = FakeState()
    result = {
        "company_name": "Acme Systems", "overview": "Builds local sales tools",
        "business_model": "Software subscription", "value_proposition": "Less manual research",
        "products_and_services": ["Sales Copilot"], "target_markets": ["B2B SaaS"],
        "differentiators": ["Local inference"], "known_facts": ["Product name is configured"],
        "unknowns": ["Revenue is not provided"],
    }
    brain = FakeBrain(result)
    profile = await generate_business_profile(brain, state, fake_config(), knowledge="Known product facts")
    assert profile["company_name"] == "Acme Systems"
    assert profile["provider"] == "ollama_local"
    assert profile["model"] == "qwen3:1.7b"
    assert "<untrusted" in brain.prompts[0][0]
    assert (await get_business_profile(state))["overview"] == "Builds local sales tools"
    assert await state.get_setting(BUSINESS_PROFILE_KEY)


@pytest.mark.asyncio
async def test_product_icp_is_saved_per_product_and_uses_profile_context(tmp_path):
    state = FakeState()
    product = (await list_products(state, fake_config()))[0]
    await state.set_setting(BUSINESS_PROFILE_KEY, json.dumps({"overview": "Acme sells sales software"}))
    result = {
        "industries": ["B2B SaaS"], "company_size": "20-200 employees",
        "titles": ["VP Sales", "Sales Operations Lead"], "geography": ["India"],
        "pain_points": ["Manual prospect research"], "buying_triggers": ["New sales team"],
        "disqualifiers": ["No outbound sales motion"], "rationale": "These teams need prospecting efficiency.",
    }
    brain = FakeBrain(result)
    generated = await generate_product_icp(brain, state, fake_config(), product["id"])
    assert generated["icp"]["industries"] == ["B2B SaaS"]
    assert generated["product"]["icp"]["model"] == "qwen3:1.7b"
    assert "Acme sells sales software" in brain.prompts[0][0]
    stored = await list_products(state, fake_config())
    assert stored[0]["icp"]["pain_points"] == ["Manual prospect research"]


@pytest.mark.asyncio
async def test_missing_company_name_is_not_fabricated(tmp_path):
    state = FakeState()
    config = fake_config()
    config.persona.company = "Your Company"
    with pytest.raises(ValueError, match="Set your company name"):
        await generate_business_profile(FakeBrain({}), state, config)


@pytest.mark.asyncio
async def test_invalid_model_json_is_rejected_not_presented_as_generated_intelligence(tmp_path):
    state = FakeState()
    with pytest.raises(RuntimeError, match="valid business profile"):
        await generate_business_profile(FakeBrain({"overview": "made up"}), state, fake_config())
    assert await get_business_profile(state) is None


@pytest.mark.asyncio
async def test_unknown_product_icp_request_is_rejected(tmp_path):
    state = FakeState()
    with pytest.raises(ValueError, match="Product not found"):
        await generate_product_icp(FakeBrain({}), state, fake_config(), "missing-id")


@pytest.mark.asyncio
async def test_ollama_outage_is_reported_and_not_replaced_with_cloud_output(tmp_path):
    state = FakeState()
    brain = FakeBrain(None)
    brain.unavailable = "Ollama offline"
    with pytest.raises(RuntimeError, match="Ollama is unavailable"):
        await generate_business_profile(brain, state, fake_config())
    assert await get_business_profile(state) is None


@pytest.mark.asyncio
async def test_sync_keeps_legacy_product_editor_aligned(tmp_path):
    from sellbird.sales_intelligence import sync_configured_product
    state = FakeState()
    config = fake_config()
    product = (await list_products(state, config))[0]
    await save_product(state, {"id": product["id"], "name": "Older product name",
                               "description": "Old description"}, config)
    config.product.name = "Updated from company settings"
    config.product.description = "Current configured description"
    synced = await sync_configured_product(state, config)
    assert synced["name"] == "Updated from company settings"
    assert synced["description"] == "Current configured description"
    assert synced["icp"]["industries"] == ["SaaS"]
