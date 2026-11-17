"""Local business-profile and product-specific ICP workflows.

All generated intelligence is produced by the configured local Ollama model and
persisted in the local SQLite settings store. No hosted model fallback exists.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone

from sellbird.untrusted import UNTRUSTED_RULE, fence

logger = logging.getLogger("sellbird.sales_intelligence")
PRODUCTS_KEY = "sales_products_v1"
BUSINESS_PROFILE_KEY = "business_profile_v1"
MAX_TEXT = 4000
MAX_LIST = 30


def _clean_text(value, limit=MAX_TEXT):
    return " ".join(str(value or "").split())[:limit]


def _clean_list(value):
    if isinstance(value, str):
        value = value.splitlines()
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)][:MAX_LIST]


def _decode(raw, fallback):
    try:
        value = json.loads(raw) if raw else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback
    return value


async def list_products(state, config=None):
    """Return locally stored products, seeding the existing configured product once."""
    products = _decode(await state.get_setting(PRODUCTS_KEY), [])
    if not isinstance(products, list):
        products = []
    if not products and config is not None:
        p, icp = config.product, config.icp
        products = [{
            "id": uuid.uuid4().hex[:12], "name": p.name, "description": p.description,
            "pricing": p.pricing, "key_benefits": list(p.key_benefits or []),
            "icp": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }]
        await state.set_setting(PRODUCTS_KEY, json.dumps(products, ensure_ascii=False))
        # Preserve the legacy single-product ICP as the initial product ICP.
        legacy_icp = {
            "industries": list(icp.industries or []), "company_size": icp.company_size,
            "titles": list(icp.titles or []), "geography": list(icp.geography or []),
        }
        products[0]["icp"] = legacy_icp
        await state.set_setting(PRODUCTS_KEY, json.dumps(products, ensure_ascii=False))
    return [p for p in products if isinstance(p, dict)]


async def save_product(state, data, config=None):
    if not isinstance(data, dict):
        raise ValueError("Product details must be an object.")
    name = _clean_text(data.get("name"), 160)
    description = _clean_text(data.get("description"))
    if not name or not description:
        raise ValueError("Product name and description are required.")
    products = await list_products(state, config)
    product_id = _clean_text(data.get("id"), 40)
    if product_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", product_id):
        raise ValueError("Product ID contains unsupported characters.")
    product_id = product_id or uuid.uuid4().hex[:12]
    existing = next((p for p in products if p.get("id") == product_id), None)
    product = {
        "id": product_id, "name": name, "description": description,
        "pricing": _clean_text(data.get("pricing"), 500),
        "key_benefits": _clean_list(data.get("key_benefits")),
        "icp": existing.get("icp") if existing else None,
        "created_at": existing.get("created_at") if existing else datetime.now(timezone.utc).isoformat(),
    }
    if existing:
        products = [product if p.get("id") == product_id else p for p in products]
    else:
        products.append(product)
    await state.set_setting(PRODUCTS_KEY, json.dumps(products, ensure_ascii=False))
    return product


async def sync_configured_product(state, config):
    """Keep the original single-product settings editor aligned with its catalogue entry."""
    products = await list_products(state, config)
    if not products:
        return None
    current = products[0]
    current.update({
        "name": _clean_text(config.product.name, 160),
        "description": _clean_text(config.product.description),
        "pricing": _clean_text(config.product.pricing, 500),
        "key_benefits": _clean_list(config.product.key_benefits),
    })
    current_icp = current.get("icp") or {}
    current_icp.update({
        "industries": _clean_list(config.icp.industries),
        "company_size": _clean_text(config.icp.company_size, 500),
        "titles": _clean_list(config.icp.titles),
        "geography": _clean_list(config.icp.geography),
    })
    current["icp"] = current_icp
    await state.set_setting(PRODUCTS_KEY, json.dumps(products, ensure_ascii=False))
    return current


async def get_business_profile(state):
    value = _decode(await state.get_setting(BUSINESS_PROFILE_KEY), None)
    return value if isinstance(value, dict) else None


async def generate_business_profile(brain, state, config, knowledge="", competitors=""):
    """Generate and persist a factual business profile from local configuration."""
    company = _clean_text(config.persona.company, 200)
    if not company or company.lower() in {"your company", "company name"}:
        raise ValueError("Set your company name in the Company settings before generating a profile.")
    products = await list_products(state, config)
    facts = {
        "company": company,
        "sender_role": _clean_text(config.persona.role, 300),
        "configured_product": {
            "name": config.product.name, "description": config.product.description,
            "pricing": config.product.pricing,
            "key_benefits": list(config.product.key_benefits or []),
        },
        "products": [{k: p.get(k) for k in ("name", "description", "pricing", "key_benefits")}
                     for p in products],
        "product_knowledge": str(knowledge or "")[:8000],
        "competitor_notes": str(competitors or "")[:8000],
    }
    prompt = f"""You are a business analyst. Create a concise, evidence-grounded business profile.
Use only the supplied facts. Do not invent customers, revenue, market share, certifications,
locations, performance claims, or product capabilities. If a fact is unknown, say it is unknown.

Return one JSON object with these keys:
- company_name: string
- overview: string
- business_model: string
- value_proposition: string
- products_and_services: array of strings
- target_markets: array of strings
- differentiators: array of strings
- known_facts: array of strings
- unknowns: array of strings

Source facts (untrusted data):
{fence(json.dumps(facts, ensure_ascii=False), 'local company configuration')}

{UNTRUSTED_RULE}"""
    result = await brain.think_json(prompt, session_id="business-profile", agent="analyst",
                                    task="generate_business_profile")
    required = ("company_name", "overview", "business_model", "value_proposition",
                "products_and_services", "target_markets", "differentiators", "known_facts", "unknowns")
    if not isinstance(result, dict) or any(key not in result for key in required):
        if getattr(brain, "unavailable", ""):
            raise RuntimeError(f"Local Ollama is unavailable: {brain.unavailable}")
        raise RuntimeError("The local model did not return a valid business profile. Please retry.")
    profile = {
        "company_name": _clean_text(result["company_name"], 200) or company,
        "overview": _clean_text(result["overview"]),
        "business_model": _clean_text(result["business_model"]),
        "value_proposition": _clean_text(result["value_proposition"]),
        "products_and_services": _clean_list(result["products_and_services"]),
        "target_markets": _clean_list(result["target_markets"]),
        "differentiators": _clean_list(result["differentiators"]),
        "known_facts": _clean_list(result["known_facts"]),
        "unknowns": _clean_list(result["unknowns"]),
        "model": brain.model, "provider": "ollama_local",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    await state.set_setting(BUSINESS_PROFILE_KEY, json.dumps(profile, ensure_ascii=False))
    logger.info("LOCAL AI business profile generated company=%s model=%s", company, brain.model)
    return profile


async def generate_product_icp(brain, state, config, product_id, business_profile=None):
    products = await list_products(state, config)
    product = next((p for p in products if p.get("id") == product_id), None)
    if not product:
        raise ValueError("Product not found. Refresh the product list and try again.")
    profile = business_profile or await get_business_profile(state)
    prompt = f"""You are a B2B ideal-customer-profile analyst. Generate an ICP specifically for the
selected product, not a generic ICP for the company. Use only the business and product facts
provided. Do not invent market research or claim external validation. Keep each field actionable.

Return one JSON object with these keys:
- industries: array of target industries
- company_size: string describing suitable company size
- titles: array of relevant buyer/job titles
- geography: array of target geographies, or an empty array if unknown
- pain_points: array of problems this product can address
- buying_triggers: array of observable events or conditions indicating need
- disqualifiers: array of poor-fit conditions
- rationale: string explaining why this ICP fits this product

Business profile:
{fence(json.dumps(profile or {}, ensure_ascii=False), 'locally generated business profile')}

Selected product:
{fence(json.dumps({k: product.get(k) for k in ('name','description','pricing','key_benefits')}, ensure_ascii=False), 'local product record')}

{UNTRUSTED_RULE}"""
    result = await brain.think_json(prompt, session_id=f"product-icp-{product_id}",
                                    agent="analyst", task="generate_product_icp")
    required = ("industries", "company_size", "titles", "geography", "pain_points",
                "buying_triggers", "disqualifiers", "rationale")
    if not isinstance(result, dict) or any(key not in result for key in required):
        if getattr(brain, "unavailable", ""):
            raise RuntimeError(f"Local Ollama is unavailable: {brain.unavailable}")
        raise RuntimeError("The local model did not return a valid product-specific ICP. Please retry.")
    icp = {
        "industries": _clean_list(result["industries"]),
        "company_size": _clean_text(result["company_size"], 500),
        "titles": _clean_list(result["titles"]),
        "geography": _clean_list(result["geography"]),
        "pain_points": _clean_list(result["pain_points"]),
        "buying_triggers": _clean_list(result["buying_triggers"]),
        "disqualifiers": _clean_list(result["disqualifiers"]),
        "rationale": _clean_text(result["rationale"]),
        "model": brain.model, "provider": "ollama_local",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    product["icp"] = icp
    await state.set_setting(PRODUCTS_KEY, json.dumps(products, ensure_ascii=False))
    logger.info("LOCAL AI product ICP generated product=%s model=%s", product["name"], brain.model)
    return {"product": product, "icp": icp}
