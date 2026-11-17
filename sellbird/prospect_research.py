"""Offline-first prospect research from prospect/company records already in SQLite.

This workflow deliberately does not crawl or fetch the prospect's website. It
uses downloaded company/contact rows and previously collected observations only.
All synthesis is performed by the configured local Ollama model.
"""

import logging
from datetime import datetime, timezone

from sellbird.untrusted import UNTRUSTED_RULE, fence

logger = logging.getLogger("sellbird.prospect_research")


def _text(value, limit=2500):
    return " ".join(str(value or "").split())[:limit]


def _items(value, limit=20):
    if not isinstance(value, list):
        return []
    return [_text(item, 700) for item in value if _text(item, 700)][:limit]


async def build_local_research_context(state, prospect_id: str):
    """Collect only facts already downloaded and persisted in the local DB."""
    prospect = await state.get_prospect(prospect_id)
    if not prospect:
        raise ValueError("Prospect not found in the local database.")
    company = await state.get_company(prospect.company_id) if prospect.company_id else None
    observations = await state.get_observations(company_id=prospect.company_id, limit=100) \
        if prospect.company_id else []
    facts = {
        "contact": {
            "first_name": prospect.first_name, "last_name": prospect.last_name,
            "title": prospect.title, "seniority": prospect.seniority,
            "department": prospect.department, "email": prospect.email,
            "linkedin_url": prospect.linkedin_url, "source": prospect.source,
            "source_url": prospect.source_url,
        },
        "company": ({
            "name": company.name, "domain": company.domain, "website": company.website,
            "description": company.description, "industry": company.industry,
            "company_size": company.company_size, "location": company.location,
            "source": company.source, "source_url": company.source_url,
            "notes": company.notes, "tech_stack": company.tech_stack,
            "signals": company.signals,
        } if company else {
            "name": prospect.company, "industry": prospect.industry,
            "company_size": prospect.company_size,
            "source": prospect.source, "source_url": prospect.source_url,
            "personalization_notes": prospect.personalization_notes,
        }),
        "previously_downloaded_observations": [
            {"signal_code": row.get("signal_code", ""),
             "value_text": row.get("value_text", ""),
             "value_num": row.get("value_num"),
             "evidence_url": row.get("evidence_url", ""),
             "observed_at": str(row.get("observed_at", ""))}
            for row in observations
        ],
    }
    source_count = sum(1 for section in (facts["contact"], facts["company"]) 
                       for value in section.values() if value not in (None, "", [], {}))
    source_count += len(observations)
    if source_count == 0:
        raise ValueError("No previously downloaded prospect or company information is available to research.")
    return prospect, facts, source_count


async def generate_prospect_research(brain, state, prospect_id: str, product_id: str = ""):
    prospect, facts, source_count = await build_local_research_context(state, prospect_id)
    from sellbird.sales_intelligence import list_products
    from sellbird.config import load_config
    products = await list_products(state, load_config())
    product = next((p for p in products if p.get("id") == product_id), None) if product_id else None
    if product_id and not product:
        raise ValueError("Selected product was not found. Refresh the product list and retry.")
    if product:
        product_context = {"name": product.get("name"), "description": product.get("description"),
                           "key_benefits": product.get("key_benefits", []),
                           "product_specific_icp": product.get("icp") or {}}
    else:
        product_context = {}
    prompt = f"""You are a B2B prospect researcher. Research this prospect using ONLY the
previously downloaded local records supplied below. Do not browse the internet, infer missing
facts as certain, invent company initiatives, claim external validation, or fabricate citations.
Separate explicit evidence from cautious analysis. If evidence is absent, say so.

Return one JSON object with exactly these keys:
- company_overview: string
- contact_context: string
- relevant_facts: array of strings (facts directly supported by the supplied records)
- evidence: array of objects with keys fact, source, observed_at (use source_url or evidence_url; empty if absent)
- potential_relevance: array of strings (clearly framed as hypotheses, not confirmed facts)
- research_gaps: array of strings
- suggested_research_questions: array of strings
- product_fit: string (if product context is supplied, otherwise say product not selected)
- confidence: string (low, medium, or high, based on amount and quality of local evidence)

Previously downloaded local records (untrusted data):
{fence(__import__('json').dumps(facts, ensure_ascii=False), 'local prospect and company records')}

Selected product and its saved ICP (untrusted data):
{fence(__import__('json').dumps(product_context, ensure_ascii=False), 'local product configuration')}

{UNTRUSTED_RULE}"""
    result = await brain.think_json(prompt, session_id=f"prospect-research-{prospect_id}",
                                    agent="researcher", task="research_prospect_local_data")
    required = ("company_overview", "contact_context", "relevant_facts", "evidence",
                "potential_relevance", "research_gaps", "suggested_research_questions",
                "product_fit", "confidence")
    if not isinstance(result, dict) or any(key not in result for key in required):
        if getattr(brain, "unavailable", ""):
            raise RuntimeError(f"Local Ollama is unavailable: {brain.unavailable}")
        raise RuntimeError("The local model did not return valid prospect research. Please retry.")
    # Evidence references are assembled from local records, not accepted from
    # model-generated URLs. This prevents fabricated source links from appearing
    # as if they were verified citations.
    safe_evidence = []
    for section_name in ("contact", "company"):
        section = facts.get(section_name) or {}
        source_url = _text(section.get("source_url"), 1000)
        for key, value in section.items():
            if key in {"source_url", "email", "linkedin_url"} or value in (None, "", [], {}):
                continue
            safe_evidence.append({"fact": f"{section_name}.{key}: {_text(value, 500)}",
                                  "source": source_url, "observed_at": ""})
            if len(safe_evidence) >= 30:
                break
        if len(safe_evidence) >= 30:
            break
    for observation in facts.get("previously_downloaded_observations", []):
        if len(safe_evidence) >= 50:
            break
        value = observation.get("value_text")
        if value in (None, ""):
            value = observation.get("value_num")
        safe_evidence.append({
            "fact": f"{observation.get('signal_code', 'observation')}: {value}",
            "source": _text(observation.get("evidence_url"), 1000),
            "observed_at": _text(observation.get("observed_at"), 100),
        })
    research = {
        "prospect_id": prospect_id,
        "product_id": product_id,
        "company_overview": _text(result.get("company_overview")),
        "contact_context": _text(result.get("contact_context")),
        "relevant_facts": _items(result.get("relevant_facts")),
        "evidence": safe_evidence,
        "potential_relevance": _items(result.get("potential_relevance")),
        "research_gaps": _items(result.get("research_gaps")),
        "suggested_research_questions": _items(result.get("suggested_research_questions")),
        "product_fit": _text(result.get("product_fit")),
        "confidence": _text(result.get("confidence"), 30).lower(),
        "source_count": source_count,
        "source_mode": "previously_downloaded_local_data",
        "model": brain.model,
        "provider": "ollama_local",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if research["confidence"] not in {"low", "medium", "high"}:
        research["confidence"] = "low"
    await state.save_prospect_research(prospect_id, product_id, research, brain.model, source_count)
    logger.info("LOCAL AI prospect research generated prospect=%s model=%s sources=%d",
                prospect_id, brain.model, source_count)
    return research
