"""Buying-signal classification from previously downloaded local prospect data."""

import json
import logging
from datetime import datetime, timezone

from sellbird.untrusted import UNTRUSTED_RULE, fence
from sellbird.prospect_research import build_local_research_context, _text, _items

logger = logging.getLogger("sellbird.buying_signals")

ALLOWED_STRENGTH = {"strong", "moderate", "weak", "no_signal"}
ALLOWED_TYPES = {
    "hiring", "funding", "expansion", "technology", "leadership_change",
    "business_growth", "engagement", "other",
}


async def classify_buying_signals(brain, state, prospect_id: str, product_id: str = ""):
    """Classify evidence already in SQLite; no web discovery or paid enrichment."""
    prospect, facts, source_count = await build_local_research_context(state, prospect_id)
    from sellbird.sales_intelligence import list_products
    from sellbird.config import load_config
    products = await list_products(state, load_config())
    product = next((p for p in products if p.get("id") == product_id), None) if product_id else None
    if product_id and not product:
        raise ValueError("Selected product was not found. Refresh the product list and retry.")

    saved_research = await state.get_prospect_research(prospect_id) if hasattr(state, "get_prospect_research") else None
    # Prospect research is stored as the latest product-specific run. Never
    # mix it into a different product's buying-signal classification.
    if saved_research and (saved_research.get("product_id") or "") == (product_id or ""):
        research_context = saved_research.get("research", {})
    else:
        research_context = {}
    product_context = ({
        "name": product.get("name"), "description": product.get("description"),
        "key_benefits": product.get("key_benefits", []),
        "product_specific_icp": product.get("icp") or {},
    } if product else {})

    prompt = f"""You are a B2B buying-signal analyst. Classify buying signals for this prospect using ONLY
facts in the supplied previously downloaded local records and saved local research. Do not browse,
request enrichment, invent events, or treat a hypothesis as evidence. Distinguish an actual buying
signal from generic company facts. If no meaningful signal is present, return no_signal and explain
that evidence is insufficient. Consider recency when dates are present, but do not invent dates.

Return one JSON object with exactly these keys:
- overall_strength: one of strong, moderate, weak, no_signal
- signals: array of objects with keys type, strength, evidence, evidence_date, rationale
  (type must be hiring, funding, expansion, technology, leadership_change, business_growth,
  engagement, or other; strength must be strong, moderate, weak, or no_signal)
- buying_stage: string (early_research, problem_aware, solution_evaluation, active_purchase, or unknown)
- urgency: string (high, medium, low, or unknown)
- product_relevance: string explaining relevance to the selected product and its ICP, or say no
  product selected
- recommended_next_action: string
- confidence: one of low, medium, high
- evidence_gaps: array of strings

Previously downloaded local prospect/company records (untrusted data):
{fence(json.dumps(facts, ensure_ascii=False), 'local prospect records')}

Previously generated local research (untrusted data; may be absent):
{fence(json.dumps(research_context, ensure_ascii=False), 'saved local prospect research')}

Selected product and product-specific ICP (untrusted data; may be absent):
{fence(json.dumps(product_context, ensure_ascii=False), 'product and ICP')}

{UNTRUSTED_RULE}"""
    result = await brain.think_json(
        prompt, session_id=f"buying-signals-{prospect_id}", agent="researcher",
        task="classify_buying_signals_local_data",
    )
    required = ("overall_strength", "signals", "buying_stage", "urgency",
                "product_relevance", "recommended_next_action", "confidence", "evidence_gaps")
    if not isinstance(result, dict) or any(key not in result for key in required):
        if getattr(brain, "unavailable", ""):
            raise RuntimeError(f"Local Ollama is unavailable: {brain.unavailable}")
        raise RuntimeError("The local model did not return a valid buying-signal classification. Please retry.")

    safe_signals = []
    # The model cannot create evidence. Every returned signal must match a
    # concrete field or observation already present in the local input.
    evidence_candidates = []
    for section_name in ("contact", "company"):
        section = facts.get(section_name) or {}
        source_url = _text(section.get("source_url"), 1000)
        for field_name, value in section.items():
            if field_name in {"source_url", "email", "linkedin_url"} or value in (None, "", [], {}):
                continue
            rendered = _text(", ".join(map(str, value)) if isinstance(value, list) else value, 700)
            if rendered:
                evidence_candidates.append({"value": rendered, "source": source_url, "date": ""})
    local_observations = facts.get("previously_downloaded_observations", [])
    for observation in local_observations:
        value = observation.get("value_text")
        if value in (None, ""):
            value = observation.get("value_num")
        rendered = _text(value, 700)
        if rendered:
            evidence_candidates.append({
                "value": rendered,
                "source": _text(observation.get("evidence_url"), 1000),
                "date": _text(observation.get("observed_at"), 100),
            })
    for signal in result.get("signals", []) if isinstance(result.get("signals"), list) else []:
        if not isinstance(signal, dict):
            continue
        signal_type = _text(signal.get("type"), 40).lower()
        strength = _text(signal.get("strength"), 20).lower()
        if signal_type not in ALLOWED_TYPES or strength not in ALLOWED_STRENGTH:
            continue
        model_evidence = _text(signal.get("evidence"), 700).lower()
        if not model_evidence:
            continue
        matching = next((candidate for candidate in sorted(
                             evidence_candidates, key=lambda item: len(item["value"]), reverse=True)
                         if len(candidate["value"]) >= 4 and
                         (candidate["value"].lower() in model_evidence or
                          model_evidence in candidate["value"].lower())), None)
        if not matching:
            continue
        safe_signals.append({
            "type": signal_type,
            "strength": strength,
            "evidence": matching["value"],
            "evidence_date": matching["date"],
            "evidence_source": matching["source"],
            "rationale": _text(signal.get("rationale"), 1000),
        })

    overall = _text(result.get("overall_strength"), 20).lower()
    if overall not in ALLOWED_STRENGTH or not safe_signals:
        overall = "no_signal"
    confidence = _text(result.get("confidence"), 20).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    stage = _text(result.get("buying_stage"), 40).lower()
    if stage not in {"early_research", "problem_aware", "solution_evaluation", "active_purchase", "unknown"}:
        stage = "unknown"
    urgency = _text(result.get("urgency"), 20).lower()
    if urgency not in {"high", "medium", "low", "unknown"}:
        urgency = "unknown"
    if overall == "no_signal":
        stage, urgency, confidence = "unknown", "unknown", "low"

    classification = {
        "prospect_id": prospect_id, "product_id": product_id or "",
        "overall_strength": overall, "signals": safe_signals,
        "buying_stage": stage, "urgency": urgency,
        "product_relevance": _text(result.get("product_relevance")),
        "recommended_next_action": _text(result.get("recommended_next_action")),
        "confidence": confidence, "evidence_gaps": _items(result.get("evidence_gaps")),
        "source_count": source_count, "source_mode": "previously_downloaded_local_data",
        "model": brain.model, "provider": "ollama_local",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    await state.save_buying_signal_classification(
        prospect_id, product_id, classification, brain.model, source_count,
    )
    logger.info("LOCAL AI buying signals classified prospect=%s model=%s signals=%d",
                prospect_id, brain.model, len(safe_signals))
    return classification
