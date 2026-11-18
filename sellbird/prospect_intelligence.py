"""Product-specific problem statements and AI prospect scoring.

Both workflows use only information already stored locally and the configured
Ollama Brain. They never fetch prospect data or send outreach.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sellbird.outreach import _context
from sellbird.untrusted import UNTRUSTED_RULE, fence
from sellbird.prospect_research import _text, _items

logger = logging.getLogger("sellbird.prospect_intelligence")



def _local_fact_strings(context: dict) -> list[str]:
    """Flatten prospect/company/research/signal values for evidence checks."""
    values = []

    def visit(value):
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            clean = _text(value, 2000)
            if len(clean) >= 4:
                values.append(clean)

    for key in ("prospect", "company", "saved_local_research", "saved_buying_signals"):
        visit(context.get(key))
    return list(dict.fromkeys(values))


def _matches_local_fact(claim: str, facts: list[str]) -> bool:
    claim = _text(claim, 1000).casefold()
    return bool(claim) and any(
        len(fact) >= 4 and (fact.casefold() in claim or claim in fact.casefold())
        for fact in facts
    )

def _confidence(value: object) -> str:
    value = _text(value, 20).lower()
    return value if value in {"low", "medium", "high"} else "low"


def _score_value(value: object) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError("The local model did not return a valid prospect score. Please retry.")
    if not 0 <= number <= 100:
        raise RuntimeError("The local model returned a score outside the 0–100 range.")
    return number


def _score_tier(score: int) -> str:
    if score >= 80:
        return "high_priority"
    if score >= 60:
        return "medium_priority"
    if score >= 40:
        return "low_priority"
    return "poor_fit"


async def generate_problem_statement(brain, state, prospect_id: str, product_id: str = ""):
    """Generate an evidence-grounded prospect problem statement for a product."""
    _prospect, context = await _context(state, prospect_id, product_id)
    prompt = f"""You are a B2B sales discovery analyst. Formulate the prospect's likely
business problem in relation to the selected product. Use only facts in the supplied
local context. Clearly separate verified evidence from hypotheses. Do not claim the
prospect has a problem merely because the product could address it. If evidence is
insufficient, say so and state what should be validated in discovery.

Return one JSON object with exactly these keys:
- problem_statement: concise statement of the potential business problem
- supporting_evidence: array of objects with keys fact and source_context
- hypothesis: concise explanation of what may be happening, or an empty string
- product_relevance: how the selected product may relate to the problem
- discovery_questions: array of questions to validate the problem
- evidence_gaps: array of missing facts
- confidence: low, medium, or high

Local prospect, company, saved research, buying signals and product/ICP (untrusted data):
{fence(json.dumps(context, ensure_ascii=False), 'local sales context')}

{UNTRUSTED_RULE}"""
    result = await brain.think_json(
        prompt, session_id=f"problem-statement-{prospect_id}",
        agent="researcher", task="generate_problem_statement_local",
    )
    if not isinstance(result, dict) or not _text(result.get("problem_statement")):
        if getattr(brain, "unavailable", ""):
            raise RuntimeError(f"Local Ollama is unavailable: {brain.unavailable}")
        raise RuntimeError("The local model did not return a valid problem statement. Please retry.")
    evidence = []
    local_facts = _local_fact_strings(context)
    for item in result.get("supporting_evidence", []) if isinstance(result.get("supporting_evidence"), list) else []:
        if not isinstance(item, dict):
            continue
        fact = _text(item.get("fact"), 700)
        # Never persist model-created evidence as if it came from local records.
        if fact and _matches_local_fact(fact, local_facts):
            evidence.append({
                "fact": fact,
                "source_context": _text(item.get("source_context"), 500),
            })
    statement = {
        "prospect_id": prospect_id, "product_id": product_id or "",
        "problem_statement": _text(result.get("problem_statement"), 2000),
        "supporting_evidence": evidence,
        "hypothesis": _text(result.get("hypothesis"), 1500),
        "product_relevance": _text(result.get("product_relevance"), 1500),
        "discovery_questions": _items(result.get("discovery_questions")),
        "evidence_gaps": _items(result.get("evidence_gaps")),
        "confidence": _confidence(result.get("confidence")),
        "model": brain.model, "provider": "ollama_local",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    await state.save_problem_statement(prospect_id, product_id, statement, brain.model)
    logger.info("LOCAL AI problem statement generated prospect=%s product=%s model=%s",
                prospect_id, product_id or "default", brain.model)
    return statement


async def score_prospect(brain, state, prospect_id: str, product_id: str = ""):
    """Score one prospect against one product's ICP using local evidence."""
    _prospect, context = await _context(state, prospect_id, product_id)
    prompt = f"""You are a B2B sales qualification analyst. Score this prospect's fit for the
selected product using only the supplied local records, saved research, buying signals,
and product-specific ICP. Do not invent facts, engagement, budget, authority, or intent.
A missing fact is unknown, not a negative fact. Explain the score and list evidence gaps.

Return one JSON object with exactly these keys:
- score: integer from 0 to 100
- summary: concise explanation of the score
- factors: array of objects with keys name, points, evidence, rationale
- disqualifiers: array of evidence-backed reasons the prospect may not fit
- evidence_gaps: array of unknowns that could change the score
- confidence: low, medium, or high

Use a balanced score across product/industry fit, buyer-role fit, problem/signal fit,
reachability, and evidence quality. Do not award points for information not supplied.

Local prospect, company, saved research, buying signals and product/ICP (untrusted data):
{fence(json.dumps(context, ensure_ascii=False), 'local sales context')}

{UNTRUSTED_RULE}"""
    result = await brain.think_json(
        prompt, session_id=f"prospect-score-{prospect_id}",
        agent="researcher", task="score_prospect_product_fit_local",
    )
    if not isinstance(result, dict) or "score" not in result or not _text(result.get("summary")):
        if getattr(brain, "unavailable", ""):
            raise RuntimeError(f"Local Ollama is unavailable: {brain.unavailable}")
        raise RuntimeError("The local model did not return a valid prospect score. Please retry.")
    score = _score_value(result.get("score"))
    factors = []
    local_facts = _local_fact_strings(context)
    for item in result.get("factors", []) if isinstance(result.get("factors"), list) else []:
        if not isinstance(item, dict) or not _text(item.get("name")):
            continue
        try:
            points = max(0, min(100, int(float(item.get("points", 0)))))
        except (TypeError, ValueError, OverflowError):
            points = 0
        evidence = _text(item.get("evidence"), 700)
        factors.append({
            "name": _text(item.get("name"), 100), "points": points,
            "evidence": evidence if _matches_local_fact(evidence, local_facts) else "",
            "rationale": _text(item.get("rationale"), 1000),
        })
    scored = {
        "prospect_id": prospect_id, "product_id": product_id or "",
        "score": score, "tier": _score_tier(score),
        "summary": _text(result.get("summary"), 2000),
        "factors": factors, "disqualifiers": _items(result.get("disqualifiers")),
        "evidence_gaps": _items(result.get("evidence_gaps")),
        "confidence": _confidence(result.get("confidence")),
        "model": brain.model, "provider": "ollama_local",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    await state.save_prospect_ai_score(prospect_id, product_id, scored, brain.model)
    logger.info("LOCAL AI prospect scored prospect=%s product=%s score=%d model=%s",
                prospect_id, product_id or "default", score, brain.model)
    return scored
