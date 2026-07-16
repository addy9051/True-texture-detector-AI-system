"""Concierge domain logic — system prompt, fabric ontology integration,
diagnosis enrichment.

The interview engine itself is in ``graph.py`` (LangGraph StateGraph).
This module provides the shared domain functions that the graph nodes and
scripts consume:

- ``build_system_prompt()``  — context + interview policy for the LLM
- ``resolve_materials()``    — fiber / weave resolution from a listing
- ``classify_case()``        — 2×2 response-matrix classification
- ``enrich_diagnosis()``     — ground-truth ontology enrichment of the LLM's
                               raw diagnosis payload

The ``ConciergeSession`` re-export is in ``graph.py`` — imported here for
backward compatibility so ``from src.concierge.concierge import ConciergeSession``
still works.
"""

from pathlib import Path

from src.physics.category_materials import likely_materials, normalize_to_ontology
from src.physics.fabric_ontology import FabricOntology

MAX_QUESTIONS = 3

# The four response quadrants (feel = material_issue_suspected,
# weather = weather_suitability_mismatch). Computed deterministically by the
# engine so the dashboard and seller logic never depend on the LLM's wording.
CASE_FEEL_ONLY = "FEEL_ONLY"            # A: defect, weather fine
CASE_WEATHER_ONLY = "WEATHER_ONLY"      # B: no defect, wrong weather
CASE_FEEL_AND_WEATHER = "FEEL_AND_WEATHER"  # C: both
CASE_NO_ISSUE = "NO_ISSUE"              # D: product fine, returned anyway


def classify_case(material_issue, weather_mismatch) -> str:
    feel = bool(material_issue)
    weather = bool(weather_mismatch)  # None (weather not discussed) -> not a mismatch
    if feel and not weather:
        return CASE_FEEL_ONLY
    if not feel and weather:
        return CASE_WEATHER_ONLY
    if feel and weather:
        return CASE_FEEL_AND_WEATHER
    return CASE_NO_ISSUE


# Procedural memory: the interview policy / RESPONSE MATRIX lives in skill.md so
# it can be iterated without touching code. Loaded once, cached.
_SKILL_PATH = Path(__file__).resolve().parent / "skill.md"
_SKILL_BODY = None


def _skill_policy() -> str:
    """Return the procedural policy from skill.md (the section after the '---'
    separator), with {max_questions} filled in."""
    global _SKILL_BODY
    if _SKILL_BODY is None:
        raw = _SKILL_PATH.read_text(encoding="utf-8")
        _SKILL_BODY = raw.split("---", 1)[-1].strip()
    return _SKILL_BODY.replace("{max_questions}", str(MAX_QUESTIONS))


def _ontology_line(m: str, ontology: FabricOntology) -> str:
    spec = ontology.expectations(m) or {}
    return (f"- {m}: genuine feel = {', '.join(spec.get('expected_texture', []))}. "
            f"Thermal = {spec.get('thermal', 'unknown')}; ideal weather = "
            f"{', '.join(spec.get('weather_suitability', [])) or 'unknown'}. "
            f"Red flags = {', '.join(spec.get('failing_adjectives', []))}. "
            f"Common substitution = {', '.join(spec.get('substitution_suspects', [])) or 'none'}.")


def _weave_line(w: str, ontology: FabricOntology) -> str:
    spec = ontology.weave_expectations(w) or {}
    return (f"- {w} (weave): should feel = {', '.join(spec.get('expected_texture', []))}. "
            f"Feels wrong if = {', '.join(spec.get('failing_adjectives', []))}. "
            f"Suited to weather = {', '.join(spec.get('weather_suitability', [])) or 'any'} "
            f"({spec.get('warmth', 'neutral')}).")


def resolve_materials(product: dict, ontology: FabricOntology,
                      category: str | None = None) -> tuple[list[str], list[str], str]:
    """Return (fibers, weaves, prior_note). When the listing states no fiber but
    the category is known, fall back to the category's common-materials prior."""
    fibers = ontology.materials_from_listing(product)
    weaves = ontology.weaves_from_listing(product)
    prior_note = ""
    if not fibers and category:
        prior = likely_materials(category)
        seen = set()
        for mat in prior:
            for f in normalize_to_ontology(mat):
                if f not in seen:
                    seen.add(f)
                    fibers.append(f)
            for w in ontology.weaves_from_listing({"title": mat}):
                if w not in weaves:
                    weaves.append(w)
        if prior:
            prior_note = (
                f"\n\nNO MATERIAL IS LISTED. This is a '{category}', which commonly "
                f"uses: {', '.join(prior[:8])}. Ask the customer which fabric it is "
                f"(or its closest feel) before diagnosing. Likely fibers/weaves below.")
    return fibers, weaves, prior_note


def build_system_prompt(product: dict, ontology: FabricOntology,
                        diagnosis_row: dict | None, category: str | None = None) -> str:
    claimed, weaves, prior_block = resolve_materials(product, ontology, category)
    ontology_lines = [_ontology_line(m, ontology) for m in claimed]
    weave_lines = [_weave_line(w, ontology) for w in weaves]

    evidence_block = "None on file."
    if diagnosis_row:
        complaints = diagnosis_row.get("complaint_adjectives") or []
        sentences = [h["sentence"] for h in diagnosis_row.get("hits", [])
                     if h.get("complaint")][:5]
        if complaints:
            evidence_block = (
                f"Prior customers reported: {', '.join(complaints)}. "
                f"Example quotes: " + " | ".join(f'"{s}"' for s in sentences))

    context = f"""You are a returns assistant for a fashion marketplace. A customer is returning:
  "{(product.get('title') or '')[:140]}"
  Listed materials: {', '.join(claimed) or 'not stated'}.{prior_block}

FABRIC ONTOLOGY (ground truth for the listed / likely materials):
{chr(10).join(ontology_lines) or '- (no ontology entry for the listed materials)'}

WEAVE / CONSTRUCTION (surface feel independent of fiber — a satin should be smooth+glossy, a velvet plush, regardless of fiber):
{chr(10).join(weave_lines) or '- (no specific weave detected)'}

PRIOR EVIDENCE from other customers (INTERNAL — never reveal or quote this to the customer, never put it in question options; use it only to decide which dimension to probe first):
{evidence_block}"""
    return context + "\n\n" + _skill_policy()


def enrich_diagnosis(
    payload: dict,
    ontology: FabricOntology,
    claimed_materials: list[str],
    weaves: list[str],
) -> dict:
    """Ground the LLM's diagnosis with ontology facts the engine holds
    deterministically — the seller always sees claim vs physical truth,
    regardless of what the model chose to mention."""
    return {
        **payload,
        "case_class": classify_case(payload.get("material_issue_suspected"),
                                    payload.get("weather_suitability_mismatch")),
        "claimed_materials": claimed_materials,
        "material_ground_truth": [
            {"material": m,
             "genuine_feel": spec.get("expected_texture", []),
             "thermal": spec.get("thermal"),
             "ideal_weather": spec.get("weather_suitability", []),
             "common_substitutes": spec.get("substitution_suspects", [])}
            for m in claimed_materials
            if (spec := ontology.expectations(m))],
        "weaves": weaves,
        "weave_ground_truth": [
            {"weave": w,
             "should_feel": spec.get("expected_texture", []),
             "feels_wrong_if": spec.get("failing_adjectives", []),
             "warmth": spec.get("warmth")}
            for w in weaves
            if (spec := ontology.weave_expectations(w))],
    }


