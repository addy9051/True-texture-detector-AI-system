"""Diagnose claimed-vs-reported fabric mismatch and hypothesise substitution.

Inputs (produced by Phase 1):
    - product listings with claimed materials (via FabricOntology)
    - texture/feel sentences already filtered by the semantic filter

For each product:
  1. Find, per claimed material, the `failing_adjectives` that actually appear
     in its texture sentences — negation-gated (src/diagnosis/negation.py) and
     rating-gated (a "complaint" is a hit in a <=3-star review).
  2. Score the mismatch from the number of DISTINCT complaint adjectives.
  3. Hypothesise which cheaper fiber it was likely substituted with, using the
     ontology's `substitution_suspects` plus which suspect's expected_texture
     the reported complaints line up with.

Rule-based and explainable on purpose — every flag carries its evidence
sentences, which is the whole point (Amazon's return-reason codes are the
untrustworthy thing we're replacing).
"""

import re
from dataclasses import dataclass, field, asdict

from src.diagnosis.negation import is_negated
from src.physics.fabric_ontology import FabricOntology

COMPLAINT_MAX_RATING = 3.0

# Listing titles that are clearly non-fabric accessories: a fiber word in their
# details usually describes a pouch/case/frame, not the product. Excluded so the
# mismatch shortlist stays about garments.
NON_GARMENT = re.compile(
    r"\b(sunglass|eyeglass|glasses|watch|necklace|bracelet|earring|"
    r"\bring\b|pendant|brooch|keychain|phone case)\b", re.IGNORECASE)


@dataclass
class Hit:
    material: str
    adjective: str
    sentence: str
    rating: float | None
    negated: bool
    complaint: bool

    def to_dict(self):
        return asdict(self)


@dataclass
class Diagnosis:
    parent_asin: str
    title: str
    claimed_materials: list[str]
    n_reviews: int
    n_texture_sentences: int
    complaint_adjectives: list[str] = field(default_factory=list)
    n_complaint_sentences: int = 0
    substitution_hypothesis: dict | None = None
    priority: str = "NONE"
    hits: list[Hit] = field(default_factory=list)
    negated_suppressed: int = 0

    def to_dict(self):
        d = asdict(self)
        return d


def _adj_pattern(adj: str) -> re.Pattern:
    # Match the adjective and simple inflections (scratchy -> scratchier),
    # tolerating a hyphen (paper-thin). Word-boundary anchored.
    return re.compile(rf"\b{re.escape(adj)}\w*", re.IGNORECASE)


def _hits_for_material(material: str, sentences: list[dict],
                       ontology: FabricOntology) -> list[Hit]:
    spec = ontology.expectations(material)
    if not spec:
        return []
    hits: list[Hit] = []
    seen: set[tuple] = set()
    for adj in spec.get("failing_adjectives", []):
        pat = _adj_pattern(adj)
        for s in sentences:
            text = s["sentence"]
            m = pat.search(text)
            if not m:
                continue
            key = (material, adj, text)
            if key in seen:
                continue
            seen.add(key)
            negated = is_negated(text, m.start())
            rating = s.get("rating")
            complaint = (not negated) and ((rating if rating is not None else 5) <= COMPLAINT_MAX_RATING)
            hits.append(Hit(material, adj, text, rating, negated, complaint))
    return hits


def _hypothesise_substitution(material: str, complaint_adjs: set[str],
                              ontology: FabricOntology) -> dict | None:
    """Pick the substitution suspect whose expected feel best matches the
    complaints, falling back to the ontology's declared first suspect."""
    spec = ontology.expectations(material)
    suspects = spec.get("substitution_suspects", []) if spec else []
    if not suspects or not complaint_adjs:
        return None
    scored = []
    for suspect in suspects:
        sub = ontology.expectations(suspect) or {}
        # substitution_signature = the tells a fiber leaves when masquerading as
        # another (cotton reported "shiny"/"slick" -> polyester). Falls back to
        # expected_texture for fibers without a declared signature.
        signature = set(sub.get("substitution_signature") or sub.get("expected_texture", []))
        overlap = complaint_adjs & signature
        scored.append((len(overlap), suspect, sorted(overlap)))
    scored.sort(reverse=True)
    best_overlap, suspect, overlap = scored[0]
    return {
        "suspected_fiber": suspect,
        "matching_signals": overlap,
        "confidence": "corroborated" if best_overlap else "default_suspect",
    }


def _priority(distinct_complaints: int, distinct_sentences: int,
              has_substitution: bool) -> str:
    # CRITICAL demands corroboration from independent sentences: one review
    # saying "rough & scratchy" is one piece of evidence, not two.
    if distinct_complaints >= 2 and distinct_sentences >= 2:
        return "CRITICAL"
    if distinct_complaints >= 2 or (distinct_complaints == 1 and has_substitution):
        return "HIGH"
    if distinct_complaints == 1:
        return "MEDIUM"
    return "NONE"


def diagnose_product(product: dict, sentences: list[dict],
                     ontology: FabricOntology, n_reviews: int) -> Diagnosis:
    title = (product.get("title") or "")
    claimed = ontology.materials_from_listing(product)
    dx = Diagnosis(
        parent_asin=product.get("parent_asin", ""),
        title=title[:140],
        claimed_materials=claimed,
        n_reviews=n_reviews,
        n_texture_sentences=len(sentences),
    )
    if not claimed:
        return dx

    all_hits: list[Hit] = []
    for material in claimed:
        all_hits.extend(_hits_for_material(material, sentences, ontology))
    dx.hits = all_hits
    dx.negated_suppressed = sum(1 for h in all_hits if h.negated)

    complaint_hits = [h for h in all_hits if h.complaint]
    complaint_adjs = {h.adjective for h in complaint_hits}
    dx.complaint_adjectives = sorted(complaint_adjs)
    dx.n_complaint_sentences = len({h.sentence for h in complaint_hits})

    # Substitution hypothesis anchored on the most-complained claimed material.
    if complaint_hits:
        by_material: dict[str, set[str]] = {}
        for h in complaint_hits:
            by_material.setdefault(h.material, set()).add(h.adjective)
        lead_material = max(by_material, key=lambda m: len(by_material[m]))
        dx.substitution_hypothesis = _hypothesise_substitution(
            lead_material, by_material[lead_material], ontology)

    dx.priority = _priority(len(complaint_adjs), dx.n_complaint_sentences,
                            dx.substitution_hypothesis is not None)
    return dx


def is_garment(product: dict) -> bool:
    return not NON_GARMENT.search(product.get("title") or "")
