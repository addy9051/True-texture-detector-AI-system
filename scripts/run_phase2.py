"""Phase 2: turn Phase-1 texture sentences into a ranked supplier-audit shortlist.

Reads the cached Phase-1 outputs (no re-embedding, runs in seconds):
    data/processed/texture_sentences.jsonl   filtered fabric-feel sentences
    data/raw/products.jsonl                   listings (for claimed materials)
    data/raw/reviews.jsonl                    (only to count reviews per product)

Writes:
    data/processed/diagnosis.jsonl            full per-product diagnosis
and prints the CRITICAL/HIGH shortlist with evidence + substitution hypothesis.

Run:
    uv run python scripts/run_phase2.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnosis.diagnose import diagnose_product, is_garment
from src.physics.fabric_ontology import FabricOntology

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "NONE": 3}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Missing {path} — run Phase 1 first (download_dataset + run_phase1).")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    sentences = read_jsonl(PROCESSED / "texture_sentences.jsonl")
    products = {p["parent_asin"]: p for p in read_jsonl(RAW / "products.jsonl")}
    review_counts = Counter(r["parent_asin"] for r in read_jsonl(RAW / "reviews.jsonl"))

    by_asin: dict[str, list[dict]] = defaultdict(list)
    for s in sentences:
        by_asin[s["parent_asin"]].append(s)

    ontology = FabricOntology()
    diagnoses = []
    skipped_non_garment = 0
    for asin, product in products.items():
        if not is_garment(product):
            skipped_non_garment += 1
            continue
        dx = diagnose_product(product, by_asin.get(asin, []), ontology,
                              review_counts.get(asin, 0))
        diagnoses.append(dx)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / "diagnosis.jsonl").open("w", encoding="utf-8") as f:
        for dx in diagnoses:
            f.write(json.dumps(dx.to_dict(), ensure_ascii=False) + "\n")

    flagged = [d for d in diagnoses if d.priority != "NONE"]
    flagged.sort(key=lambda d: (PRIORITY_ORDER[d.priority], -d.n_complaint_sentences,
                                -len(d.complaint_adjectives)))
    counts = Counter(d.priority for d in diagnoses)
    total_suppressed = sum(d.negated_suppressed for d in diagnoses)

    print(f"Diagnosed {len(diagnoses)} garment products "
          f"({skipped_non_garment} non-garment skipped)")
    print(f"Priority mix: CRITICAL={counts['CRITICAL']} HIGH={counts['HIGH']} "
          f"MEDIUM={counts['MEDIUM']} NONE={counts['NONE']}")
    print(f"Negation-suppressed false hits (would have been flagged in Phase 1): "
          f"{total_suppressed}\n")

    print("=== CRITICAL / HIGH shortlist ===")
    for d in flagged:
        if d.priority not in ("CRITICAL", "HIGH"):
            break
        sub = d.substitution_hypothesis
        sub_txt = ""
        if sub:
            sig = ", ".join(sub["matching_signals"]) or "declared default"
            sub_txt = f" -> likely {sub['suspected_fiber']} ({sub['confidence']}: {sig})"
        print(f"\n[{d.priority}] {d.title}")
        print(f"  claims: {', '.join(d.claimed_materials)} | "
              f"complaints: {', '.join(d.complaint_adjectives)} "
              f"({d.n_complaint_sentences} independent sentence(s)){sub_txt}")
        # One line per evidence sentence, all matched adjectives grouped —
        # the same sentence can hit several claimed materials.
        evidence: dict[str, dict] = {}
        for h in d.hits:
            if h.complaint:
                e = evidence.setdefault(h.sentence, {"rating": h.rating, "adjs": set()})
                e["adjs"].add(h.adjective)
        for sentence, e in evidence.items():
            print(f"    - \"{sentence[:120]}\"  ({e['rating']}★, matched: "
                  f"{', '.join(sorted(e['adjs']))})")

    print(f"\nFull diagnosis written to {PROCESSED / 'diagnosis.jsonl'}")


if __name__ == "__main__":
    main()
