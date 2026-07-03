"""Phase 1 end-to-end: dataset sample -> texture sentences -> mismatch shortlist.

Run from the project root AFTER `python -m src.ingest.download_dataset`:
    python scripts/run_phase1.py

Outputs:
    data/processed/texture_sentences.jsonl  every fabric-feel sentence found
    data/processed/texture_report.jsonl     per-product evidence summary
and prints the top products ranked by texture-complaint density.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp.semantic_filter import TextureFilter
from src.physics.fabric_ontology import FabricOntology

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Missing {path} — run `python -m src.ingest.download_dataset` first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    reviews = read_jsonl(RAW / "reviews.jsonl")
    products = {p["parent_asin"]: p for p in read_jsonl(RAW / "products.jsonl")}
    print(f"Loaded {len(reviews)} reviews across {len(products)} products")

    texture_sentences = TextureFilter().filter_reviews(reviews)
    print(f"Extracted {len(texture_sentences)} texture/feel sentences")

    by_asin = defaultdict(list)
    for ts in texture_sentences:
        by_asin[ts.parent_asin].append(ts)

    ontology = FabricOntology()
    n_reviews = defaultdict(int)
    for r in reviews:
        n_reviews[r["parent_asin"]] += 1

    report = []
    for asin, sentences in by_asin.items():
        product = products.get(asin, {})
        claimed = ontology.materials_from_listing(product)
        negative = [ts for ts in sentences if (ts.rating or 5) <= 3]
        mismatch_hits = []
        for material in claimed:
            for adj, sentence in ontology.failing_hits(material, [ts.sentence for ts in sentences]):
                mismatch_hits.append({"claimed_material": material,
                                      "failing_adjective": adj,
                                      "evidence": sentence})
        report.append({
            "parent_asin": asin,
            "title": (product.get("title") or "")[:120],
            "claimed_materials": claimed,
            "n_reviews": n_reviews[asin],
            "n_texture_sentences": len(sentences),
            "n_negative_texture_sentences": len(negative),
            "complaint_density": round(len(negative) / max(n_reviews[asin], 1), 3),
            "mismatch_hits": mismatch_hits,
        })

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / "texture_sentences.jsonl").open("w", encoding="utf-8") as f:
        for ts in texture_sentences:
            f.write(json.dumps(ts.to_dict(), ensure_ascii=False) + "\n")
    with (PROCESSED / "texture_report.jsonl").open("w", encoding="utf-8") as f:
        for row in report:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    flagged = sorted((r for r in report if r["mismatch_hits"]),
                     key=lambda r: r["complaint_density"], reverse=True)
    print(f"\n{len(flagged)} products have claimed-material mismatch evidence. Top 10:")
    for r in flagged[:10]:
        materials = ", ".join(r["claimed_materials"]) or "unknown"
        print(f"  [{r['complaint_density']:.0%} complaint density] {r['title']}")
        print(f"    claims: {materials} | example: {r['mismatch_hits'][0]['evidence'][:110]}")


if __name__ == "__main__":
    main()
