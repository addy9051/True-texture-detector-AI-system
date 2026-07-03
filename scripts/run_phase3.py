"""Phase 3: visually corroborate the Phase-2 flagged products.

Downloads official + review photos for FLAGGED products only (priority !=
NONE, ~10 products, a few dozen images cached in data/images/), compares them
with CLIP + color histograms, and writes data/processed/visual_audit.jsonl.
The dashboard picks the results up automatically.

First run downloads the CLIP model (~350MB) into the HF cache.

    uv run python scripts/run_phase3.py
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visual.clip_audit import (ClipAuditor, HAVE_REMBG, fetch_image,
                                   official_urls, review_urls)

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
IMAGES = ROOT / "data" / "images"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Missing {path} — run Phases 1-2 first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_set(auditor, diagnoses, products, reviews_by_asin) -> list[dict]:
    results = []
    for d in diagnoses:
        asin = d["parent_asin"]
        product = products[asin]
        cache = IMAGES / asin
        official = [img for j, u in enumerate(official_urls(product))
                    if (img := fetch_image(u, cache / f"official_{j}.jpg"))]
        review = [img for j, u in enumerate(review_urls(reviews_by_asin[asin]))
                  if (img := fetch_image(u, cache / f"review_{j}.jpg"))]
        if not official or not review:
            print(f"[skip] {asin}: official={len(official)} review={len(review)} images")
            continue
        row = {"parent_asin": asin, "title": d["title"][:80],
               "n_official": len(official), "n_review": len(review),
               **auditor.audit(official, review)}
        results.append(row)
        print(f"[{row['visual_corroboration']:12}] clip={row['clip_similarity_mean']:.2f} "
              f"color_delta={row['color_delta_mean']:.2f}  {d['title'][:60]}")
    return results


def summarize(label: str, rows: list[dict]):
    if not rows:
        return
    clip = sum(r["clip_similarity_mean"] for r in rows) / len(rows)
    color = sum(r["color_delta_mean"] for r in rows) / len(rows)
    print(f"  {label:8} n={len(rows):3}  clip_mean={clip:.3f}  color_delta_mean={color:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=int, default=12,
                        help="Also audit N unflagged (priority NONE) products as a "
                             "control group — if their scores match the flagged "
                             "group's, the visual signal is not discriminating")
    args = parser.parse_args()

    products = {p["parent_asin"]: p for p in read_jsonl(RAW / "products.jsonl")}
    reviews_by_asin = defaultdict(list)
    for r in read_jsonl(RAW / "reviews.jsonl"):
        reviews_by_asin[r["parent_asin"]].append(r)
    diagnoses = read_jsonl(PROCESSED / "diagnosis.jsonl")
    flagged = [d for d in diagnoses if d["priority"] != "NONE"]
    clean = [d for d in diagnoses
             if d["priority"] == "NONE" and not d.get("complaint_adjectives")]
    control = random.Random(42).sample(clean, min(args.control, len(clean)))
    print(f"Auditing {len(flagged)} flagged + {len(control)} control products "
          f"(background removal: {'rembg' if HAVE_REMBG else 'off — full frame'})")

    auditor = ClipAuditor()
    print("--- flagged ---")
    results = audit_set(auditor, flagged, products, reviews_by_asin)
    print("--- control (no texture complaints) ---")
    control_results = audit_set(auditor, control, products, reviews_by_asin)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / "visual_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (PROCESSED / "visual_audit_control.jsonl").open("w", encoding="utf-8") as f:
        for row in control_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nCalibration comparison (do the groups separate?):")
    summarize("flagged", results)
    summarize("control", control_results)
    print(f"\nWrote {len(results)} + {len(control_results)} audits to data/processed/")


if __name__ == "__main__":
    main()
