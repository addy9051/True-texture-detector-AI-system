"""Calibrate the texture-filter similarity threshold on real data.

Runs the filter with a low floor (0.25) so sub-threshold sentences are visible,
buckets everything by similarity band, and prints counts plus a seeded random
sample per band. Judge each band's precision by eye ("is this sentence really
about fabric feel?") and set TextureFilter's default threshold at the lowest
band that is still mostly on-topic.

Usage (after download_dataset):
    uv run python scripts/calibrate_threshold.py

Writes data/processed/calibration_samples.json for later reference.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp.semantic_filter import TextureFilter, split_sentences

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

BANDS = [(0.25, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50),
         (0.50, 0.55), (0.55, 0.60), (0.60, 1.01)]
SAMPLES_PER_BAND = 12
FLOOR = 0.25


def main():
    reviews_path = RAW / "reviews.jsonl"
    if not reviews_path.exists():
        sys.exit(f"Missing {reviews_path} — run the downloader first.")
    with reviews_path.open(encoding="utf-8") as f:
        reviews = [json.loads(line) for line in f if line.strip()]

    total_sentences = sum(len(split_sentences(r.get("text", ""))) for r in reviews)
    print(f"{len(reviews)} reviews -> {total_sentences} candidate sentences")

    hits = TextureFilter(threshold=FLOOR).filter_reviews(reviews)
    print(f"{len(hits)} sentences scored >= {FLOOR}\n")

    by_band = defaultdict(list)
    for ts in hits:
        for lo, hi in BANDS:
            if lo <= ts.similarity < hi:
                by_band[(lo, hi)].append(ts)
                break

    rng = random.Random(42)
    samples = {}
    for lo, hi in BANDS:
        band = by_band[(lo, hi)]
        share = len(band) / max(total_sentences, 1)
        label = f"[{lo:.2f}-{hi:.2f})"
        print(f"=== {label}  {len(band)} sentences ({share:.1%} of all) ===")
        picked = rng.sample(band, min(SAMPLES_PER_BAND, len(band)))
        samples[label] = [ts.to_dict() for ts in picked]
        for ts in picked:
            print(f"  ({ts.similarity:.2f}) {ts.sentence[:160]}")
        print()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "calibration_samples.json"
    out.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
