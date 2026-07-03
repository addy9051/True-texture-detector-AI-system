"""Download a working sample of the Amazon Reviews 2023 dataset (fashion category).

Source: McAuley Lab, UCSD — https://amazon-reviews-2023.github.io/ — served via
Hugging Face. This is a public research dataset, so no scraping of amazon.com
and no ToS problems (this replaces the BeautifulSoup/Selenium step in the
original plan).

The raw files are plain JSONL on the hub. We download each file once with
hf_hub_download (resumable, retry-friendly) and then parse locally — streaming
the 1GB file over HTTP twice proved fragile on a flaky connection.

Two-pass strategy over the local file (the category has ~2.5M reviews):
    pass 1: count photo-reviews per product, pick the top --max-products
    pass 2: collect up to 60 reviews for just those products

Keeps only products that end up with BOTH official listing images (metadata)
and user review photos, because the Phase 3 visual audit needs both sides.

Usage (from the project root):
    uv run python -m src.ingest.download_dataset --max-products 300

Outputs:
    data/raw/reviews.jsonl   one review per line
    data/raw/products.jsonl  one product (listing metadata) per line
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "McAuley-Lab/Amazon-Reviews-2023"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
HF_DIR = DATA_DIR / "hf"


def fetch(filename: str) -> Path:
    """Download (or reuse) a dataset file into data/hf/. Resumable."""
    print(f"Fetching {filename} ...", flush=True)
    return Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
                                filename=filename, local_dir=HF_DIR))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def pick_products(reviews_file: Path, max_products: int, min_image_reviews: int,
                  max_scan: int) -> list[str]:
    """Pass 1: count reviews-with-photos per product, return the best ASINs."""
    image_reviews = Counter()
    scanned = 0
    for r in iter_jsonl(reviews_file):
        scanned += 1
        if scanned > max_scan:
            break
        if r.get("images") and (r.get("text") or "").strip() and r.get("parent_asin"):
            image_reviews[r["parent_asin"]] += 1
        if scanned % 500_000 == 0:
            qualified = sum(1 for c in image_reviews.values() if c >= min_image_reviews)
            print(f"  pass 1: scanned {scanned:,} | {qualified} products qualified", flush=True)
    ranked = [asin for asin, c in image_reviews.most_common()
              if c >= min_image_reviews][:max_products]
    print(f"Pass 1 done: scanned {scanned:,} reviews, selected {len(ranked)} products", flush=True)
    return ranked


def collect_reviews(reviews_file: Path, asins: set[str], max_scan: int,
                    max_reviews_per_product: int = 60) -> dict[str, list[dict]]:
    """Pass 2: collect reviews for the selected products only."""
    by_asin: dict[str, list[dict]] = defaultdict(list)
    scanned = 0
    for r in iter_jsonl(reviews_file):
        scanned += 1
        if scanned > max_scan:
            break
        asin = r.get("parent_asin")
        text = (r.get("text") or "").strip()
        if asin not in asins or not text or len(by_asin[asin]) >= max_reviews_per_product:
            continue
        by_asin[asin].append({
            "parent_asin": asin,
            "rating": r.get("rating"),
            "title": r.get("title"),
            "text": text,
            "images": r.get("images") or [],
            "verified_purchase": r.get("verified_purchase"),
            "helpful_vote": r.get("helpful_vote"),
        })
    print(f"Pass 2 done: {sum(len(v) for v in by_asin.values())} reviews "
          f"for {len(by_asin)} products", flush=True)
    return by_asin


def has_official_images(images) -> bool:
    if isinstance(images, dict):
        return any(images.values())
    return bool(images)


def collect_metadata(meta_file: Path, asins: set[str]) -> dict[str, dict]:
    """Scan listing metadata for the selected products."""
    out: dict[str, dict] = {}
    for m in iter_jsonl(meta_file):
        asin = m.get("parent_asin")
        if asin not in asins:
            continue
        details = m.get("details") or {}
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except json.JSONDecodeError:
                details = {"raw": details}
        out[asin] = {
            "parent_asin": asin,
            "title": m.get("title"),
            "store": m.get("store"),
            "main_category": m.get("main_category"),
            "average_rating": m.get("average_rating"),
            "rating_number": m.get("rating_number"),
            "price": m.get("price"),
            "features": m.get("features") or [],
            "description": m.get("description") or [],
            "images": m.get("images") or [],
            "details": details,
        }
        if len(out) == len(asins):
            break
    print(f"Collected metadata for {len(out)}/{len(asins)} products", flush=True)
    return out


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Amazon_Fashion",
                        help="Category file name (Amazon_Fashion is ~2.5M reviews / 1GB; "
                             "Clothing_Shoes_and_Jewelry is ~66M / 28GB — much slower)")
    parser.add_argument("--max-products", type=int, default=300)
    parser.add_argument("--min-image-reviews", type=int, default=3,
                        help="Minimum user reviews with photos for a product to qualify")
    parser.add_argument("--max-scan", type=int, default=3_000_000,
                        help="Cap on reviews scanned per pass")
    args = parser.parse_args()

    reviews_file = fetch(f"raw/review_categories/{args.category}.jsonl")
    meta_file = fetch(f"raw/meta_categories/meta_{args.category}.jsonl")

    selected = pick_products(reviews_file, args.max_products,
                             args.min_image_reviews, args.max_scan)
    reviews_by_asin = collect_reviews(reviews_file, set(selected), args.max_scan)
    products = collect_metadata(meta_file, set(reviews_by_asin))

    # Drop products whose listing has no official images — the visual audit needs them.
    products = {a: p for a, p in products.items() if has_official_images(p["images"])}
    reviews = [r for a in products for r in reviews_by_asin[a]]

    write_jsonl(RAW_DIR / "reviews.jsonl", reviews)
    write_jsonl(RAW_DIR / "products.jsonl", products.values())
    print(f"Final sample: {len(products)} products, {len(reviews)} reviews", flush=True)


if __name__ == "__main__":
    main()
