"""Interactive Returns Concierge demo (terminal chat).

Simulates a customer returning one of the sampled products. The concierge asks
1-3 adaptive questions (options are derived from the fabric ontology + steered
by Phase-2 evidence), then emits the structured diagnosis that would flow to
the seller dashboard.

    uv run python scripts/run_concierge.py            # first HIGH-priority product
    uv run python scripts/run_concierge.py --list     # show flagged products
    uv run python scripts/run_concierge.py --asin B0XXXXXXX

Saves each completed session to the SQLite insights store (episodic memory).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows consoles default to cp1252 and choke on chars models sometimes emit
# (e.g. non-breaking hyphen U+2011). Force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.concierge.graph import ConciergeSession
from src.concierge.portkey_llm import provider_name
from src.physics.fabric_ontology import FabricOntology

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "NONE": 3}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Missing {path} — run Phases 1-2 first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_product(products, diag_by_asin, asin_arg):
    if asin_arg:
        if asin_arg not in products:
            sys.exit(f"ASIN {asin_arg} not in data/raw/products.jsonl")
        return asin_arg
    flagged = sorted((d for d in diag_by_asin.values() if d["priority"] != "NONE"),
                     key=lambda d: PRIORITY_ORDER[d["priority"]])
    if not flagged:
        sys.exit("No flagged products in diagnosis.jsonl.")
    return flagged[0]["parent_asin"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asin")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--mock", action="store_true",
                        help="Offline canned model — exercises the full pipeline "
                             "without any cloud calls (responses are scripted, not AI)")
    args = parser.parse_args()

    products = {p["parent_asin"]: p for p in read_jsonl(RAW / "products.jsonl")}
    diag_by_asin = {d["parent_asin"]: d
                    for d in read_jsonl(PROCESSED / "diagnosis.jsonl")}

    if args.list:
        for d in sorted(diag_by_asin.values(),
                        key=lambda d: PRIORITY_ORDER[d["priority"]]):
            if d["priority"] != "NONE":
                print(f"[{d['priority']:8}] {d['parent_asin']}  {d['title'][:90]}")
        return

    asin = pick_product(products, diag_by_asin, args.asin)
    product = products[asin]
    diagnosis_row = diag_by_asin.get(asin)

    print("=" * 72)
    print(f"RETURNING: {(product.get('title') or '')[:100]}")
    print(f"ASIN {asin} | Phase-2 priority: "
          f"{diagnosis_row['priority'] if diagnosis_row else 'n/a'}")
    print("=" * 72)

    provider = "mock" if args.mock else None
    session = ConciergeSession(product, FabricOntology(), diagnosis_row,
                               provider=provider)
    if provider == "mock":
        print("(MOCK MODE — scripted responses, no cloud call is made)")
    else:
        print(f"(provider: {provider_name()} via Portkey AI gateway)")

    event = session.start()

    while event["type"] == "question":
        print(f"\nConcierge: {event['question']}")
        for i, opt in enumerate(event["options"], 1):
            print(f"  {i}. {opt}")
        raw = input("Your answer (number or free text): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(event["options"]):
            raw = event["options"][int(raw) - 1]
        event = session.answer(raw)

    dx = event["data"]
    closing = dx.get("customer_closing_message")
    if closing:
        print("\n" + "=" * 72)
        print("MESSAGE TO CUSTOMER")
        print(closing)
    print("\n" + "=" * 72)
    print("STRUCTURED DIAGNOSIS (what the seller dashboard receives)")
    print(json.dumps(dx, indent=2, ensure_ascii=False))
    print(f"\nModel: {session.model_id} | Questions: {session.questions_asked}")

    from src.concierge.insights_store import save_session, DB_PATH
    save_session({
        "parent_asin": asin,
        "title": (product.get("title") or "")[:140],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnosis": dx,
        "transcript": session.transcript,
        "cost_usd": 0.0,
        "model_id": session.model_id,
    })
    print(f"Saved to {DB_PATH}")


if __name__ == "__main__":
    main()
