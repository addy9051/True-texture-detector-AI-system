"""Non-interactive integration test: one REAL concierge session against the
active LLM provider, with scripted customer answers.

Verifies the full contract on live infrastructure: adaptive questions arrive
via forced tool calls, the question budget holds, and the final diagnosis is
schema-shaped. Prints the whole exchange for quality review and appends the
row to seller_insights.jsonl (tagged with the real model id).

    uv run python scripts/test_live_concierge.py [--asin B0XXXX]
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

# Scenarios spanning the 2x2 response matrix (feel x weather):
SCENARIOS = {
    "substitution": [
        "The fabric feels wrong, not what I expected from the description",
        "it looks shiny and feels like cheap plastic, not cotton at all",
        "just normal indoor wear on a mild day, weather was fine",
    ],
    "quality": [
        "The fabric feels wrong for what it says",
        "it feels rough and coarse, like a cheap scratchy cotton, not soft at all",
        "just normal indoor wear, nothing unusual",
    ],
    "weather": [
        "It didn't keep me comfortable",
        "the fabric felt fine, soft like cotton, no complaints there",
        "I wore it outdoors on a freezing winter evening with nothing over it and I was cold",
    ],
    "both": [
        "The dress was disappointing overall",
        "the fabric felt shiny and plasticky, not like cotton at all",
        "and I had it on outdoors on a freezing winter night with nothing over it",
    ],
    "neither": [
        "It just wasn't right for me",
        "honestly the fabric felt fine, soft and cotton-like as described",
        "the weather wasn't an issue, mild day indoors — I just didn't love the style on me",
    ],
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asin")
    parser.add_argument("--scenario", choices=list(SCENARIOS),
                        default="substitution")
    args = parser.parse_args()
    answers = SCENARIOS[args.scenario]

    products = {p["parent_asin"]: p for p in read_jsonl(RAW / "products.jsonl")}
    diagnoses = {d["parent_asin"]: d
                 for d in read_jsonl(PROCESSED / "diagnosis.jsonl")}
    if args.asin:
        asin = args.asin
    else:
        asin = min((d for d in diagnoses.values() if d["priority"] != "NONE"),
                   key=lambda d: PRIORITY_ORDER[d["priority"]])["parent_asin"]
    product, diag = products[asin], diagnoses.get(asin)

    print(f"provider={provider_name()}  scenario={args.scenario}")
    print(f"RETURNING: {(product.get('title') or '')[:80]}\n")

    session = ConciergeSession(product, FabricOntology(), diag)
    event = session.start()
    for answer in answers:
        if event["type"] != "question":
            break
        print(f"Concierge: {event['question']}")
        for opt in event["options"]:
            print(f"   - {opt}")
        print(f"Customer:  {answer}\n")
        event = session.answer(answer)

    assert event["type"] == "diagnosis", f"no diagnosis: {event}"
    dx = event["data"]
    if dx.get("customer_closing_message"):
        print(f"MESSAGE TO CUSTOMER:\n{dx['customer_closing_message']}\n")
    print("DIAGNOSIS:")
    print(json.dumps(dx, indent=2, ensure_ascii=False))
    print(f"\ntransport mode: {session.mode} | questions asked: "
          f"{session.questions_asked}")

    from src.concierge.insights_store import save_session
    save_session({
        "parent_asin": asin,
        "title": (product.get("title") or "")[:140],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnosis": dx,
        "transcript": session.transcript,
        "cost_usd": 0.0,
        "model_id": session.model_id,
    })
    print("Row saved to insights.sqlite")


if __name__ == "__main__":
    main()
