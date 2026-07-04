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

from src.concierge.concierge import ConciergeSession
from src.concierge.provider import make_chat, provider_name
from src.physics.fabric_ontology import FabricOntology

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "NONE": 3}

# Scenarios spanning the 2x2 response matrix (feel x weather):
#   substitution -> CASE A: shiny/plastic feel, weather fine -> SUPPLY_CHAIN_AUDIT
#   quality      -> CASE A: rough/coarse cotton, weather fine -> QUALITY_IMPROVEMENT
#   weather      -> CASE B: no feel issue, wrong weather -> weather education
#   both         -> CASE C: shiny/plastic feel AND wrong weather
#   neither      -> CASE D: product fine, returned anyway -> NO_ACTION (threshold)
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
    # Non-cotton material check: wool is ideal for COLD, so a HOT-weather
    # complaint should be the weather mismatch (opposite of the cotton cases).
    # Pair with --asin for a wool product.
    "wool_weather": [
        "These weren't comfortable for me",
        "the wool felt fine actually, warm and soft as I expected",
        "but I wore them on a hot humid summer day and my feet were sweating and overheating",
    ],
    "wool_feel": [
        "The material felt off",
        "they felt squeaky and plasticky, scratchy — not like real wool at all",
        "just normal wear in cool weather, which is what I bought them for",
    ],
    # Weave check: fleece should feel soft/fuzzy/warm; reporting it rough and
    # thin is a WEAVE-level mismatch (not a fiber substitution). Use --asin for
    # a fleece product (e.g. B09BC2STXP).
    "weave_feel": [
        "The material wasn't what I expected",
        "the fleece felt rough and thin and scratchy, not soft or fuzzy or warm at all",
        "wore it indoors in cool weather, so weather was fine",
    ],
    # Weave-driven WEATHER: fleece is a warm/cold-weather weave. Worn in hot
    # weather it should flag a weather mismatch EVEN THOUGH the cotton fiber is
    # hot-ideal — the construction overrides the fiber for warmth.
    "weave_weather": [
        "It was too uncomfortable to wear",
        "the fleece itself felt soft and fine, no complaint about the feel",
        "but I wore it on a hot humid afternoon and was overheating and sweating badly",
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

    chat = make_chat()
    print(f"provider={provider_name()}  model={chat.model_id}  scenario={args.scenario}")
    print(f"RETURNING: {(product.get('title') or '')[:80]}\n")

    session = ConciergeSession(chat, product, FabricOntology(), diag)
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
          f"{session.questions_asked} | cost: {chat.meter.summary()}")

    from src.concierge.insights_store import save_session
    save_session({
        "parent_asin": asin,
        "title": (product.get("title") or "")[:140],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnosis": dx,
        "transcript": session.transcript,
        "cost_usd": round(chat.meter.usd, 6),
        "model_id": chat.model_id,
    })
    print("Row saved to insights.sqlite")


if __name__ == "__main__":
    main()
