"""Seed data/processed/seller_insights.jsonl with offline mock concierge
sessions so the Phase-5 dashboard has rows to render while AWS is blocked.

Runs the REAL ConciergeSession engine against MockBedrockChat with scripted
customer answers — so it also serves as an integration test of the session
machinery. Rows carry model_id "mock.offline-concierge".

    uv run python scripts/seed_mock_insights.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.concierge.concierge import ConciergeSession
from src.concierge.mock_chat import MockBedrockChat
from src.physics.fabric_ontology import FabricOntology

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "NONE": 3}

SCRIPTED_ANSWERS = [
    ["How the fabric feels",
     "it looks shiny and feels like cheap plastic, not cotton at all"],
    ["How the fabric feels",
     "rough and scratchy against my skin, could not wear it for an hour"],
    ["How the fabric feels",
     "way too thin and see-through, nothing like the thick knit in the photos"],
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Missing {path} — run Phases 1-2 first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    products = {p["parent_asin"]: p for p in read_jsonl(RAW / "products.jsonl")}
    flagged = sorted((d for d in read_jsonl(PROCESSED / "diagnosis.jsonl")
                      if d["priority"] != "NONE"),
                     key=lambda d: PRIORITY_ORDER[d["priority"]])
    ontology = FabricOntology()

    rows = []
    for diag, answers in zip(flagged, SCRIPTED_ANSWERS):
        asin = diag["parent_asin"]
        product = products[asin]
        chat = MockBedrockChat()
        session = ConciergeSession(chat, product, ontology, diag)
        event = session.start()
        for answer in answers:
            if event["type"] != "question":
                break
            event = session.answer(answer)
        assert event["type"] == "diagnosis", f"session did not converge for {asin}"
        rows.append({
            "parent_asin": asin,
            "title": (product.get("title") or "")[:140],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "diagnosis": event["data"],
            "transcript": session.transcript,
            "cost_usd": 0.0,
            "model_id": chat.model_id,
        })
        print(f"seeded session for [{diag['priority']}] {rows[-1]['title'][:70]}")

    from src.concierge.insights_store import save_session
    for row in rows:
        save_session(row)
    print(f"Saved {len(rows)} mock sessions to insights.sqlite")


if __name__ == "__main__":
    main()
