"""Guided end-to-end demo of the True-Texture pipeline. No AWS required.

Replays the pipeline's REAL computed results from data/processed/ (nothing is
invented for the demo), then auto-plays a mock concierge session. Takes about
a minute to read top to bottom.

    uv run python scripts/demo.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
WIDTH = 74


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def section(title: str):
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def need(rows, name, hint):
    if not rows:
        sys.exit(f"Missing {name} -- run `{hint}` first.")
    return rows


def main():
    section("THE PROBLEM -- why fabric mismatch is worth solving")
    print("""India's fashion ecommerce (~$21.6B in 2025, the largest category of a
~$70B market) returns 25-35% of what it ships. Unlike the US -- where fit
dominates and 'not as described' is ~10-13% of returns -- 'product is
different (color, FABRIC, design)' is the #1 stated return reason on
value-fashion platforms. A single return (Rs.200-400 all-in) erases the
margin of 3-5 good orders. Return-reason codes are unreliable; producing
trustworthy root-cause data is itself the product.""")

    reviews = need(read_jsonl(RAW / "reviews.jsonl"), "reviews",
                   "uv run python -m src.ingest.download_dataset")
    products = need(read_jsonl(RAW / "products.jsonl"), "products",
                    "uv run python -m src.ingest.download_dataset")
    sentences = need(read_jsonl(PROCESSED / "texture_sentences.jsonl"),
                     "texture sentences", "uv run python scripts/run_phase1.py")
    section("PHASE 1 -- evidence engine (public data, local models, $0)")
    print(f"""Scanned 2.5M Amazon Fashion reviews (McAuley Lab public dataset -- no
scraping) and sampled the {len(products)} products richest in photo evidence:
{len(reviews):,} reviews. A MiniLM aspect filter extracted {len(sentences):,} sentences
about fabric feel. The 0.50 similarity threshold was CALIBRATED on 48,040
real sentences: precision jumps ~50% -> ~90% at that score band.""")

    diagnoses = need(read_jsonl(PROCESSED / "diagnosis.jsonl"), "diagnoses",
                     "uv run python scripts/run_phase2.py")
    flagged = [d for d in diagnoses if d["priority"] != "NONE"]
    prio = Counter(d["priority"] for d in diagnoses)
    suppressed = sum(d.get("negated_suppressed", 0) for d in diagnoses)
    section("PHASE 2 -- diagnosis engine (claimed fiber vs reported feel)")
    print(f"""{len(diagnoses)} garments diagnosed against the fabric ontology.
Priorities: CRITICAL={prio['CRITICAL']}  HIGH={prio['HIGH']}  MEDIUM={prio['MEDIUM']}
Negation gate suppressed {suppressed} would-be false flags ("it was NOT scratchy").
A complaint only counts from a <=3-star review; CRITICAL needs >=2 independent
complaint sentences.\n""")
    star = next((d for d in flagged
                 if (d.get("substitution_hypothesis") or {}).get("confidence")
                 == "corroborated"), flagged[0] if flagged else None)
    if star:
        hyp = star.get("substitution_hypothesis") or {}
        print(f"Showcase flag [{star['priority']}]: {star['title'][:65]}")
        print(f"  claims {', '.join(star['claimed_materials'])} | "
              f"complaints: {', '.join(star['complaint_adjectives'])}"
              + (f" -> likely {hyp['suspected_fiber']} "
                 f"({', '.join(hyp.get('matching_signals', [])) or 'default suspect'})"
                 if hyp else ""))
        for h in star.get("hits", []):
            if h.get("complaint"):
                print(f"  evidence: \"{h['sentence'][:90]}\" ({h.get('rating')}*)")
                break

    visual = read_jsonl(PROCESSED / "visual_audit.jsonl")
    control = read_jsonl(PROCESSED / "visual_audit_control.jsonl")
    section("PHASE 3 -- visual audit: an honest negative result")
    if visual and control:
        vm = sum(r["clip_similarity_mean"] for r in visual) / len(visual)
        vc = sum(r["color_delta_mean"] for r in visual) / len(visual)
        cm = sum(r["clip_similarity_mean"] for r in control) / len(control)
        cc = sum(r["color_delta_mean"] for r in control) / len(control)
        crop = "garment crops (rembg)" if visual[0].get("background_removed") \
            else "whole frames"
        print(f"""We tested the "compare listing photos to customer photos with CLIP" idea
properly -- WITH a control group of products nobody complains about ({crop}):
    flagged  (n={len(visual):2}): clip={vm:.3f}  color_delta={vc:.3f}
    control  (n={len(control):2}): clip={cm:.3f}  color_delta={cc:.3f}
Verdict distribution: {dict(Counter(r['visual_corroboration'] for r in visual))}
Overlapping distributions killed the naive 'flag if CLIP < 0.75' heuristic
(it fires on every product -- studio-vs-phone gap dominates). Verdicts are now
conservative: only control-band outliers count as visual support.""")
    else:
        print("Run `uv run python scripts/run_phase3.py` to reproduce this experiment.")

    section("PHASE 4 -- returns concierge (auto-played mock session)")
    from src.concierge.graph import ConciergeSession
    from src.concierge.mock_chat import MockChatModel
    from src.physics.fabric_ontology import FabricOntology
    target = star or flagged[0]
    product = next(p for p in products if p["parent_asin"] == target["parent_asin"])
    session = ConciergeSession(product, FabricOntology(), target, provider="mock")
    scripted = ["How the fabric feels",
                "it looks shiny and feels like cheap plastic, not cotton at all"]
    print(f"Returning: {target['title'][:65]}\n")
    event = session.start()
    for answer in scripted:
        if event["type"] != "question":
            break
        print(f"  Concierge: {event['question']}")
        print(f"             options: {', '.join(event['options'])}")
        print(f"  Customer:  {answer}\n")
        event = session.answer(answer)
    print("  Structured diagnosis -> seller dashboard:")
    print("  " + json.dumps(event["data"], indent=2).replace("\n", "\n  "))
    print("\n(Scripted mock -- live Bedrock path is one env change away once the")
    print(" AISPL account restriction is lifted; same engine, same output schema.)")

    from src.concierge.insights_store import load_sessions, migrate_jsonl
    migrate_jsonl(PROCESSED / "seller_insights.jsonl")
    insights = load_sessions()
    section("PHASE 5 -- seller dashboard")
    print(f"""{len(insights)} concierge sessions logged. Explore everything above
interactively -- KPIs, the flagged shortlist, per-product evidence, and
session transcripts:

    uv run streamlit run app.py""")
    print("\nDemo complete. Total cloud spend across all of the above: $0.")


if __name__ == "__main__":
    main()
