"""One LLM Ops loop over the returns concierge.

For each scenario: run the agent → TRACE it → OBSERVE health → EVAL quality →
DIAGNOSE failures → then GATE the suite (ship vs fix) and, on a pass, RELEASE a
blessed prompt+config version.

    uv run python scripts/run_llmops.py                # offline, mock, $0 (pipeline demo)
    uv run python scripts/run_llmops.py --live         # real provider (Groq/Gemini/Bedrock)
    uv run python scripts/run_llmops.py --live --judge # + LLM-as-judge scoring

Writes data/processed/traces.jsonl (observability log, appended) and
data/processed/llmops_report.json (this run's report, read by the dashboard).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.concierge.concierge import ConciergeSession
from src.concierge.provider import make_chat, provider_name
from src.physics.fabric_ontology import FabricOntology
from src.llmops.tracer import (RunTrace, TracingChat, save_trace, emit_langfuse,
                               langfuse_enabled)
from src.llmops.observe import health
from src.llmops.evaluate import evaluate
from src.llmops.diagnose import diagnose
from src.llmops.gate import gate

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRIORITY = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "NONE": 3}

# Each scenario: scripted customer answers + the case the concierge SHOULD reach.
SCENARIOS = [
    ("substitution", "FEEL_ONLY", [
        "The fabric feels wrong, not what I expected",
        "it looks shiny and feels like cheap plastic, not cotton at all",
        "just normal indoor wear on a mild day"]),
    ("quality", "FEEL_ONLY", [
        "The fabric feels wrong for what it says",
        "it feels rough and coarse, cheap scratchy cotton, not soft",
        "just normal indoor wear"]),
    ("weather", "WEATHER_ONLY", [
        "It didn't keep me comfortable",
        "the fabric felt fine, soft like cotton, no complaints there",
        "I wore it outdoors on a freezing winter evening with nothing over it"]),
    ("both", "FEEL_AND_WEATHER", [
        "The dress was disappointing overall",
        "the fabric felt shiny and plasticky, not like cotton at all",
        "and I wore it outdoors on a freezing winter night with nothing over it"]),
    ("neither", "NO_ISSUE", [
        "It just wasn't right for me",
        "honestly the fabric felt fine, soft and cotton-like as described",
        "mild day indoors, weather was fine — I just didn't love the style"]),
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def pick_product(products, diags):
    flagged = sorted((d for d in diags.values() if d["priority"] != "NONE"),
                     key=lambda d: PRIORITY[d["priority"]])
    if not flagged:
        sys.exit("No flagged products — run Phases 1-2 first.")
    return flagged[0]["parent_asin"]


def run_one(scenario, expected, answers, product, diag, ontology, is_mock, judge_chat):
    trace = RunTrace(scenario=scenario, parent_asin=product["parent_asin"],
                     title=(product.get("title") or "")[:80], provider=provider_name(),
                     expected_case=None if is_mock else expected)
    chat = TracingChat(make_chat(), trace)
    session = ConciergeSession(chat, product, ontology, diag)
    converged = False
    run_error = None
    try:
        event = session.start()
        for ans in answers:
            if event["type"] != "question":
                break
            event = session.answer(ans)
        converged = event["type"] == "diagnosis"
        dx = event["data"] if converged else {}
    except Exception as e:
        dx = {}
        run_error = f"{type(e).__name__}: {str(e)[:140]}"
        print(f"    [{scenario}] run error: {run_error[:100]}")

    td = trace.finish(diagnosis=dx, transcript=session.transcript,
                      transport=session.mode, questions=session.questions_asked,
                      cost_usd=chat.meter.usd, converged=converged, error=run_error)
    save_trace(td)

    h = health(td)
    ev = evaluate(td, judge_chat=judge_chat)
    dg = diagnose(td, h, ev)
    url = emit_langfuse(td, ev)   # mirror to Langfuse Cloud (no-op if unconfigured)
    return {"scenario": scenario, "run_id": td["run_id"], "model_id": td["model_id"],
            "trace": td, "health": h, "eval": ev, "diagnosis": dg, "langfuse_url": url}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="use the real provider (default: mock, $0)")
    ap.add_argument("--judge", action="store_true", help="enable LLM-as-judge (implies --live)")
    args = ap.parse_args()
    if not args.live and not args.judge:
        os.environ["LLM_PROVIDER"] = "mock"
    is_mock = provider_name() == "mock"

    products = {p["parent_asin"]: p for p in read_jsonl(RAW / "products.jsonl")}
    diags = {d["parent_asin"]: d for d in read_jsonl(PROCESSED / "diagnosis.jsonl")}
    asin = pick_product(products, diags)
    product, diag = products[asin], diags.get(asin)
    ontology = FabricOntology()
    judge_chat = make_chat() if (args.judge and not is_mock) else None

    lf_status = "Langfuse Cloud ✓" if langfuse_enabled() else "local JSONL only (set LANGFUSE_* for Langfuse)"
    print(f"LLM Ops loop · provider={provider_name()} · tracing: {lf_status}")
    print(f"product={product['title'][:50]} · "
          f"{'(mock: correctness/judge skipped — pipeline demo)' if is_mock else '(live)'}\n")

    reports = []
    for scenario, expected, answers in SCENARIOS:
        r = run_one(scenario, expected, answers, product, diag, ontology, is_mock, judge_chat)
        ev, h = r["eval"], r["health"]
        flag = "PASS" if (ev["passed"] and h["healthy"]) else "FAIL"
        judge = r["eval"].get("judge")
        jtxt = f" · judge {judge['avg']}/5" if judge and not judge.get("error") else ""
        print(f"  [{flag}] {scenario:13} case={r['trace']['diagnosis'].get('case_class','-'):16} "
              f"{h['metrics']['total_latency_ms']/1000:.1f}s · {h['metrics']['tokens_in']+h['metrics']['tokens_out']} tok{jtxt}")
        for f in r["diagnosis"]:
            print(f"        ↳ {f['issue']}  →  fix: {f['knob']}")
        if r.get("langfuse_url"):
            print(f"        ↳ langfuse: {r['langfuse_url']}")
        reports.append(r)

    model_id = reports[0]["model_id"] if reports else ""
    g = gate(reports, provider=provider_name(), model_id=model_id, do_release=True)

    print(f"\n=== GATE: {g['decision']}  (pass rate {g['pass_rate']:.0%} of {g['n_runs']}, "
          f"need {g['threshold']:.0%}) ===")
    if g["decision"] == "SHIP":
        rel = g["released"]
        print(f"RELEASE → version {rel['version']} · prompt {rel['prompt_version']} blessed "
              f"(data/processed/releases.json)")
    else:
        print("FIX — knobs to turn:")
        for fx in g["fixes_by_knob"]:
            print(f"  • {fx['knob']}: {fx['fixes'][0]}")

    report = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "provider": provider_name(), "model_id": model_id, "mock": is_mock,
              "gate": g,
              "runs": [{k: r[k] for k in ("scenario", "run_id", "health", "eval", "diagnosis")}
                       | {"case_class": r["trace"]["diagnosis"].get("case_class"),
                          "latency_ms": r["trace"]["total_latency_ms"],
                          "tokens": r["trace"]["tokens_in"] + r["trace"]["tokens_out"],
                          "transport": r["trace"]["transport"]}
                       for r in reports]}
    (PROCESSED / "llmops_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport → {PROCESSED / 'llmops_report.json'} · traces → {PROCESSED / 'traces.jsonl'}")


if __name__ == "__main__":
    main()
