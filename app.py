"""True-Texture — Seller Returns Intelligence dashboard (Phase 5).

    uv run streamlit run app.py

Everything rendered here is computed from local pipeline outputs — no AWS:
    data/processed/diagnosis.jsonl         Phase 2 mismatch diagnoses
    data/processed/texture_sentences.jsonl Phase 1 evidence pool
    data/processed/insights.sqlite         Phase 4 concierge sessions (episodic memory)
    data/processed/visual_audit.jsonl      Phase 3 visual corroboration (optional)
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "NONE": 3}


def load_jsonl(name: str) -> list[dict]:
    path = PROCESSED / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


st.set_page_config(page_title="True-Texture Seller Intelligence", layout="wide")
st.title("True-Texture — Returns Intelligence")
st.caption("Fabric/texture mismatch detection for fashion marketplaces. "
           "Every flag below carries its evidence — no black-box scores.")

from src.concierge.insights_store import load_sessions, migrate_jsonl

diagnoses = load_jsonl("diagnosis.jsonl")
sentences = load_jsonl("texture_sentences.jsonl")
# Episodic memory lives in SQLite now; import any legacy JSONL once.
migrate_jsonl(PROCESSED / "seller_insights.jsonl")
insights = load_sessions()
visual = {v["parent_asin"]: v for v in load_jsonl("visual_audit.jsonl")}

if not diagnoses:
    st.error("No diagnosis data. Run Phases 1-2 first (see README quickstart).")
    st.stop()

flagged = sorted((d for d in diagnoses if d["priority"] != "NONE"),
                 key=lambda d: (PRIORITY_ORDER[d["priority"]],
                                -d.get("n_complaint_sentences", 0)))

# ---------------------------------------------------------------- KPI row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Garments analyzed", len(diagnoses))
k2.metric("Flagged for action", len(flagged))
k3.metric("Texture evidence sentences", len(sentences))
k4.metric("Negation false-hits suppressed",
          sum(d.get("negated_suppressed", 0) for d in diagnoses))
k5.metric("Concierge sessions", len(insights))

tab_shortlist, tab_concierge = st.tabs(
    ["📋 Mismatch shortlist", "💬 Concierge insights"])

# ---------------------------------------------------------- shortlist tab
with tab_shortlist:
    left, right = st.columns([3, 2])

    with left:
        priorities = st.multiselect(
            "Priority filter", ["CRITICAL", "HIGH", "MEDIUM"],
            default=["CRITICAL", "HIGH", "MEDIUM"])
        rows = [d for d in flagged if d["priority"] in priorities]
        st.dataframe(pd.DataFrame([{
            "Priority": d["priority"],
            "Product": d["title"][:70],
            "Claims": ", ".join(d["claimed_materials"]),
            "Complaints": ", ".join(d["complaint_adjectives"]),
            "Evidence": d.get("n_complaint_sentences", 0),
            "Suspected substitute": (d.get("substitution_hypothesis") or {}).get(
                "suspected_fiber", "—"),
        } for d in rows]), width="stretch", hide_index=True)

        if rows:
            adj_counts = Counter(a for d in rows for a in d["complaint_adjectives"])
            st.subheader("Complaint signals across flagged products")
            st.bar_chart(pd.DataFrame.from_dict(
                adj_counts, orient="index", columns=["mentions"]))

    with right:
        st.subheader("Product drill-down")
        options = {f"[{d['priority']}] {d['title'][:60]}": d for d in flagged}
        if not options:
            st.info("Nothing flagged at the selected priorities.")
        else:
            pick = st.selectbox("Product", list(options))
            d = options[pick]
            st.markdown(f"**Claimed materials:** {', '.join(d['claimed_materials'])}")
            hyp = d.get("substitution_hypothesis")
            if hyp:
                sig = ", ".join(hyp.get("matching_signals", [])) or "declared default"
                st.warning(f"**Substitution hypothesis:** likely "
                           f"**{hyp['suspected_fiber']}** ({hyp['confidence']}: {sig})")
            st.markdown("**Customer evidence** (negation- and rating-gated):")
            seen = set()
            for h in d.get("hits", []):
                if h.get("complaint") and h["sentence"] not in seen:
                    seen.add(h["sentence"])
                    st.markdown(f"> “{h['sentence']}” — {h.get('rating')}★ "
                                f"(matched: *{h['adjective']}*)")
            if d.get("negated_suppressed"):
                st.caption(f"{d['negated_suppressed']} negated mention(s) suppressed "
                           f"(e.g. “not scratchy”) — would have been false flags.")
            va = visual.get(d["parent_asin"])
            if va:
                st.markdown("**Visual corroboration (Phase 3):**")
                st.markdown(
                    f"- CLIP similarity official↔review images: "
                    f"mean {va['clip_similarity_mean']:.2f}, min {va['clip_similarity_min']:.2f} "
                    f"({va['n_review']} review photos)\n"
                    f"- Color delta: {va['color_delta_mean']:.2f}\n"
                    f"- Verdict: **{va['visual_corroboration']}** "
                    f"(images corroborate text evidence only — never flag alone)")

# ---------------------------------------------------------- concierge tab
with tab_concierge:
    if not insights:
        st.info("No concierge sessions yet. Run "
                "`uv run python scripts/run_concierge.py` (add `--mock` while "
                "AWS access is being resolved) or seed demo rows with "
                "`uv run python scripts/seed_mock_insights.py`.")
    else:
        from src.concierge.seller_escalation import escalations, DEFAULT_THRESHOLD
        esc = escalations(insights)
        if esc:
            st.subheader("🚨 Distributor-consultation escalations")
            st.caption(f"Products returned ≥{DEFAULT_THRESHOLD}× despite no fabric "
                       f"or weather fault — expectation/presentation gaps, not defects.")
            for e in esc.values():
                st.error(f"**{e['title'][:70]}** — {e['no_issue_returns']} "
                         f"'no-fault' returns.  \n{e['recommendation']}")

        cases = Counter(i["diagnosis"].get("case_class", "—") for i in insights)
        st.caption("Case mix: " + " · ".join(f"{k}={v}" for k, v in cases.items()))

        causes = Counter(i["diagnosis"].get("root_cause_category", "OTHER")
                         for i in insights)
        actions = Counter(i["diagnosis"].get("seller_action", "NO_ACTION")
                          for i in insights)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Root causes")
            st.bar_chart(pd.DataFrame.from_dict(
                causes, orient="index", columns=["sessions"]))
        with c2:
            st.subheader("Recommended seller actions")
            st.bar_chart(pd.DataFrame.from_dict(
                actions, orient="index", columns=["sessions"]))

        st.subheader("Session log")
        for i in reversed(insights):
            dx = i["diagnosis"]
            is_mock = "mock" in (i.get("model_id") or "")
            badge = " `MOCK`" if is_mock else ""
            with st.expander(
                    f"{i['timestamp'][:16]} — {i['title'][:60]} — "
                    f"{dx.get('root_cause_category')} → {dx.get('seller_action')}"
                    f"{badge}"):
                gt = dx.get("material_ground_truth") or []
                if gt:
                    claims = "; ".join(
                        f"**{g['material']}** (genuine feel: {', '.join(g['genuine_feel'])}; "
                        f"ideal weather: {', '.join(g['ideal_weather'])})" for g in gt)
                    reported = dx.get("reported_feel") or "—"
                    weather = dx.get("weather_context")
                    st.markdown(f"⚖️ **Claim vs experience:** listing claims {claims} "
                                f"— customer reports **{reported}**"
                                + (f", worn in: *{weather}*" if weather else "") + ".")
                wgt = dx.get("weave_ground_truth") or []
                if wgt:
                    weaves_txt = "; ".join(
                        f"**{w['weave']}** (should feel: {', '.join(w['should_feel'][:4])})"
                        for w in wgt)
                    st.markdown(f"🧵 **Weave:** {weaves_txt}")
                if dx.get("weather_suitability_mismatch"):
                    st.info("🌦️ **Weather-suitability mismatch** — item worn outside "
                            "its ideal conditions; customer was gently informed.")
                rec = dx.get("listing_fix_recommendation")
                if rec:
                    sub = dx.get("suspected_substitution")
                    tag = (f"suspected substitution → **{sub}**" if sub
                           else "genuine-fiber quality issue (no substitution)")
                    st.success(f"🛠️ **Action — {dx.get('seller_action')}** "
                               f"({tag}):  \n{rec}")
                msg = dx.get("customer_closing_message")
                if msg:
                    st.markdown(f"💬 **Message shown to customer:** _{msg}_")
                st.json(dx)
                st.markdown("**Transcript:**")
                for turn in i.get("transcript", []):
                    if turn["role"] == "concierge":
                        st.markdown(f"🤖 {turn.get('question')}  \n"
                                    f"*options: {', '.join(turn.get('options', []))}*")
                    elif turn["role"] == "customer":
                        st.markdown(f"🧑 {turn.get('text')}")
                st.caption(f"model: {i.get('model_id')} · "
                           f"cost: ${i.get('cost_usd', 0):.4f}")
