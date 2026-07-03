# True-Texture — Returns Intelligence for Fashion Ecommerce

Fashion listings lie about fabric; customers find out on their skin; returns follow.
True-Texture mines the tactile truth out of review text, cross-checks it against the
material each listing *claims*, interrogates the returner at return time with an
adaptive LLM concierge, and hands sellers an evidence-backed fix — down to *"your
supplier is likely substituting polyester for the listed cotton."*

**Primary market: Indian fashion ecommerce** (Myntra, Flipkart, Meesho, Ajio,
Amazon.in), where fashion is the largest category by order volume and "product is
different from listing" is a top return reason. Mechanics are prototyped on the
public US Amazon reviews dataset — the best public data with review photos and
material fields — with an India-validation phase planned (multilingual model,
Kaggle Flipkart/Myntra review sets).

**Status:** Phases 0–5 built and verified · **cloud spend: $0 of a $100 budget** ·
live Bedrock path blocked by an AISPL account restriction (support case open);
a mock transport keeps every stage demoable offline.

```powershell
uv run python scripts/demo.py     # guided 60-second tour of every result below
```

## Why this problem (condensed market case)

| | India (primary) | US (contrast) |
|---|---|---|
| Fashion ecommerce GMV | ~$21.6B (2025), largest ecommerce category | Amazon apparel ~$72B |
| Online fashion return rate | 25–35% → **$5.5–7.5B/yr returned** | ~24–25% |
| "Not as described / fabric" share | **#1 stated return reason** on value platforms | ~10–13% (fit dominates) |
| Unit economics | ₹200–400 per return vs ₹300–600 AOV — one return erases 3–5 orders' margin | ~25–35% of item value |
| Incumbent tooling | none at maturity | Amazon Fit Insights (2024) |

Full sourced analysis and the audit of the original project plan: [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Pipeline

```
Amazon Reviews 2023 (public, 2.5M reviews scanned)
  └─▶ 1. EVIDENCE ENGINE   MiniLM aspect filter → fabric-feel sentences   [local, $0]
        └─▶ 2. DIAGNOSIS   claimed fiber vs reported feel (fabric ontology,
                           negation + rating gates) → ranked audit shortlist
              ├─▶ 3. VISUAL AUDIT   CLIP + color vs control group          [negative result]
              ├─▶ 4. CONCIERGE      1-2 adaptive questions at return time  [Bedrock / mock]
              └─▶ 5. DASHBOARD      evidence-backed seller actions         [Streamlit]
```

## Verified results (all reproducible from the quickstart)

1. **Evidence engine, calibrated** — 300 products / 17,754 reviews sampled from
   2.5M; similarity threshold set by inspecting 48,040 scored sentences: precision
   jumps from ~50% (generic "very comfortable") to **~90%** at the 0.50 band.
2. **Diagnosis engine, gated** — 229 garments diagnosed; negation gate suppressed
   7 false flags ("it was **not** scratchy"); complaints only count from ≤3★
   reviews; CRITICAL requires ≥2 independent complaint sentences. Showcase flag:
   a "cotton" dress whose customers report *"super cheap and kind of shiny"* →
   corroborated polyester-substitution hypothesis → supply-chain audit action.
3. **Visual audit — an honest negative result, twice.** Tested the obvious idea
   ("compare listing photos with customer photos via CLIP") against a control
   group of complaint-free products, in two conditions:

   | condition | flagged (n=10) | control (n=10) |
   |---|---|---|
   | whole frame | clip 0.648 · color Δ 0.689 | clip 0.659 · color Δ 0.598 |
   | garment crops (rembg) | clip 0.727 · color Δ 0.711 | clip 0.738 · color Δ 0.686 |

   The distributions overlap in both conditions: photo-level appearance comparison
   **cannot see texture** — the popular "flag if CLIP < 0.75" heuristic fires on
   20/20 products. Verdicts are conservative (INCONCLUSIVE unless outside the
   entire control band), and text remains the primary signal by design.
4. **Returns concierge** — dual-transport interview engine (native forced tool
   calls, automatic JSON-protocol fallback for accounts that reject Converse
   toolConfig), fabric ontology + prior evidence in the system prompt (marked
   internal — never used to lead the customer), per-session cost meter with a
   $0.25 hard stop. Verified end-to-end via the mock transport.
5. **Seller dashboard** — KPIs, priority-filtered shortlist, per-product evidence
   drill-down, visual-audit verdicts, concierge transcripts (mock rows badged).
   Smoke-tested headless (AppTest: zero exceptions).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (manages Python + dependencies).

```powershell
uv sync

# 1. Sample the public dataset (no scraping; ~2GB cached locally)
uv run python -m src.ingest.download_dataset --max-products 300

# 2. Texture sentences + calibration bands
uv run python scripts/run_phase1.py
uv run python scripts/calibrate_threshold.py     # optional

# 3. Diagnosis engine -> ranked supplier-audit shortlist
uv run python scripts/run_phase2.py
uv run python scripts/check_negation.py          # fast self-test

# 4. Visual audit incl. control-group experiment (downloads CLIP ~350MB once)
uv run python scripts/run_phase3.py --control 12

# 5. Concierge (use --mock while AWS access is blocked) + dashboard
uv run python scripts/run_concierge.py --mock
uv run python scripts/seed_mock_insights.py
uv run streamlit run app.py

# Or the narrated end-to-end tour:
uv run python scripts/demo.py
```

## AWS setup (Phase 4 live path)

One-time, ~10 minutes — see the per-step details in earlier commits if needed.

1. **Models auto-enable on first invoke** (the Model-access console page is
   retired). Default model: `mistral.mistral-large-2407-v1:0` in `us-west-2`
   (temporary — Anthropic Claude is Marketplace-billed and NOT covered by
   promotional credits; Amazon Nova Pro is the target once the account issue
   clears: `$env:BEDROCK_MODEL_ID = "us.amazon.nova-pro-v1:0"`).
2. **IAM user** with `bedrock:InvokeModel` + `bedrock:InvokeModelWithResponseStream`,
   credentials via `aws configure` (or a Bedrock API key in
   `AWS_BEARER_TOKEN_BEDROCK` — supported, Bedrock-only).
3. **Budget alarm** in Billing → Budgets (a $50 alarm is set on this account).
4. Verify: `uv run python scripts/check_bedrock.py` — it checks credentials,
   pings the model, and probes native tool support.

> **Known account gotchas** (both hit during this project): AWS Free Tier *free
> plan* and AISPL (India-billed) accounts can return
> `ValidationException: Operation not allowed` on Bedrock invokes — an
> account-level gate. Fix is plan upgrade / support case, never code.

## Cost discipline

Everything in Phases 0–3 and 5 runs locally for $0. Bedrock enters in Phase 4
only, via the plain Converse API — **no** Bedrock Agents, **no** Knowledge Bases
(the KB default vector store alone would burn the entire budget). Every session
prints its exact token cost and hard-stops at $0.25.

## Layout

```
data/fabric_physics.json        fabric ontology (feel, red flags, substitution tells)
src/ingest/                     dataset sampling (McAuley Amazon Reviews 2023)
src/nlp/semantic_filter.py      MiniLM aspect filter (calibrated threshold 0.50)
src/physics/fabric_ontology.py  claimed material -> expectations -> mismatch hits
src/diagnosis/                  negation gate + priority scoring + substitution logic
src/visual/clip_audit.py        CLIP/color audit + the control-group experiment
src/concierge/                  Bedrock client, interview engine, offline mock
scripts/                        one runnable script per phase + demo.py
app.py                          Streamlit seller dashboard
PROJECT_PLAN.md                 market case, plan audit, phase log, budget
```

## Honest limitations

- Fit/size returns (the majority bucket) are out of scope by design.
- The fabric ontology is heuristic, not lab physics; listings often omit materials
  (39% of sampled products had detectable claims).
- The visual channel is currently non-discriminating (see the control experiment);
  multimodal-LLM comparison is the remaining avenue once Bedrock access clears.
- Indian-market validation (Hinglish reviews, Indian fabric vocabulary) is designed
  (Phase 1b) but not yet run — no public Indian dataset matches McAuley's quality.
