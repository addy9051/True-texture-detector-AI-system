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

**Status:** Phases 0–6 built and verified · **cloud spend: $0 of a $100 budget** ·
live LLM path: Groq free tier (no card, no billing entity) while an AISPL
account restriction blocks Bedrock (support case open — AWS infra kept as
fallback, switch-back is one env var); a mock transport keeps every stage
demoable fully offline.

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
   $0.25 hard stop. Grounded, two-audience output: the engine injects material
   ground truth into every diagnosis, and a **2×2 response matrix** on feel ×
   weather drives distinct customer messages and seller actions —
   defect/right-weather → apology + supplier action; right-feel/wrong-weather →
   weather education + intuitive adjustments (no seller fault); both → apology +
   material/weather guidance + defect mitigation; neither → apology + ideal-use
   guidance, with the seller escalated to a distributor consultation only once
   such "no-fault" returns cross a threshold. All four quadrants verified live
   on Groq (Llama/gpt-oss free tier).
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

# 6. LLM Ops loop — trace, evaluate, diagnose, gate, release (offline $0, or --live)
uv run python scripts/run_llmops.py            # mock pipeline demo
uv run python scripts/run_llmops.py --live --judge
```

### LLM Ops tracing (Langfuse Cloud, optional)

Traces always write to `data/processed/traces.jsonl` and the dashboard's 🔬 LLM Ops
tab for $0. To also stream them to the **Langfuse** UI (trace waterfall, token/cost
dashboards, eval scores per trace):

1. Sign up at <https://cloud.langfuse.com> (free tier, no card) → create a project → copy the keys.
2. `setx LANGFUSE_PUBLIC_KEY "pk-lf-..."` and `setx LANGFUSE_SECRET_KEY "sk-lf-..."` → new terminal.
   If your project is in the **US** region, also `setx LANGFUSE_HOST "https://us.cloud.langfuse.com"`
   (EU is the default host).
3. Re-run `uv run python scripts/run_llmops.py --live --judge` — each run prints its Langfuse URL.

Uses the Langfuse v4 (OpenTelemetry) SDK; emission is guarded, so an unreachable
Langfuse never breaks a run. LangSmith is intentionally not used — it's coupled to
LangChain, which this project doesn't use; Langfuse is framework-agnostic.

Narrated end-to-end tour of all results:

```powershell
uv run python scripts/demo.py
```

## LLM provider setup (Phase 4 live path)

The concierge is provider-agnostic (`LLM_PROVIDER` = `groq` | `gemini` |
`bedrock` | `mock`; default order: groq if `GROQ_API_KEY` set, then gemini if
`GEMINI_API_KEY` set, else bedrock).

**Option A — Groq (current primary).** Free tier needs no credit card and no
billing account — immune to both the AWS account-gating below and the Google
Cloud billing issues encountered.

1. Sign in at <https://console.groq.com> (Google/GitHub) → API Keys → create.
2. `setx GROQ_API_KEY "<key>"` → open a new terminal.
3. Verify: `uv run python scripts/check_llm.py` (pings the model + probes tool
   support). Default model `openai/gpt-oss-120b` (best tool-calling reasoner on
   the account's list; the client requests `reasoning_effort: low` so token
   budgets go to answers, not hidden thinking). Free tier ≈ 1,000 req/day →
   hundreds of sessions/day at $0. Alternates via `GROQ_MODEL_ID`:
   `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`.
4. Live end-to-end check (scripted, non-interactive):
   `uv run python scripts/test_live_concierge.py`

**Option B — Google AI Studio Gemini.** Same no-card story
(<https://aistudio.google.com/apikey>, `setx GEMINI_API_KEY`), parked due to a
Google Cloud billing issue on this account.

**Option C — AWS Bedrock (fallback infrastructure, kept intact).**

1. Models auto-enable on first invoke (the Model-access console page is
   retired). Current model: `mistral.mistral-large-2407-v1:0` in `us-west-2`;
   target is Nova Pro (`us.amazon.nova-pro-v1:0`) once the account issue clears.
2. IAM user with `bedrock:InvokeModel` (+ streaming), credentials via
   `aws configure` (or a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`).
3. Budget alarm in Billing → Budgets (a $50 alarm is set on this account).
4. Verify: `uv run python scripts/check_bedrock.py`.

**Switch-back plan:** when `check_bedrock.py` passes without
`ValidationException: Operation not allowed`, run
`setx LLM_PROVIDER "bedrock"` — no code changes.

> **Known AWS account gotchas** (both hit during this project): AWS Free Tier
> *free plan* and AISPL (India-billed) accounts can return
> `ValidationException: Operation not allowed` on Bedrock invokes — an
> account-level gate. Fix is plan upgrade / support case, never code.

## Cost discipline

Everything in Phases 0–3 and 5 runs locally for $0. Bedrock enters in Phase 4
only, via the plain Converse API — **no** Bedrock Agents, **no** Knowledge Bases
(the KB default vector store alone would burn the entire budget). Every session
prints its exact token cost and hard-stops at $0.25.

## Layout

```
data/fabric_physics.json        fabric ontology: fibers (feel, thermal, substitution) + weaves (surface feel)
data/category_materials.json    top-~10 materials per apparel category (India taxonomy)
src/physics/category_materials.py  category material prior + fiber normalization
src/ingest/                     dataset sampling (McAuley Amazon Reviews 2023)
src/nlp/semantic_filter.py      MiniLM aspect filter (calibrated threshold 0.50)
src/physics/fabric_ontology.py  claimed material -> expectations -> mismatch hits
src/diagnosis/                  negation gate + priority scoring + substitution logic
src/visual/clip_audit.py        CLIP/color audit + the control-group experiment
src/concierge/                  LLM clients, interview engine, offline mock
src/concierge/skill.md          procedural memory: the interview policy / RESPONSE MATRIX
src/concierge/insights_store.py episodic memory: sessions in SQLite (SQL recency)
src/llmops/                     LLM Ops loop: trace · observe · eval (LLM-judge) · diagnose · gate · release
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
