# True-Texture Returns Intelligence Platform — Revised Project Plan

> Status: Phase 0 (scaffold) ✅ · Phase 1 (data + NLP core) ✅ ran 2026-07-03 —
> 300 products / 17,754 reviews sampled, filter threshold calibrated to 0.50 on
> 48k real sentences (~90% precision above threshold vs ~50% at the old 0.45)
> Budget: $100 AWS credits · Projected total AWS spend: **under $20** · Spent so far: $0

## 0b. Agent memory architecture (staged)

Three tiers, deliberately kept lightweight — infrastructure is added only where
the simple approach breaks (see the reasoning in the git history / chat log).

| Tier | What | Implementation | Why not more |
|---|---|---|---|
| **Semantic** (durable facts) | fiber + weave ontology, category priors | `fabric_physics.json`, `category_materials.json` — injected into the prompt | Only ~a few KB; RAG over 35 facts is slower, costlier, and can retrieve-miss. Stays git-diffable. |
| **Procedural** (how to act) | interview policy + RESPONSE MATRIX | `src/concierge/skill.md`, loaded into the system prompt | Loaded every turn, not retrieved — a versioned file separates policy from code |
| **Episodic** (dated events) | concierge sessions + transcripts | `src/concierge/insights_store.py` → SQLite (SQL recency: by ASIN/date/case_class) | Vector-store relevance is a later stage, once session volume justifies it |

Later stages (not built): pgvector (Supabase/Neon) for episodic *relevance* search
at volume; a summarizer agent distilling patterns into a **separate** learned-facts
namespace with provenance + a promotion threshold (never auto-mutating the curated
ontology). Kept local (SQLite) for now to avoid another billing dependency.

## 1. The business case — India-first (validated 2026-07)

Primary target market: Indian fashion ecommerce (Myntra, Flipkart, Meesho, Ajio,
Amazon.in). US data kept as contrast — the India case is proportionally stronger.

| Metric | India | US (contrast) |
|---|---|---|
| Fashion ecommerce GMV | ~$21.6B (2025), ~24% CAGR → ~$98B by 2032 | Amazon US apparel ~$72B |
| Fashion share of the market | **Largest category**: ~19–32% of the ~$70B ecommerce GMV (definition-dependent); #1 by order volume. Meesho: fashion = 50% of GMV; Myntra/Ajio are fashion-only platforms | ~9–12% of Amazon GMV; >30% of listings |
| Online fashion return rate | **25–35%** (ethnic wear at the high end) | ~24–25% |
| Returned fashion GMV / yr | ≈ **$5.5–7.5B** | ~$17–18B (Amazon only) |
| "Not as described / quality" share | **Top-tier return reason** — "product is different (color, fabric, design)" is the #1 stated reason on Meesho; seller-tooling analyses attribute ~80% of return causes to listing-quality gaps | ~10–13% of returns (fit dominates at 65–72%) |
| Addressable slice / yr | ≈ **$1.1–2.2B** returned GMV (conservative 20–30% share) | ~$1.8–2.3B |
| Cost per return | ₹120–250 reverse shipping + ₹80–150 refurbishing + ₹35–50 RTO fees — vs value-fashion AOV of ₹300–600 | ~25–35% of item value |
| Incumbent tooling | **None at maturity** — size recommenders and generic quality scores only | Amazon Fit Insights (live 2024) |

### Why the problem is proportionally bigger in India

1. **Fashion is the core of Indian ecommerce, not a side category.** It is the largest
   category by order volume; Meesho alone carries 37% of all Indian ecommerce orders
   with half its GMV in fashion. Solving a fashion-returns problem in India means
   solving a problem for the marketplace's flagship category.
2. **"Not as described" is the dominant failure mode, not the minority slice.** The
   long tail of unbranded/white-label sellers, reseller-copied studio images, and
   rampant fiber substitution (viscose sold as "cotton", polyester "art silk" sold as
   silk, faux georgette everywhere) makes claimed-vs-actual material mismatch a
   first-order return driver — exactly what this system detects.
3. **Unit economics are brutal.** A single return costs ₹200–400 all-in against a
   ₹300–600 AOV: one texture-mismatch return erases the margin of 3–5 good orders.
   COD RTO (20–40% of COD orders) already strains seller cash flow; post-delivery
   quality returns compound it.
4. **Real white space.** Myntra has size recommendations and Meesho has generic quality
   ratings, but no Indian marketplace runs active return-time diagnosis with a
   material-truth cross-check, and none feeds structured listing-fix actions back to
   its lakhs of small sellers.

Key nuance (holds in both markets): return-reason codes are unreliable — customers pick
whatever option is free/fast — so producing trustworthy root-cause data is itself the
product.

**Verdict: worth building, and India is the better market for it.** In the US this is an
extension of Amazon's Fit Insights; in India the equivalent tooling simply does not
exist, the addressable slice is 2–3× larger as a share of the market, and the seller
base (small, unbranded, no QC infrastructure) benefits most from cheap corrective
actions. The novel core remains:
1. **Active diagnosis at return time** (structured interrogation, not passive review mining)
2. **Fabric-ontology cross-check** (claimed material vs reported sensation)
3. **Closed loop to the seller** with a specific, cheap corrective action

## 2. Audit of the original plan (conv.md) — what changed and why

| Original plan | Problem | Revision |
|---|---|---|
| Scrape Amazon with BeautifulSoup/Selenium | ToS violation, brittle, blocks fast | **McAuley Lab Amazon Reviews 2023** public dataset (HuggingFace) — has review text, user photos, official images, and material fields |
| CLIP cosine similarity (studio vs user photo), flag < 0.75 | Lighting/pose/background dominate the embedding distance; texture is a fine-grained signal CLIP is weak at; threshold arbitrary → false-positive flood | Images are **corroborating** evidence only (color/sheen), text reviews are **primary** evidence. Segment the garment first; compare attributes, or have a multimodal LLM describe differences in words |
| Bedrock Agent + Knowledge Base (S3 RAG) for fabric_physics.json | Bedrock KB needs a vector store — OpenSearch Serverless bills per OCU-hour and would **burn the entire $100 in days**. The JSON is a few KB | **No RAG.** Inject the ontology directly into the system prompt / use tool use via the plain Converse API |
| Claude 3 Haiku | Outdated — and ALL Anthropic models on Bedrock are Marketplace-billed, so promotional credits don't cover them | Amazon Nova Pro (`apac.amazon.nova-pro-v1:0`) — first-party/credit-eligible, tool use on Converse, multimodal (reusable in Phase 3) |
| DynamoDB + live WebSocket to dashboard | Overkill for a prototype; more IAM/config surface than value | SQLite locally; Streamlit reads it directly. Swap for DynamoDB only if demoing "AWS-native" matters |
| LDA topic modeling → upgraded to embedding aspect filtering | (conv.md already made this upgrade — it's correct) | Keep: MiniLM sentence embeddings vs texture anchor phrases, runs free + local |
| "82% of returns cite texture mismatch" style demo copy | Invented numbers undermine credibility | Compute real statistics from the dataset |
| Chat interrogation in the return flow | Returns UX is deliberately frictionless; open-ended chat would hurt CSAT at real scale | Design as 1–2 adaptive questions with option chips + one optional free-text field |

Also noted: conv.md's framing quotes ("River of Returns", executive interviews) trace to
Instagram reels / LinkedIn posts — the underlying problem is real (NRF: $890B returns in
2024) but treat the narrative as embellished, don't repeat it in a demo.

## 3. Revised architecture

```
                        ┌─────────────────────────────────────────────┐
                        │  DATA LAYER (local, free)                   │
                        │  McAuley Amazon Reviews 2023 (Fashion)      │
                        │  → data/raw/{reviews,products}.jsonl        │
                        └──────────────┬──────────────────────────────┘
                                       ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  EVIDENCE ENGINE (local, free)                                    │
   │  1. semantic_filter.py  — MiniLM: isolate texture/feel sentences  │
   │  2. fabric_ontology.py  — claimed material → expected properties  │
   │  3. visual_audit.py     — garment-cropped image corroboration     │
   │  → per-ASIN mismatch evidence file (SQLite/JSONL)                 │
   └──────────────┬────────────────────────────────────────────────────┘
                  ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  BEDROCK LAYER (the ~$10-20 of AWS spend)                         │
   │  4. Returns Concierge — Nova Pro, Converse API, ontology in       │
   │     system prompt, 1-2 adaptive questions + option chips          │
   │  5. Insight Summarizer — chat log + evidence → seller_insight     │
   │     JSON (root cause, supply-chain audit flag, listing fix)       │
   └──────────────┬────────────────────────────────────────────────────┘
                  ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  STREAMLIT APP (two tabs, one codebase)                           │
   │  Buyer tab:  return flow with concierge                          │
   │  Seller tab: returns-intelligence dashboard (real computed stats) │
   └───────────────────────────────────────────────────────────────────┘
```

## 4. Phase plan

- **Phase 0 — Scaffold** ✅ this repo structure, ontology, plan
- **Phase 1 — Data + NLP core** ✅ (ran 2026-07-03): 300 products / 17,754 reviews
  sampled (117 with detectable material claims); similarity threshold calibrated on
  48,040 sentences via `scripts/calibrate_threshold.py` → default set to 0.50.
  Rerun: `uv sync`, then `uv run python -m src.ingest.download_dataset`,
  then `uv run python scripts/run_phase1.py`
- **Phase 1b — India validation**: rerun the filter on an Indian review set (Kaggle has
  several Flipkart/Myntra product-review datasets, text-only) using the multilingual
  MiniLM model (`MULTILINGUAL_MODEL_NAME` in semantic_filter.py) to handle
  Hinglish/regional-language reviews; verify the ontology aliases catch Indian listing
  vocabulary (art silk, faux georgette, hosiery cotton)
- **Phase 2 — Diagnosis engine** ✅ (code written 2026-07-03): negation-gated +
  rating-gated `failing_adjectives` matching (fixes the Phase-1 "not scratchy"
  false-positives), `substitution_suspects` reasoning (cotton reported as
  "slick/shiny" → likely polyester), non-garment filter (drops sunglasses/watches),
  CRITICAL/HIGH/MEDIUM priority → ranked supplier-audit shortlist.
  Recalibrated on the first real run: CRITICAL now requires ≥2 *independent*
  complaint sentences (one review saying "rough & scratchy" is one piece of
  evidence, not two), and suspects carry a `substitution_signature` so
  cotton+"shiny" → polyester is *corroborated*, not a default guess.
  Run: `uv run python scripts/check_negation.py` (fast self-test) then
  `uv run python scripts/run_phase2.py` (reads cached Phase-1 sentences, no re-embed)
- **Phase 3 — Visual corroboration** ✅ run + calibrated with a control group
  2026-07-03, **honest negative result**: whole-frame CLIP + color histograms do
  NOT separate flagged from clean products (flagged clip 0.648/color 0.689 vs
  control 0.659/0.598, n=10 each) — the studio-vs-phone gap dominates, empirically
  killing the original plan's "flag if CLIP < 0.75" (it fires on 20/20 products).
  Verdicts are now conservative: INCONCLUSIVE unless outside the entire control
  band. **Iteration 2 (same day): repeated the experiment on rembg garment crops —
  still no separation (flagged clip 0.727/color 0.711 vs control 0.738/0.686).**
  Photo-level appearance comparison cannot see texture; text stays primary.
  Remaining visual avenue: multimodal-LLM image comparison via Bedrock when
  account access is restored. Run: `uv run python scripts/run_phase3.py --control 12`
- **Phase 4 — Returns Concierge** ✅ code written (2026-07-03), now
  **provider-agnostic and LIVE-VERIFIED 2026-07-04**: primary = gpt-oss-120b via
  Groq free tier (no card/billing entity — sidesteps the AISPL gate; Gemini
  adapter also available but parked over a Google billing issue). First real
  session passed review: neutral first question, correct polyester-substitution
  inference from the customer's words, native forced-tool transport, 2-question
  budget held, $0 billed. **Diagnosis upgraded (grounded + two-remedy): 3-question
  budget adds a weather/wear probe; the engine injects material ground truth
  (genuine feel, thermal, ideal weather) into every diagnosis from the ontology
  rather than trusting the model to recite it; recommendations must name the
  claimed material and match the remedy to the cause — SUPPLY_CHAIN_AUDIT +
  declare/verify fiber for a suspected substitution vs. the new QUALITY_IMPROVEMENT
  + source premium-grade material for a low-grade genuine fiber. Both paths
  verified live (shiny→polyester→audit; rough→null→quality). Plus a customer-
  facing 2×2 RESPONSE MATRIX (feel × weather), engine-classified into case_class:
  FEEL_ONLY (apology + supplier action), WEATHER_ONLY (weather education +
  intuitive adjustments, no seller fault), FEEL_AND_WEATHER (apology + material/
  weather guidance + defect mitigation), NO_ISSUE (apology + ideal-use guidance;
  seller escalated to distributor consultation only past a returns threshold via
  seller_escalation.py). All four quadrants verified live on Groq.** Fallback = AWS Bedrock
  (Mistral Large 24.07 now, Nova Pro when the account clears; Anthropic ruled
  out — Marketplace-billed, not covered by AWS credits). `LLM_PROVIDER` env
  switches; `scripts/check_llm.py` verifies the active provider.
  Original AWS design notes:
  tool-use-forced structure (`ask_question` max 2 rounds → `submit_diagnosis`),
  ontology + Phase-2 evidence in the system prompt (evidence steers probing but is
  never revealed to the customer), per-session cost meter with $0.25 hard stop.
  Run: `uv run python scripts/check_bedrock.py` then
  `uv run python scripts/run_concierge.py` (see README "AWS setup")
- **Phase 5 — Seller dashboard** ✅ built + smoke-tested 2026-07-03 (`app.py`):
  KPI row, priority-filtered mismatch shortlist, per-product drill-down with
  evidence quotes + substitution hypothesis + Phase-3 visual verdicts, concierge
  session log with transcripts (mock rows badged `MOCK`), complaint-signal and
  root-cause charts. Run: `uv run streamlit run app.py`.
  Seed demo sessions while AWS is blocked: `uv run python scripts/seed_mock_insights.py`
- **Category material prior** ✅ 2026-07-04: `data/category_materials.json` maps 93
  apparel categories (Myntra-style India taxonomy: Men's/Women's Topwear, Indian &
  Festive, Bottomwear, Innerwear, Activewear, Lingerie) to their top ~10 materials
  (930 entries). `src/physics/category_materials.py` normalizes catalog materials
  to ontology fibers. Wired into the concierge: when a listing states no fabric,
  it grounds on the category's likely fibers. Report: `uv run python scripts/category_coverage.py`
- **Weave ontology (second axis)** ✅ 2026-07-04: `fabric_physics.json` gained a
  `weaves` section — 23 constructions (satin, crepe, velvet, denim, fleece, net,
  jacquard, organza, lace…) with surface-feel expectations, failing adjectives, and
  a warmth hint, orthogonal to fiber. A satin reported as "rough" is now a flaggable
  feel mismatch regardless of fiber. This lifted catalog grounding 86.5% → **98%**
  (only "tissue/zari" and generic "blended" remain). The concierge detects both a
  product's fiber AND weave, injects both into the prompt and the grounded
  diagnosis (`weave_ground_truth`), and the dashboard shows a 🧵 Weave line.
  Weaves also carry `weather_suitability` (web-grounded 2026-07): the concierge
  combines fiber + weave weather and is told construction usually dominates
  warmth — a cotton fleece flags a hot-weather mismatch even though cotton is
  hot-ideal, because the fleece construction traps heat.
- **Phase 6 — Demo polish** ✅ 2026-07-03: `scripts/demo.py` — narrated end-to-end
  tour replaying only real computed results (market case → calibrated filter →
  gated diagnosis → the two-condition negative result → auto-played mock concierge
  → dashboard pointer); README rewritten as the portfolio-facing results document.
  Remaining nice-to-have: dashboard screenshot/GIF for the README.
- **Phase 7 — LLM Ops** ✅ 2026-07-05 (`src/llmops/`): the self-evolving feedback
  loop from the reference architecture. **Trace** (1 trace/run — `TracingChat`
  wraps any provider transparently, records per-call latency/tokens/tool-use/errors
  to `traces.jsonl`; **also streams to Langfuse Cloud** when `LANGFUSE_*` env is set
  — v4 OTel SDK, a root `agent` observation with a child `generation` per LLM call,
  tokens→cost dashboards, eval+judge scores attached to the trace; guarded so an
  unreachable Langfuse never breaks a run. LangSmith deliberately not used — it's
  LangChain-coupled; this project uses raw provider SDKs) → **Observe** (healthy? latency,
  tokens, cost, convergence) + **Eval** (good? deterministic checks — schema,
  question budget, case-class consistency, action⇒recommendation, expected-case —
  plus optional LLM-as-judge on neutrality/grounding/matrix-fit/actionability) →
  **Diagnose** (each failure → the knob to turn: skill.md / model / transport) →
  **Gate** (ship vs fix at an 80% pass bar) → **Release** (bless a prompt+config
  version = skill.md hash, in `releases.json` — the feedback arrow). Runs offline
  on the mock ($0 pipeline demo) or live: `uv run python scripts/run_llmops.py
  [--live --judge]`. Dashboard gains a 🔬 LLM Ops tab. Local-first, no new deps.

## 5. AWS budget for the $100 credits

| Item | Est. cost |
|---|---|
| Bedrock Nova Pro (~2–3k concierge/summarizer sessions, ~$0.005 each) | ~$5–10 |
| Nova Lite/Micro for bulk experiments (optional, 13–25× cheaper) | ~$1 |
| S3 (ontology + insight files, GBs at most) | < $1 |
| **Avoided**: OpenSearch Serverless (Bedrock KB default vector store) | would be ~$100+/mo — **do not enable** |
| Everything else (embeddings, CLIP, dataset, Streamlit, SQLite) | $0 — local |

Set a $25 budget alert in AWS Billing before first Bedrock call.

## 6. Risks / honest caveats

- Fit/size is out of scope by design — say so up front in any demo. (In India it's a
  smaller share of returns than in the US, but still significant.)
- **India data gap**: no public Indian equivalent of the McAuley dataset with review
  photos + material metadata. Strategy: build mechanics on the US Amazon dataset,
  validate the NLP filter and ontology on text-only Kaggle Flipkart/Myntra review sets.
- **Language**: Indian reviews mix English, Hinglish, and regional languages — the
  English-only MiniLM misses these; Phase 1b switches to the multilingual model.
- Bedrock is available in ap-south-1 (Mumbai) with Nova models — use it for the demo
  story ("runs in-region"); pricing is comparable, budget unchanged.
- Amazon Fit Insights already mines reviews for fabric sentiment passively (US); no
  Indian marketplace has an equivalent — our defensible novelty there is the entire
  *active* return-time diagnosis + ontology cross-check + seller action loop.
- The ontology is heuristic ("linen should breathe"), not physics — GSM/RET values are
  rarely on listings. Fine for a prototype; label it an ontology, not a lab instrument.
- User review photos are sparse per product; the dataset downloader filters for products
  that actually have them.
