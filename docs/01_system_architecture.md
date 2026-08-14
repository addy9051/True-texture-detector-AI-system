# 01. System Architecture & Core Design

## Architectural Overview

The **True-Texture Intelligence Platform** is architected as a modular, decoupled AI system designed to operate both in batch analytical mode (mined catalog insights) and low-latency real-time mode (interactive customer return concierge).

```mermaid
flowchart TB
    subgraph Offline Batch Pipeline
        RawData[(Amazon/Myntra Raw Data)] --> Ingest[Dataset Ingestion <br/> src/ingest/download_dataset.py]
        Ingest --> Filter[Semantic Sentence Filter <br/> src/nlp/semantic_filter.py]
        Filter --> Negation[Negation Pruning Engine <br/> src/diagnosis/negation.py]
        Negation --> Diagnosis[Texture Diagnosis Engine <br/> src/diagnosis/diagnose.py]
        Diagnosis --> EvidencePool[(Evidence DB <br/> data/processed/diagnosis.jsonl)]
    end

    subgraph Ground Truth Knowledge
        Physics[(Fabric Physics JSON <br/> data/fabric_physics.json)] --> Ontology[Fabric Ontology API <br/> src/physics/fabric_ontology.py]
        Priors[(Category Materials JSON <br/> data/category_materials.json)] --> CatMaterials[Category Materials API <br/> src/physics/category_materials.py]
    end

    subgraph Online Agentic Service
        User([Customer / Return Portal]) <-->|Initiate Return / Answer| Concierge[Returns Concierge Engine <br/> src/concierge/graph.py]
        Ontology --> Concierge
        CatMaterials --> Concierge
        EvidencePool --> Concierge
        Concierge --> Store[Episodic Insights Store <br/> src/concierge/insights_store.py]
        Store --> SQLite[(SQLite DB <br/> data/processed/insights.sqlite)]
    end

    subgraph Observability & Ops
        Concierge -.-> Tracer[Trace & Metrics Handler <br/> src/llmops/tracer.py]
        Tracer -.-> Langfuse[(Langfuse Platform / JSONL)]
        Tracer --> Evals[LLMOps Eval & Gating <br/> src/llmops/evaluate.py]
        Evals --> Releases[(Releases Registry <br/> data/processed/releases.json)]
    end

    subgraph Presentation Layer
        SQLite --> Dashboard[Streamlit Seller Dashboard <br/> app.py]
        EvidencePool --> Dashboard
    end
```

---

## The 3-Tier Cognitive Memory Architecture

The agentic return concierge utilizes a specialized cognitive memory architecture designed for predictability, zero-latency grounding, and high KV-cache efficiency.

```mermaid
classDiagram
    class SemanticMemory {
        +fabric_physics.json
        +category_materials.json
        +FabricOntology
        +normalize_to_ontology()
        +expected_texture
        +failing_adjectives
        +substitution_signatures
    }
    class ProceduralMemory {
        +skill.md
        +MAX_QUESTIONS = 3
        +_skill_policy()
        +2x2 Response Matrix
        +Remedy Decision Tree
    }
    class EpisodicMemory {
        +insights.sqlite
        +sessions table
        +load_sessions()
        +SQL Recency & Case Filtering
        +Historical ASIN Evidence
    }
    class WorkingMemory {
        +LangGraph MemorySaver
        +ConciergeState
        +messages history
        +questions_asked counter
        +transcript accumulator
    }

    SemanticMemory <.. ConciergeSession : Grounds Physical Facts
    ProceduralMemory <.. ConciergeSession : Enforces Rules of Engagement
    EpisodicMemory <.. ConciergeSession : Queries Prior Complaints & Stores Outcomes
    WorkingMemory <.. ConciergeSession : Tracks Active Multi-Turn Thread
```

### 1. Semantic Memory (Durable Domain Knowledge)
* **Storage**: [`data/fabric_physics.json`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/data/fabric_physics.json) & [`data/category_materials.json`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/data/category_materials.json).
* **Role**: Provides authoritative, deterministic fabric physics. Separates **fiber** expectations (cotton, linen, silk) from **weave/construction** mechanics (corduroy, satin, velvet, fleece).
* **Rationale**: RAG over a small fixed domain ontology introduces vector retrieval misses and adds 150–300ms of latency. Direct injection ensures 100% precision.

### 2. Procedural Memory (*How-to-Act* Policy)
* **Storage**: [`src/concierge/skill.md`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/src/concierge/skill.md).
* **Role**: Governs the interview strategy, maximum question budget, conversational tone, and the deterministic 2×2 Response Matrix (Material Defect vs. Weather Discomfort).
* **Rationale**: Markdown-based policy isolation decouples prompt engineering from Python orchestration code, allowing non-engineering domain experts to refine return policies.

### 3. Episodic Memory (Historical Case & Transcript Records)
* **Storage**: SQLite database ([`data/processed/insights.sqlite`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/data/processed/insights.sqlite)) via [`src/concierge/insights_store.py`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/src/concierge/insights_store.py).
* **Role**: Captures completed sessions, user transcripts, classified outcomes, and token costs. Enables SQL-speed aggregations and feeds prior product evidence into future sessions.

### 4. Working / Short-Term Memory (Session State)
* **Storage**: In-memory LangGraph checkpointer ([`MemorySaver`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/src/concierge/graph.py#L26)).
* **Role**: Maintains state across tool calls and user interactions within an active thread (`thread_id`).

---

## Prefix Optimization & KV-Cache Strategy

A critical design consideration in high-throughput customer support agents is **KV Cache reuse** on LLM inference servers (e.g., Anthropic Prompt Caching, OpenAI automatic caching, vLLM RadixAttention).

### Problem Statement
Because prompt caching performs exact token matching from token `0` forward, placing dynamic per-request metadata (such as product title or ASIN) at the beginning of the `SystemMessage` invalidates the cache across concurrent users returning different items.

### Optimized Architecture
To achieve optimal cache efficiency and strong prompt injection isolation:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SystemMessage (100% Invariant Prefix)                     │ ─── [GLOBAL KV CACHE HIT]
│    - Role definition                                        │     Cached across all tenants,
│    - Procedural Memory (skill.md)                           │     users, and products.
├─────────────────────────────────────────────────────────────┤
│ 2. HumanMessage / Turn 1 Context Payload                    │ ─── [DYNAMIC TAIL DELTA]
│    <case_context>                                           │     Computed once per session
│       <product title="..." listed_materials="..." />        │     as a lightweight tail (~200 tokens).
│       <fabric_ontology> ... </fabric_ontology>              │
│       <weave_construction> ... </weave_construction>        │
│       <prior_evidence> ... </prior_evidence>                │
│    </case_context>                                          │
│    The customer just clicked 'Return item'. Begin interview.│
└─────────────────────────────────────────────────────────────┘
```

1. **Global Cache Hit on System Prompt**: The system prompt is 100% static, guaranteeing that concurrent requests across thousands of distinct products share the identical pre-warmed KV cache.
2. **Multi-Turn Cache Hit within Session**: On Turns 2, 3, and 4, the entire prefix (System prompt + Initial Case Context + Turn 1 messages) is 100% cached; only the customer's delta response is processed.
3. **Prompt Injection Defense**: Product titles and user-generated review texts are enclosed in structured XML tags within the user space rather than executing in the authoritative system instruction space.

---

## Multi-Turn Interaction Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor Customer
    participant App as Concierge UI / Frontend
    participant Graph as LangGraph Engine (graph.py)
    participant LLM as Foundation Model (Portkey)
    participant Tracer as Langfuse / Local Tracer
    participant DB as SQLite Insights Store

    Customer->>App: Clicks "Return Product" (ASIN: B001...)
    App->>Graph: ConciergeSession.start()
    Graph->>LLM: Invoke with Static System Prompt + Dynamic Case Context
    LLM-->>Graph: ToolCall: ask_question(question, options)
    Graph->>Graph: Interrupt Execution & Yield Question Event
    Graph-->>App: Return question payload & UI options
    App-->>Customer: Render Question 1 (e.g. "How does the fabric feel?")

    Customer->>App: Selects "Feels slick and plasticky"
    App->>Graph: ConciergeSession.answer("Feels slick and plasticky")
    Graph->>LLM: Resume with ToolMessage(customer_answer)
    
    alt Needs Follow-Up (questions < 3)
        LLM-->>Graph: ToolCall: ask_question("In what weather was it worn?", options)
        Graph-->>Customer: Render Question 2
        Customer->>Graph: Answers Question 2
    end

    LLM-->>Graph: ToolCall: submit_diagnosis(payload)
    Graph->>Graph: enrich_diagnosis() with Ground Truth Facts
    Graph->>DB: Save Session (Diagnosis, Transcripts, Cost, Case Class)
    Graph->>Tracer: Emit Structured Trace Event
    Graph-->>App: Return Final Grounded Diagnosis & Customer Message
    App-->>Customer: Display empathetic customer closing message
```
