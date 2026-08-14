# 01. System Architecture & Core Design

## Architectural Overview

The **True-Texture Intelligence Platform** is architected as a modular, decoupled AI system designed to operate both in batch analytical mode (mined catalog insights) and low-latency real-time mode (interactive customer return concierge).

```mermaid
flowchart TD
    subgraph Offline Batch Pipeline
        RawData[(Catalog & Reviews)] --> NLP[NLP Mismatch Engine]
        Physics[(Fabric Physics)] --> NLP
        NLP --> EvidencePool[(Evidence Pool)]
    end

    subgraph Online Agentic Service
        Customer([Customer]) <--> Concierge[LangGraph Returns Concierge]
        Physics --> Concierge
        EvidencePool --> Concierge
        Concierge --> SQLite[(Episodic SQLite Store)]
    end

    subgraph Operations & Presentation
        Concierge -.-> LLMOps[LLMOps & Telemetry]
        LLMOps --> Releases[(Releases Registry)]
        SQLite --> Dashboard[Streamlit Dashboard]
        EvidencePool --> Dashboard
    end
```

#### Pipeline Highlights
* **Offline Ingestion & NLP**: Ingests raw reviews, applies `MiniLM-L6-v2` semantic filtering (0.50 threshold), and performs negation-aware adjective extraction to produce `diagnosis.jsonl`.
* **Online Returns Concierge**: Multi-turn LangGraph agent that diagnoses returns in $\le 3$ questions using grounded physics and past review evidence.
* **Storage & Operations**: Persists session transcripts to `insights.sqlite`, streams telemetry to Langfuse, and feeds real-time metrics to the Streamlit UI.

---

## The 3-Tier Cognitive Memory Architecture

The agentic return concierge utilizes a specialized cognitive memory architecture designed for predictability, zero-latency grounding, and high KV-cache efficiency.

```mermaid
graph TD
    A[Returns Concierge Session] --> B[1. Semantic Memory <br/> Durable Facts]
    A --> C[2. Procedural Memory <br/> How-to-Act Policy]
    A --> D[3. Episodic Memory <br/> Case History]
    A --> E[4. Working Memory <br/> Active Session]
```

#### Memory Tier Specifications

1. **Semantic Memory (Durable Domain Knowledge)**
   * **Storage**: [`data/fabric_physics.json`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/data/fabric_physics.json) & [`data/category_materials.json`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/data/category_materials.json).
   * **Role**: Provides authoritative, deterministic fabric physics. Separates **fiber** expectations (cotton, linen, silk) from **weave/construction** mechanics (corduroy, satin, velvet, fleece).
   * **Engineering Decision**: Direct injection into the prompt rather than RAG over a vector DB. Direct injection eliminates retrieval misses and removes 150–300ms of DB latency.

2. **Procedural Memory (*How-to-Act* Policy)**
   * **Storage**: [`src/concierge/skill.md`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/src/concierge/skill.md).
   * **Role**: Governs the interview strategy, maximum question budget (`MAX_QUESTIONS = 3`), conversational tone, and the deterministic 2×2 Response Matrix (Material Defect vs. Weather Discomfort).
   * **Engineering Decision**: Markdown policy isolation decouples prompt engineering from Python orchestration code.

3. **Episodic Memory (Historical Case & Transcript Records)**
   * **Storage**: SQLite database ([`data/processed/insights.sqlite`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/data/processed/insights.sqlite)) via [`src/concierge/insights_store.py`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/src/concierge/insights_store.py).
   * **Role**: Captures completed sessions, user transcripts, classified outcomes, and token costs. Enables SQL-speed aggregations and feeds prior product evidence into future sessions.

4. **Working / Short-Term Memory (Session State)**
   * **Storage**: In-memory LangGraph checkpointer ([`MemorySaver`](file:///d:/Downloads/projects/True-texture%20detector%20AI%20system/src/concierge/graph.py#L26)).
   * **Role**: Maintains state across tool calls and user interactions within an active thread (`thread_id`).

---

## Prefix Optimization & KV-Cache Strategy

A critical design consideration in high-throughput customer support agents is **KV Cache reuse** on LLM inference servers (e.g., Anthropic Prompt Caching, OpenAI automatic caching, vLLM RadixAttention).

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

* **Global Cache Hit on System Prompt**: The system prompt is 100% static, guaranteeing that concurrent requests across thousands of distinct products share the identical pre-warmed KV cache.
* **Multi-Turn Cache Hit within Session**: On Turns 2, 3, and 4, the entire prefix (System prompt + Initial Case Context + Turn 1 messages) is 100% cached; only the customer's delta response is processed.
* **Prompt Injection Defense**: Product titles and user-generated review texts are enclosed in structured XML tags within the user space rather than executing in the authoritative system instruction space.

---

## Multi-Turn Interaction Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor Customer
    participant App as Concierge UI
    participant Graph as LangGraph Engine
    participant LLM as Foundation Model
    participant DB as SQLite DB

    Customer->>App: Initiate Return
    App->>Graph: start()
    Graph->>LLM: Invoke (Static System + Case Context)
    LLM-->>Graph: ToolCall: ask_question
    Graph->>Graph: Interrupt & Pause
    Graph-->>App: Render Question & Options
    Customer->>App: Submits Answer
    App->>Graph: answer(text)
    Graph->>LLM: Resume with ToolMessage
    LLM-->>Graph: ToolCall: submit_diagnosis
    Graph->>DB: Save Session & Transcripts
    Graph-->>App: Return Diagnosis & Resolution
    App-->>Customer: Render Closing Message
```

#### Lifecycle Step Highlights
1. **Initiation**: Customer clicks "Return" on a product.
2. **Context Setup**: `ConciergeSession` loads static system prompt and context-engineered XML payload.
3. **Execution & Interrupt**: LangGraph invokes the LLM, intercepts the `ask_question` tool call, and pauses via `interrupt()`.
4. **Resumption**: Customer selects an answer; `Command(resume=...)` updates the thread state.
5. **Finalization & Enrichment**: On `submit_diagnosis`, the engine grounds facts against the physics ontology and saves the final record to SQLite.
