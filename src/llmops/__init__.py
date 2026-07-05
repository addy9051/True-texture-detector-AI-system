"""LLM Ops for the returns concierge — the feedback loop that keeps the agent
honest and healthy.

Maps 1:1 to the architecture: every agent run is TRACED (1 trace/run), then
OBSERVED (was it healthy? — latency, tokens, cost, errors) and EVALUATED (was it
good? — deterministic checks + optional LLM-as-judge). Failures are DIAGNOSED
(where/why), a GATE decides ship-vs-fix, and a passing gate RELEASES a blessed
prompt/config version that feeds back into the agent run.

Local-first and $0 by default (traces in JSONL, judge optional); an optional
Langfuse emitter plugs in via env vars.
"""
