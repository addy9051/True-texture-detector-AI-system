"""TRACE — one structured trace per agent run.

A concierge run = user starts a return → agent asks 1-3 questions → emits a
diagnosis. This records the *tree of events* for that run: every LLM call with
its latency / tokens / whether tools were used / errors, plus the interview
transcript and the final structured diagnosis.

``TracingCallbackHandler`` hooks into LangChain's callback system so it works
transparently with the LangGraph agent — no wrapper around the chat client is
needed.

Traces append to data/processed/traces.jsonl. If LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY are set and ``langfuse`` is installed, each trace is also
emitted there — otherwise the local JSONL is the trace store.
"""

import json
import os
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

TRACES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "traces.jsonl"


@dataclass
class RunTrace:
    scenario: str = ""
    parent_asin: str = ""
    title: str = ""
    provider: str = ""
    model_id: str = ""
    expected_case: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    events: list[dict] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def add_generation(self, latency_ms, tokens_in, tokens_out, used_tools,
                       forced=False, stop_reason=None, error=None):
        self.events.append({
            "seq": len(self.events),
            "kind": "generation",
            "latency_ms": round(latency_ms, 1),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "used_tools": used_tools,
            "forced_tool": forced,
            "stop_reason": stop_reason,
            "error": error,
        })

    def finish(self, diagnosis: dict, transcript: list[dict], transport: str,
               questions: int, cost_usd: float, converged: bool,
               error: str | None = None) -> dict:
        gens = self.events
        return {
            "error": error,                  # hard run error (e.g. provider unreachable)
            "run_id": self.run_id,
            "started_at": self.started_at,
            "scenario": self.scenario,
            "parent_asin": self.parent_asin,
            "title": self.title,
            "provider": self.provider,
            "model_id": self.model_id,
            "expected_case": self.expected_case,
            "transport": transport,          # "langgraph" for the new agent
            "n_llm_calls": len([e for e in gens if not e["error"]]),
            "n_questions": questions,
            "converged": converged,
            "total_latency_ms": round(sum(e["latency_ms"] for e in gens), 1),
            "tokens_in": sum(e["tokens_in"] for e in gens),
            "tokens_out": sum(e["tokens_out"] for e in gens),
            "cost_usd": round(cost_usd, 6),
            "n_errors": len([e for e in gens if e["error"]]),
            "events": gens,
            "transcript": transcript,
            "diagnosis": diagnosis,
        }


class TracingCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records each LLM call to a RunTrace.

    Replaces the old ``TracingChat`` wrapper — works with any LangChain model
    (ChatOpenAI via Portkey, mock, etc.) and plugs into LangGraph's callback
    system transparently.
    """

    def __init__(self, trace: RunTrace):
        self.trace = trace
        self._t0: float | None = None
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0

    # Cost approximation: accumulate token counts; the cost is computed
    # from the trace after the run (same as before).
    @property
    def usd(self) -> float:
        """Approximate cost — Portkey tracks the real cost in its dashboard."""
        return 0.0  # Free tier; real cost tracked by Portkey

    def summary(self) -> str:
        return (f"{self.total_tokens_in} in / {self.total_tokens_out} out tokens"
                f" = ${self.usd:.4f}")

    # ---- LangChain callback hooks ----

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs):
        self._t0 = time.perf_counter()

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        self._t0 = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs):
        latency_ms = (time.perf_counter() - (self._t0 or time.perf_counter())) * 1000
        tokens_in = 0
        tokens_out = 0
        used_tools = False

        # Extract token usage from the response
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    used_tools = True

        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.trace.add_generation(latency_ms, tokens_in, tokens_out, used_tools)

    def on_llm_error(self, error: BaseException, **kwargs):
        latency_ms = (time.perf_counter() - (self._t0 or time.perf_counter())) * 1000
        self.trace.add_generation(
            latency_ms, 0, 0, False,
            error=f"{type(error).__name__}: {str(error)[:140]}")


def save_trace(trace_dict: dict, path: Path = TRACES_PATH) -> None:
    """Append the trace to the local JSONL store (always, $0). Langfuse emission
    is separate — see emit_langfuse — so it can include the eval scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace_dict, ensure_ascii=False) + "\n")


def load_traces(path: Path = TRACES_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def langfuse_enabled() -> bool:
    # registry-aware: setx-set keys work even in a terminal opened before setx
    from src.concierge.portkey_llm import _env_or_registry
    return bool(_env_or_registry("LANGFUSE_PUBLIC_KEY")
                and _env_or_registry("LANGFUSE_SECRET_KEY"))


def _client():
    # promote keys from the Windows registry into os.environ so the Langfuse
    # SDK (which reads env at construction) picks them up without a new shell
    from src.concierge.portkey_llm import _env_or_registry
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        _env_or_registry(k)
    from langfuse import get_client
    return get_client()


def emit_langfuse(trace_dict: dict, evaluation: dict | None = None) -> str | None:
    """Mirror a completed run to Langfuse Cloud as a trace: one root ``agent``
    observation with a child ``generation`` per LLM call (tokens → Langfuse's
    usage/cost dashboards, errors flagged), plus the eval/judge scores attached
    to the trace. Returns the trace URL. Never fatal — observability must not
    break the run. No-op unless LANGFUSE_* env vars are set.

    Uses the Langfuse v4 (OpenTelemetry) SDK: nested start_as_current_observation
    context managers guarantee correct parent→child nesting.
    """
    if not langfuse_enabled():
        return None
    try:
        lf = _client()
        from langfuse import propagate_attributes
        dx = trace_dict.get("diagnosis") or {}

        metadata = {
            "provider": str(trace_dict["provider"]), "model_id": str(trace_dict["model_id"]),
            "transport": str(trace_dict["transport"]), "run_id": str(trace_dict["run_id"]),
            "n_questions": str(trace_dict["n_questions"]),
            "total_latency_ms": str(trace_dict["total_latency_ms"])
        }

        with propagate_attributes(
            session_id=trace_dict["run_id"],
            tags=[trace_dict["provider"], trace_dict["scenario"]],
            metadata=metadata
        ):
            with lf.start_as_current_observation(
                name=f"concierge:{trace_dict['scenario']}", as_type="span",
                input={"scenario": trace_dict["scenario"], "asin": trace_dict["parent_asin"],
                       "title": trace_dict["title"]},
                output={"case_class": dx.get("case_class"),
                        "seller_action": dx.get("seller_action"),
                        "converged": trace_dict["converged"]},
                cost_details={"total": float(trace_dict.get("cost_usd", 0.0))},
            ) as root:
                trace_id = root.trace_id
                for e in trace_dict["events"]:
                    with lf.start_as_current_observation(
                        name=f"llm_call_{e['seq']}", as_type="generation",
                        model=trace_dict["model_id"],
                        usage_details={"input": e["tokens_in"], "output": e["tokens_out"],
                                       "total": e["tokens_in"] + e["tokens_out"]},
                        metadata={"latency_ms": str(e["latency_ms"]), "used_tools": str(e["used_tools"]),
                                  "forced_tool": str(e["forced_tool"]), "stop_reason": str(e["stop_reason"])},
                        level="ERROR" if e["error"] else "DEFAULT",
                        status_message=e["error"] or None,
                    ):
                        pass
                if evaluation is not None:
                    root.score_trace(name="eval_passed",
                                     value=1.0 if evaluation.get("passed") else 0.0)
                    j = evaluation.get("judge")
                    if j and not j.get("error"):
                        root.score_trace(name="judge_avg", value=float(j.get("avg", 0)))
        lf.flush()
        return lf.get_trace_url(trace_id=trace_id)
    except Exception as ex:
        print(f"    (langfuse emit skipped: {type(ex).__name__}: {str(ex)[:100]})")
        return None
