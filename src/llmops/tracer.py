"""TRACE — one structured trace per agent run.

A concierge run = user starts a return → agent asks 1-3 questions → emits a
diagnosis. This records the *tree of events* for that run: every LLM call with
its latency / tokens / whether tools were used / errors, plus the interview
transcript and the final structured diagnosis.

`TracingChat` wraps ANY chat client (Groq/Gemini/Bedrock/mock) transparently —
it times each `converse`, reads token deltas off the client's own CostMeter, and
notes tool-config use and errors (e.g. the tools→JSON fallback shows up as an
errored generation followed by a successful one). No change to the concierge
engine is required.

Traces append to data/processed/traces.jsonl. If LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY are set and `langfuse` is installed, each trace is also
emitted there — otherwise the local JSONL is the trace store.
"""

import json
import os
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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
            "transport": transport,          # "tools" | "json"
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


class TracingChat:
    """Transparent wrapper: proxies a chat client, records each call to a RunTrace."""

    def __init__(self, inner, trace: RunTrace):
        self.inner = inner
        self.trace = trace
        trace.model_id = getattr(inner, "model_id", "")

    # proxy the duck-typed surface the engine relies on
    @property
    def model_id(self):
        return self.inner.model_id

    @property
    def region(self):
        return getattr(self.inner, "region", "")

    @property
    def meter(self):
        return self.inner.meter

    @staticmethod
    def _lf_input(messages):
        """Only the conversation turns (the big system prompt is passed
        separately and NOT logged — per the skill's 'set only relevant input')."""
        turns = []
        for msg in messages:
            for b in msg.get("content", []):
                if "text" in b:
                    turns.append({"role": msg.get("role"), "text": b["text"][:600]})
                elif "toolUse" in b:
                    turns.append({"role": "assistant", "tool": b["toolUse"].get("name")})
                elif "toolResult" in b:
                    inner = (b["toolResult"].get("content") or [{}])[0].get("json", {})
                    turns.append({"role": "tool_result", "answer": inner})
        return turns

    @staticmethod
    def _lf_output(resp):
        content = resp.get("output", {}).get("message", {}).get("content", [])
        for b in content:
            if "toolUse" in b:
                return {"tool": b["toolUse"].get("name"), "input": b["toolUse"].get("input")}
        return {"text": " ".join(b.get("text", "") for b in content)[:600]}

    def converse(self, system, messages, tool_config=None, *args, **kwargs):
        m = self.inner.meter
        in0, out0 = m.input_tokens, m.output_tokens
        t0 = time.perf_counter()
        used_tools = tool_config is not None
        forced = bool(tool_config and tool_config.get("toolChoice"))
        # Real-time Langfuse generation — the span's duration is the actual
        # wall-clock of the call (best practice), and it nests under the run's
        # root span. A no-op context when Langfuse isn't configured.
        gen_cm = (_client().start_as_current_observation(
                      name="llm-call", as_type="generation", model=self.inner.model_id,
                      input=self._lf_input(messages),
                      metadata={"used_tools": str(used_tools), "forced_tool": str(forced)})
                  if langfuse_enabled() else nullcontext())
        with gen_cm as gen:
            try:
                resp = self.inner.converse(system, messages, tool_config, *args, **kwargs)
            except Exception as e:
                self.trace.add_generation((time.perf_counter() - t0) * 1000, 0, 0,
                                          used_tools, forced,
                                          error=f"{type(e).__name__}: {str(e)[:140]}")
                if gen is not None:
                    gen.update(level="ERROR", status_message=str(e)[:200])
                raise
            ti, to = m.input_tokens - in0, m.output_tokens - out0
            self.trace.add_generation((time.perf_counter() - t0) * 1000, ti, to,
                                      used_tools, forced, stop_reason=resp.get("stopReason"))
            if gen is not None:
                gen.update(output=self._lf_output(resp),
                           usage_details={"input": ti, "output": to, "total": ti + to})
        return resp


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
    from src.concierge.provider import _env_or_registry
    return bool(_env_or_registry("LANGFUSE_PUBLIC_KEY")
                and _env_or_registry("LANGFUSE_SECRET_KEY"))


def _client():
    # promote keys from the Windows registry into os.environ so the Langfuse
    # SDK (which reads env at construction) picks them up without a new shell
    from src.concierge.provider import _env_or_registry
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        _env_or_registry(k)
    from langfuse import get_client
    return get_client()


def emit_langfuse(trace_dict: dict, evaluation: dict | None = None) -> str | None:
    """Mirror a completed run to Langfuse Cloud as a trace: one root `agent`
    observation with a child `generation` per LLM call (tokens → Langfuse's
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
