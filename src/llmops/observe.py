"""OBSERVE — "was it healthy?"

Deterministic, $0 health metrics pulled from a trace: latency, tokens, cost,
LLM-call count, transport, errors, convergence. A run is healthy if it converged
(reached a diagnosis) within the latency budget. Note: the tools→JSON fallback
records an errored generation that was then recovered — that is a transport note,
not a health failure, so health keys on `converged`, not raw error count.
"""

# Generous latency budget: on Groq's free tier, per-minute token limits trigger
# backoff that can add 20-30s to a multi-question run. Tune per provider/SLA.
DEFAULT_THRESHOLDS = {"max_latency_ms": 45_000, "max_avg_call_ms": 20_000}


def health(trace: dict, thresholds: dict = None) -> dict:
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    n_calls = max(trace.get("n_llm_calls", 0), 1)
    latency = trace.get("total_latency_ms", 0)
    avg_call = latency / n_calls
    reasons = []
    if not trace.get("converged"):
        reasons.append("run did not converge to a diagnosis")
    if latency > thr["max_latency_ms"]:
        reasons.append(f"total latency {latency/1000:.1f}s > {thr['max_latency_ms']/1000:.0f}s budget")
    if avg_call > thr["max_avg_call_ms"]:
        reasons.append(f"avg LLM-call {avg_call/1000:.1f}s high (likely rate-limit backoff)")
    return {
        "healthy": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {
            "total_latency_ms": latency,
            "avg_call_ms": round(avg_call, 1),
            "n_llm_calls": trace.get("n_llm_calls", 0),
            "n_questions": trace.get("n_questions", 0),
            "tokens_in": trace.get("tokens_in", 0),
            "tokens_out": trace.get("tokens_out", 0),
            "cost_usd": trace.get("cost_usd", 0.0),
            "transport": trace.get("transport", ""),
            "n_errors_recovered": trace.get("n_errors", 0),
        },
    }
