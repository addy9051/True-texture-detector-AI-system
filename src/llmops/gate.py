"""GATE + RELEASE — ship-vs-fix, and bless a config version.

Aggregates the per-run reports into a suite verdict. If enough runs are both
GOOD (eval passed) and HEALTHY, the gate SHIPs — recording a blessed prompt+config
version (the feedback arrow). Otherwise it says FIX and hands back the diagnoses,
so the loop routes to a bug-fix / prompt change rather than a release.
"""

from src.llmops.config import current_config, record_release

DEFAULT_PASS_THRESHOLD = 0.80


def gate(reports: list[dict], provider: str = "", model_id: str = "",
         threshold: float = DEFAULT_PASS_THRESHOLD, do_release: bool = True) -> dict:
    n = len(reports)
    good = [r for r in reports if r["eval"]["passed"] and r["health"]["healthy"]]
    pass_rate = len(good) / n if n else 0.0

    decision = "SHIP" if pass_rate >= threshold else "FIX"

    # collect the distinct fixes the failing runs point at
    fixes = {}
    for r in reports:
        for f in r.get("diagnosis", []):
            fixes.setdefault(f["knob"], set()).add(f["suggested_fix"])
    fix_summary = [{"knob": k, "fixes": sorted(v)} for k, v in fixes.items()]

    result = {
        "decision": decision,
        "pass_rate": round(pass_rate, 3),
        "threshold": threshold,
        "n_runs": n,
        "n_good": len(good),
        "fixes_by_knob": fix_summary,
        "released": None,
    }

    if decision == "SHIP" and do_release:
        entry = record_release(current_config(provider, model_id), pass_rate,
                               note=f"gate passed at {pass_rate:.0%} over {n} runs")
        result["released"] = {"version": entry["version"],
                              "prompt_version": entry["config"]["prompt_version"]}
    return result
