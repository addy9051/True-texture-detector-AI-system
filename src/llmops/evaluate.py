"""EVAL — "was it good?"

Two layers:
  1. Deterministic checks ($0, always run): schema validity, question-budget,
     convergence, case-class consistency (recompute from the diagnosis's own
     flags and compare), the "action ⇒ recommendation" rule, grounding, and
     (when a scenario ships a ground-truth) the expected-case match.
  2. Optional LLM-as-judge (needs a model): scores the *qualitative* things a
     rule can't — did Q1 avoid leading the witness, is the diagnosis grounded in
     the claimed material, does the customer message fit the response matrix, is
     the seller recommendation specific and actionable. Off by default; pass a
     chat client to enable.

An assertion marked `critical` must pass for the run to be "good". Non-critical
ones are informative.
"""

import json
import re

from src.concierge.concierge import MAX_QUESTIONS, classify_case

_JSON = re.compile(r"\{.*\}", re.S)


def _assert(name, passed, critical, detail=""):
    return {"name": name, "passed": bool(passed), "critical": critical, "detail": detail}


def deterministic_checks(trace: dict) -> list[dict]:
    dx = trace.get("diagnosis") or {}
    checks = []

    required = ["root_cause_category", "material_issue_suspected", "seller_action",
                "customer_closing_message", "case_class", "confidence"]
    missing = [k for k in required if k not in dx]
    checks.append(_assert("schema_valid", not missing, True,
                          f"missing: {missing}" if missing else "all required fields present"))

    checks.append(_assert("converged", trace.get("converged"), True,
                          "reached a diagnosis" if trace.get("converged") else "no diagnosis emitted"))

    q = trace.get("n_questions", 0)
    checks.append(_assert("question_budget_respected", q <= MAX_QUESTIONS, True,
                          f"asked {q} (budget {MAX_QUESTIONS})"))

    expected_case = classify_case(dx.get("material_issue_suspected"),
                                  dx.get("weather_suitability_mismatch"))
    consistent = dx.get("case_class") == expected_case
    checks.append(_assert("case_class_consistent", consistent, True,
                          f"case_class={dx.get('case_class')} vs flags→{expected_case}"))

    action = dx.get("seller_action")
    rec = dx.get("listing_fix_recommendation")
    ok_rec = (action == "NO_ACTION") or bool(rec)
    checks.append(_assert("action_has_recommendation", ok_rec, True,
                          "NO_ACTION" if action == "NO_ACTION"
                          else ("recommendation present" if rec else "action set but recommendation is null")))

    grounded = bool(dx.get("material_ground_truth") or dx.get("weave_ground_truth")
                    or dx.get("claimed_materials"))
    checks.append(_assert("grounded_in_ontology", grounded, False,
                          "engine injected material/weave ground truth" if grounded
                          else "no ground truth (material may be unlisted)"))

    exp = trace.get("expected_case")
    if exp:
        checks.append(_assert("expected_case_match", dx.get("case_class") == exp, True,
                              f"got {dx.get('case_class')}, expected {exp}"))
    return checks


JUDGE_SYSTEM = """You are a strict QA reviewer for a fashion-returns concierge agent.
Score the run 1-5 on each dimension (5 = excellent). Return ONLY a JSON object:
{"neutral_first_question": n, "grounded_diagnosis": n, "matrix_appropriate_message": n,
 "actionable_recommendation": n, "notes": "<one short sentence>"}
Rubric:
- neutral_first_question: did the FIRST question offer neutral, non-leading options
  (not presuming a defect)?
- grounded_diagnosis: is the diagnosis consistent with the claimed material's real
  properties and the customer's words (no invented facts)?
- matrix_appropriate_message: does the customer message fit the case — apology for a
  defect, weather education for wrong-weather, etc. — and stay warm/non-blaming?
- actionable_recommendation: is the seller recommendation specific and useful (names
  the material + a concrete fix), or vague/empty?"""


def judge_run(trace: dict, chat) -> dict | None:
    """Optional LLM-as-judge. `chat` is any provider client (Groq/Gemini/...)."""
    dx = trace.get("diagnosis") or {}
    convo = "\n".join(
        (f"Q: {t.get('question')} [options: {', '.join(t.get('options', []))}]"
         if t.get("role") == "concierge" else
         f"Customer: {t.get('text')}" if t.get("role") == "customer" else "")
        for t in trace.get("transcript", [])).strip()
    payload = (f"CLAIMED MATERIALS: {dx.get('claimed_materials')}\n"
               f"INTERVIEW:\n{convo}\n\nDIAGNOSIS:\n{json.dumps(dx, ensure_ascii=False)}")
    try:
        resp = chat.converse(system=JUDGE_SYSTEM,
                             messages=[{"role": "user", "content": [{"text": payload}]}],
                             max_tokens=400, temperature=0.0)
        text = " ".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        m = _JSON.search(text)
        scores = json.loads(m.group()) if m else None
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    if not scores:
        return {"error": "judge returned no parseable JSON"}
    dims = ["neutral_first_question", "grounded_diagnosis",
            "matrix_appropriate_message", "actionable_recommendation"]
    vals = [scores.get(d, 0) for d in dims]
    scores["avg"] = round(sum(vals) / len(vals), 2)
    scores["passed"] = scores["avg"] >= 4.0
    return scores


def evaluate(trace: dict, judge_chat=None) -> dict:
    # A hard run error (provider unreachable, etc.) means there's nothing to
    # evaluate — don't run the content checks and produce a misleading cascade.
    if trace.get("error"):
        return {"assertions": [_assert("agent_run_completed", False, True, trace["error"])],
                "n_passed": 0, "n_total": 1, "passed": False,
                "critical_failures": ["agent_run_completed"]}
    checks = deterministic_checks(trace)
    critical_fail = [c for c in checks if c["critical"] and not c["passed"]]
    result = {
        "assertions": checks,
        "n_passed": sum(c["passed"] for c in checks),
        "n_total": len(checks),
        "passed": len(critical_fail) == 0,
        "critical_failures": [c["name"] for c in critical_fail],
    }
    if judge_chat is not None:
        result["judge"] = judge_run(trace, judge_chat)
        j = result["judge"]
        if j and not j.get("error") and not j.get("passed", True):
            result["passed"] = False
    return result
