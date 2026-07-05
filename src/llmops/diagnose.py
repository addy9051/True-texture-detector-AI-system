"""DIAGNOSE — "where/why was it broken, and what knob fixes it?"

Turns a failing run (health + eval) into concrete findings, each pointing at the
knob the Release step can turn: the prompt (skill.md), the model/provider config,
the transport, or the retrieval/grounding. This is what makes the loop
actionable rather than just a red X.
"""


def diagnose(trace: dict, health: dict, evaluation: dict) -> list[dict]:
    findings = []

    def add(issue, cause, fix, knob):
        findings.append({"issue": issue, "likely_cause": cause,
                         "suggested_fix": fix, "knob": knob})

    # Hard run error → an infrastructure problem, NOT a prompt/model-quality one.
    # Short-circuit so we don't send the user editing skill.md for a network drop.
    if trace.get("error"):
        add("Agent run failed to execute (infrastructure, not model quality)",
            trace["error"][:200],
            "check network connectivity to the provider (e.g. can you reach api.groq.com?), "
            "the provider's status page, VPN/firewall/proxy, and that the API key is set — "
            "then re-run. This is not a skill.md or model-quality issue.",
            "infrastructure / connectivity")
        return findings

    failed = {a["name"] for a in evaluation["assertions"]
              if a["critical"] and not a["passed"]}

    if "converged" in failed:
        add("Run never produced a diagnosis",
            "the interview loop failed — a provider/tool-config rejection or a parse error the fallback couldn't recover",
            "check the transport fallback and the JSON protocol prompt; verify provider credentials",
            "transport / model config")

    if "case_class_consistent" in failed:
        add("case_class contradicts the diagnosis flags",
            "the model set material/weather flags that don't match the case it reported — response-matrix drift",
            "tighten the RESPONSE MATRIX wording in skill.md; the engine already classifies deterministically, so re-state that the model must not self-assign case_class",
            "prompt (skill.md)")

    if "action_has_recommendation" in failed:
        add("Seller action set but recommendation is null",
            "the model skipped the 'non-NO_ACTION ⇒ concrete recommendation' rule",
            "strengthen that rule in skill.md, or enforce it deterministically in the engine",
            "prompt (skill.md)")

    if "expected_case_match" in failed:
        add("Diagnosis doesn't match the scenario ground truth",
            "the model mis-read the customer's signal (or, under mock, the stub returns a fixed answer)",
            "if live: revise the probing questions / matrix in skill.md or try a stronger model; if mock: expected — the mock is a fixed stub",
            "prompt / model")

    if "schema_valid" in failed:
        add("Diagnosis is missing required fields",
            "the model omitted fields the seller dashboard needs",
            "mark the fields required in the tool schema and restate them in the JSON-protocol template",
            "tool schema / prompt")

    if "question_budget_respected" in failed:
        add("Agent exceeded the question budget",
            "the forced-diagnosis toolChoice didn't fire",
            "verify MAX_QUESTIONS and the toolChoice-forcing / JSON force-diagnosis directive",
            "prompt / model config")

    if not health["healthy"]:
        for r in health["reasons"]:
            add(f"Health: {r}",
                "elevated latency is usually free-tier token-per-minute backoff on Groq",
                "switch to a smaller/faster model (GROQ_MODEL_ID=llama-3.1-8b-instant), trim the system prompt, or use provisioned throughput",
                "model config")

    if trace.get("transport") == "json":
        add("Fell back to the JSON protocol (native tool-use rejected)",
            "the provider/account rejected Converse toolConfig (seen on restricted Bedrock/AISPL accounts)",
            "handled automatically — no action needed; switch provider to restore forced tool-use",
            "provider (note only)")

    judge = evaluation.get("judge")
    if judge and not judge.get("error"):
        if judge.get("neutral_first_question", 5) < 4:
            add("Judge: leading first question",
                "the opening question presumed a defect instead of offering neutral buckets",
                "reinforce 'first question separates neutral buckets' in skill.md",
                "prompt (skill.md)")
        if judge.get("actionable_recommendation", 5) < 4:
            add("Judge: vague seller recommendation",
                "the recommendation didn't name the material or a concrete fix",
                "restate the recommendation requirement with an example in skill.md",
                "prompt (skill.md)")

    return findings
