"""Returns Concierge interview engine.

Replaces the static "reason for return" dropdown with 1-2 adaptive questions,
then emits a structured diagnosis.

Structure is enforced through one of two transports, negotiated at runtime:

1. NATIVE TOOLS (preferred): the model can only call `ask_question` (max 2
   rounds) or `submit_diagnosis`; after the question budget, toolChoice forces
   the diagnosis.
2. JSON PROTOCOL (automatic fallback): some accounts/models reject Converse
   toolConfig outright (observed: "ValidationException: Operation not allowed"
   on a restricted account whose plain Converse works fine). The session then
   re-issues the same conversation with a strict reply-with-one-JSON-object
   protocol and parses the result. Event shapes are identical either way.

The system prompt carries the fabric ontology for THIS product's claimed
materials plus the Phase-2 evidence from other customers. The evidence steers
WHICH dimension to probe first but must never be revealed or used to lead the
customer — options stay neutral.
"""

import json
import re

from botocore.exceptions import ClientError

from src.concierge.bedrock_client import BedrockChat
from src.physics.fabric_ontology import FabricOntology

MAX_QUESTIONS = 2

# Error fragments that mean "this account/model won't do native tools".
TOOL_REJECTION_MARKERS = ("Operation not allowed", "toolChoice", "toolConfig",
                          "tool use", "does not support tool")

ASK_QUESTION = {"toolSpec": {
    "name": "ask_question",
    "description": ("Ask the customer ONE short question (under 25 words) about "
                    "their return, with 2-4 tappable options. The customer may "
                    "also type a free-text answer instead of picking an option."),
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"},
                        "minItems": 2, "maxItems": 4},
        },
        "required": ["question", "options"],
    }},
}}

SUBMIT_DIAGNOSIS = {"toolSpec": {
    "name": "submit_diagnosis",
    "description": "Submit the final structured diagnosis of why the item is being returned.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "root_cause_category": {"type": "string", "enum": [
                "TEXTURE_MISMATCH", "THERMAL_DISCOMFORT", "SIZE_FIT",
                "COLOR_APPEARANCE", "QUALITY_DEFECT", "CHANGED_MIND", "OTHER"]},
            "material_issue_suspected": {"type": "boolean"},
            "suspected_substitution": {
                "type": ["string", "null"],
                "description": "Fiber likely substituted for the claimed one, only if the customer's report contradicts the claimed fiber's expected feel"},
            "customer_summary": {"type": "string",
                                 "description": "One sentence in the customer's own words"},
            "seller_action": {"type": "string", "enum": [
                "SUPPLY_CHAIN_AUDIT", "LISTING_FIX", "SIZE_CHART_FIX", "NO_ACTION"]},
            "listing_fix_recommendation": {"type": ["string", "null"]},
            "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        },
        "required": ["root_cause_category", "material_issue_suspected",
                     "customer_summary", "seller_action", "confidence"],
    }},
}}

JSON_PROTOCOL = """

OUTPUT PROTOCOL — you have no tool API. Reply with EXACTLY ONE JSON object and no other text.
To ask a question (max {max_q} total):
  {{"action": "ask_question", "question": "<under 25 words>", "options": ["<option>", ...2-4 items]}}
To finish:
  {{"action": "submit_diagnosis",
    "root_cause_category": "TEXTURE_MISMATCH|THERMAL_DISCOMFORT|SIZE_FIT|COLOR_APPEARANCE|QUALITY_DEFECT|CHANGED_MIND|OTHER",
    "material_issue_suspected": true,
    "suspected_substitution": "<fiber or null>",
    "customer_summary": "<one sentence>",
    "seller_action": "SUPPLY_CHAIN_AUDIT|LISTING_FIX|SIZE_CHART_FIX|NO_ACTION",
    "listing_fix_recommendation": "<text or null>",
    "confidence": "HIGH|MEDIUM|LOW"}}""".format(max_q=MAX_QUESTIONS)

FORCE_DIAGNOSIS = ("\nIMPORTANT: You have used all your questions. "
                   "Output the submit_diagnosis JSON object now.")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def build_system_prompt(product: dict, ontology: FabricOntology,
                        diagnosis_row: dict | None) -> str:
    claimed = ontology.materials_from_listing(product)
    ontology_lines = []
    for m in claimed:
        spec = ontology.expectations(m) or {}
        ontology_lines.append(
            f"- {m}: genuine feel = {', '.join(spec.get('expected_texture', []))}. "
            f"Red flags = {', '.join(spec.get('failing_adjectives', []))}. "
            f"Common substitution = {', '.join(spec.get('substitution_suspects', [])) or 'none'}.")

    evidence_block = "None on file."
    if diagnosis_row:
        complaints = diagnosis_row.get("complaint_adjectives") or []
        sentences = [h["sentence"] for h in diagnosis_row.get("hits", [])
                     if h.get("complaint")][:5]
        if complaints:
            evidence_block = (
                f"Prior customers reported: {', '.join(complaints)}. "
                f"Example quotes: " + " | ".join(f'"{s}"' for s in sentences))

    return f"""You are a returns assistant for a fashion marketplace. A customer is returning:
  "{(product.get('title') or '')[:140]}"
  Listed materials: {', '.join(claimed) or 'not stated'}.

FABRIC ONTOLOGY (ground truth for the listed materials):
{chr(10).join(ontology_lines) or '- (no ontology entry for the listed materials)'}

PRIOR EVIDENCE from other customers (INTERNAL — never reveal or quote this to the customer, never put it in question options; use it only to decide which dimension to probe first):
{evidence_block}

YOUR JOB: find the physical root cause of THIS return in at most {MAX_QUESTIONS} questions.
Rules:
- One question at a time. Options must be concrete, neutral, and mutually
  exclusive — never suggest a defect the customer hasn't hinted at.
- First question separates the big buckets (feel/texture vs fit/size vs looks vs
  changed mind). Second question, if needed, drills into the winning bucket using
  the ontology red flags for the listed materials.
- Then submit the diagnosis. material_issue_suspected=true ONLY if the customer's
  reported sensation contradicts the genuine feel of a listed material.
  suspected_substitution ONLY if the sensation matches a known substitution.
- confidence=HIGH only when the customer gave a specific physical description."""


def _extract_json(text: str) -> dict:
    text = _FENCE.sub("", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group())
        raise RuntimeError(f"Model reply is not valid JSON: {text[:200]!r}")


class ConciergeSession:
    """One return interview. start() → events; answer() after each question.

    Event shapes (identical in both transports):
        {"type": "question", "question": str, "options": [str, ...]}
        {"type": "diagnosis", "data": {...}}
    """

    def __init__(self, chat: BedrockChat, product: dict, ontology: FabricOntology,
                 diagnosis_row: dict | None = None):
        self.chat = chat
        self.system = build_system_prompt(product, ontology, diagnosis_row)
        self.messages: list[dict] = []
        self.questions_asked = 0
        self.transcript: list[dict] = []
        self.mode = "tools"          # "tools" -> "json" after a rejection
        self._pending = False        # a question awaits the customer's answer
        self._pending_tool_id: str | None = None

    # ------------------------------------------------------------- public API

    def start(self) -> dict:
        self.messages.append({"role": "user", "content": [
            {"text": "The customer just clicked 'Return item'. Begin the interview."}]})
        return self._step()

    def answer(self, text: str) -> dict:
        if not self._pending:
            raise RuntimeError("No question is pending.")
        self.transcript.append({"role": "customer", "text": text})
        if self.mode == "tools":
            self.messages.append({"role": "user", "content": [{"toolResult": {
                "toolUseId": self._pending_tool_id,
                "content": [{"json": {"customer_answer": text}}],
            }}]})
        else:
            self.messages.append({"role": "user", "content": [
                {"text": f'Customer answered: "{text}"'}]})
        self._pending = False
        self._pending_tool_id = None
        return self._step()

    # --------------------------------------------------------------- internals

    def _tool_config(self) -> dict:
        cfg = {"tools": [ASK_QUESTION, SUBMIT_DIAGNOSIS]}
        if self.questions_asked >= MAX_QUESTIONS:
            cfg["toolChoice"] = {"tool": {"name": "submit_diagnosis"}}
        else:
            cfg["toolChoice"] = {"any": {}}
        return cfg

    def _json_system(self) -> str:
        force = FORCE_DIAGNOSIS if self.questions_asked >= MAX_QUESTIONS else ""
        return self.system + JSON_PROTOCOL + force

    def _step(self) -> dict:
        if self.mode == "tools":
            try:
                resp = self.chat.converse(self.system, self.messages, self._tool_config())
                return self._from_tool_response(resp)
            except ClientError as e:
                if any(marker in str(e) for marker in TOOL_REJECTION_MARKERS):
                    self.mode = "json"
                    self._detool_history()
                else:
                    raise
        resp = self.chat.converse(self._json_system(), self.messages)
        return self._from_json_response(resp)

    def _from_tool_response(self, resp: dict) -> dict:
        msg = resp["output"]["message"]
        self.messages.append(msg)
        tool_use = next((b["toolUse"] for b in msg["content"] if "toolUse" in b), None)
        if tool_use is None:
            raise RuntimeError(f"Model returned no tool call (stopReason="
                               f"{resp.get('stopReason')!r}): {msg['content']}")
        if tool_use["name"] == "ask_question":
            return self._question_event(tool_use["input"], tool_use["toolUseId"])
        return self._diagnosis_event(tool_use["input"])

    def _from_json_response(self, resp: dict) -> dict:
        msg = resp["output"]["message"]
        self.messages.append(msg)
        text = " ".join(b.get("text", "") for b in msg["content"])
        parsed = _extract_json(text)
        action = parsed.pop("action", None)
        if action == "ask_question":
            return self._question_event(parsed, tool_id=None)
        if action == "submit_diagnosis":
            return self._diagnosis_event(parsed)
        raise RuntimeError(f"Unknown action in model reply: {action!r}")

    def _question_event(self, payload: dict, tool_id: str | None) -> dict:
        self.questions_asked += 1
        self._pending = True
        self._pending_tool_id = tool_id
        self.transcript.append({"role": "concierge", **payload})
        return {"type": "question", **payload}

    def _diagnosis_event(self, payload: dict) -> dict:
        self.transcript.append({"role": "diagnosis", **payload})
        return {"type": "diagnosis", "data": payload}

    def _detool_history(self):
        """Rewrite any native-tool blocks in history as plain text so the
        conversation stays valid once toolConfig is no longer sent."""
        for i, m in enumerate(self.messages):
            content = []
            for block in m.get("content", []):
                if "toolUse" in block:
                    content.append({"text": json.dumps(
                        {"action": block["toolUse"]["name"], **block["toolUse"]["input"]})})
                elif "toolResult" in block:
                    inner = block["toolResult"]["content"][0].get("json", {})
                    content.append({"text": f'Customer answered: '
                                            f'"{inner.get("customer_answer", "")}"'})
                else:
                    content.append(block)
            self.messages[i] = {"role": m["role"], "content": content}
