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
from pathlib import Path

from src.concierge.bedrock_client import BedrockChat
from src.physics.category_materials import likely_materials, normalize_to_ontology
from src.physics.fabric_ontology import FabricOntology

MAX_QUESTIONS = 3

# Error fragments that mean "this account/model won't do native tools".
# Provider-agnostic (Bedrock ValidationExceptions, Gemini function-calling
# errors) — anything else is re-raised untouched.
TOOL_REJECTION_MARKERS = ("Operation not allowed", "toolChoice", "toolConfig",
                          "tool use", "does not support tool", "function_call",
                          "FunctionDeclaration", "function calling",
                          "tool_choice", "tool_use_failed", "output_parse_failed")

# Generous budget: reasoning models (gpt-oss) think before the tool call, and
# the diagnosis payload is sizeable — 700 tokens proved too tight live.
MAX_RESPONSE_TOKENS = 1600

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
            "reported_feel": {"type": ["string", "null"],
                              "description": "The customer's tactile description in their own adjectives, 2-6 words (e.g. 'rough and stiff'); null if feel was never discussed"},
            "weather_context": {"type": ["string", "null"],
                                "description": "Weather/wear context in which the garment disappointed (e.g. 'sweltering in humid heat'); null if not mentioned"},
            "weather_suitability_mismatch": {
                "type": ["boolean", "null"],
                "description": "true if the customer wore the item in weather OUTSIDE the listed material's ideal range WITHOUT compensating precautions (e.g. layering, outerwear); false if worn in suitable conditions or precautions were taken; null if weather was never discussed"},
            "suspected_substitution": {
                "type": ["string", "null"],
                "description": "Fiber likely substituted for the claimed one, only if the customer's report matches a known substitution signature; null for quality issues within the genuine fiber (e.g. coarse low-grade cotton)"},
            "customer_summary": {"type": "string",
                                 "description": "One sentence in the customer's own words"},
            "seller_action": {"type": "string", "enum": [
                "SUPPLY_CHAIN_AUDIT", "QUALITY_IMPROVEMENT", "LISTING_FIX",
                "SIZE_CHART_FIX", "NO_ACTION"]},
            "listing_fix_recommendation": {
                "type": ["string", "null"],
                "description": "Must name the claimed material, the specific gap, and the remedy matching the hypothesis: suspected substitution -> verify/declare the actual fiber or source the genuine one; low-grade genuine fiber -> source premium-grade material (e.g. combed cotton) or adjust the listing's feel claims; give BOTH remedies when one session cannot distinguish. For a pure weather-suitability mismatch, recommend clarifying the ideal season/weather in the listing"},
            "customer_closing_message": {
                "type": "string",
                "description": "A warm, polite 2-4 sentence message shown to THE CUSTOMER. Always sincerely thank them and say their feedback is genuinely helpful and forwarded to the team to improve the product. If weather_suitability_mismatch is true, ALSO gently note the weather the product is best suited for and that wearing it outside those conditions (unless layered/paired with suitable items) can explain the discomfort — polite and non-blaming, framed as helpful guidance"},
            "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        },
        "required": ["root_cause_category", "material_issue_suspected",
                     "reported_feel", "customer_summary", "seller_action",
                     "customer_closing_message", "confidence"],
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
    "reported_feel": "<customer's tactile adjectives, or null>",
    "weather_context": "<weather/wear context, or null>",
    "weather_suitability_mismatch": "<true|false|null>",
    "suspected_substitution": "<fiber or null>",
    "customer_summary": "<one sentence>",
    "seller_action": "SUPPLY_CHAIN_AUDIT|QUALITY_IMPROVEMENT|LISTING_FIX|SIZE_CHART_FIX|NO_ACTION",
    "listing_fix_recommendation": "<must name the claimed material, the gap, and the matching remedy (declare/verify fiber for substitution; premium sourcing for quality; both if unsure), or null>",
    "customer_closing_message": "<warm 2-4 sentence message to the customer: thank them, feedback forwarded to the team; if weather mismatch, gently note the ideal weather and the layering caveat>",
    "confidence": "HIGH|MEDIUM|LOW"}}""".format(max_q=MAX_QUESTIONS)

FORCE_DIAGNOSIS = ("\nIMPORTANT: You have used all your questions. "
                   "Output the submit_diagnosis JSON object now.")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)

# Procedural memory: the interview policy / RESPONSE MATRIX lives in skill.md so
# it can be iterated without touching code. Loaded once, cached.
_SKILL_PATH = Path(__file__).resolve().parent / "skill.md"
_SKILL_BODY = None


def _skill_policy() -> str:
    """Return the procedural policy from skill.md (the section after the '---'
    separator), with {max_questions} filled in."""
    global _SKILL_BODY
    if _SKILL_BODY is None:
        raw = _SKILL_PATH.read_text(encoding="utf-8")
        _SKILL_BODY = raw.split("---", 1)[-1].strip()
    return _SKILL_BODY.replace("{max_questions}", str(MAX_QUESTIONS))

# The four response quadrants (feel = material_issue_suspected,
# weather = weather_suitability_mismatch). Computed deterministically by the
# engine so the dashboard and seller logic never depend on the LLM's wording.
CASE_FEEL_ONLY = "FEEL_ONLY"            # A: defect, weather fine
CASE_WEATHER_ONLY = "WEATHER_ONLY"      # B: no defect, wrong weather
CASE_FEEL_AND_WEATHER = "FEEL_AND_WEATHER"  # C: both
CASE_NO_ISSUE = "NO_ISSUE"              # D: product fine, returned anyway


def classify_case(material_issue, weather_mismatch) -> str:
    feel = bool(material_issue)
    weather = bool(weather_mismatch)  # None (weather not discussed) -> not a mismatch
    if feel and not weather:
        return CASE_FEEL_ONLY
    if not feel and weather:
        return CASE_WEATHER_ONLY
    if feel and weather:
        return CASE_FEEL_AND_WEATHER
    return CASE_NO_ISSUE


def _ontology_line(m: str, ontology: FabricOntology) -> str:
    spec = ontology.expectations(m) or {}
    return (f"- {m}: genuine feel = {', '.join(spec.get('expected_texture', []))}. "
            f"Thermal = {spec.get('thermal', 'unknown')}; ideal weather = "
            f"{', '.join(spec.get('weather_suitability', [])) or 'unknown'}. "
            f"Red flags = {', '.join(spec.get('failing_adjectives', []))}. "
            f"Common substitution = {', '.join(spec.get('substitution_suspects', [])) or 'none'}.")


def _weave_line(w: str, ontology: FabricOntology) -> str:
    spec = ontology.weave_expectations(w) or {}
    return (f"- {w} (weave): should feel = {', '.join(spec.get('expected_texture', []))}. "
            f"Feels wrong if = {', '.join(spec.get('failing_adjectives', []))}. "
            f"Suited to weather = {', '.join(spec.get('weather_suitability', [])) or 'any'} "
            f"({spec.get('warmth', 'neutral')}).")


def resolve_materials(product: dict, ontology: FabricOntology,
                      category: str | None = None) -> tuple[list[str], list[str], str]:
    """Return (fibers, weaves, prior_note). When the listing states no fiber but
    the category is known, fall back to the category's common-materials prior."""
    fibers = ontology.materials_from_listing(product)
    weaves = ontology.weaves_from_listing(product)
    prior_note = ""
    if not fibers and category:
        prior = likely_materials(category)
        seen = set()
        for mat in prior:
            for f in normalize_to_ontology(mat):
                if f not in seen:
                    seen.add(f)
                    fibers.append(f)
            for w in ontology.weaves_from_listing({"title": mat}):
                if w not in weaves:
                    weaves.append(w)
        if prior:
            prior_note = (
                f"\n\nNO MATERIAL IS LISTED. This is a '{category}', which commonly "
                f"uses: {', '.join(prior[:8])}. Ask the customer which fabric it is "
                f"(or its closest feel) before diagnosing. Likely fibers/weaves below.")
    return fibers, weaves, prior_note


def build_system_prompt(product: dict, ontology: FabricOntology,
                        diagnosis_row: dict | None, category: str | None = None) -> str:
    claimed, weaves, prior_block = resolve_materials(product, ontology, category)
    ontology_lines = [_ontology_line(m, ontology) for m in claimed]
    weave_lines = [_weave_line(w, ontology) for w in weaves]

    evidence_block = "None on file."
    if diagnosis_row:
        complaints = diagnosis_row.get("complaint_adjectives") or []
        sentences = [h["sentence"] for h in diagnosis_row.get("hits", [])
                     if h.get("complaint")][:5]
        if complaints:
            evidence_block = (
                f"Prior customers reported: {', '.join(complaints)}. "
                f"Example quotes: " + " | ".join(f'"{s}"' for s in sentences))

    context = f"""You are a returns assistant for a fashion marketplace. A customer is returning:
  "{(product.get('title') or '')[:140]}"
  Listed materials: {', '.join(claimed) or 'not stated'}.{prior_block}

FABRIC ONTOLOGY (ground truth for the listed / likely materials):
{chr(10).join(ontology_lines) or '- (no ontology entry for the listed materials)'}

WEAVE / CONSTRUCTION (surface feel independent of fiber — a satin should be smooth+glossy, a velvet plush, regardless of fiber):
{chr(10).join(weave_lines) or '- (no specific weave detected)'}

PRIOR EVIDENCE from other customers (INTERNAL — never reveal or quote this to the customer, never put it in question options; use it only to decide which dimension to probe first):
{evidence_block}"""
    return context + "\n\n" + _skill_policy()


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
                 diagnosis_row: dict | None = None, category: str | None = None):
        self.chat = chat
        self.ontology = ontology
        self.claimed_materials, self.weaves, _ = resolve_materials(
            product, ontology, category)
        self.system = build_system_prompt(product, ontology, diagnosis_row, category)
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
                resp = self.chat.converse(self.system, self.messages, self._tool_config(),
                                          max_tokens=MAX_RESPONSE_TOKENS)
                return self._from_tool_response(resp)
            except Exception as e:
                if any(marker in str(e) for marker in TOOL_REJECTION_MARKERS):
                    self.mode = "json"
                    self._detool_history()
                else:
                    raise
        resp = self.chat.converse(self._json_system(), self.messages,
                                  max_tokens=MAX_RESPONSE_TOKENS)
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
        # Ground the LLM's diagnosis with ontology facts the engine holds
        # deterministically — the seller always sees claim vs physical truth,
        # regardless of what the model chose to mention.
        payload = {
            **payload,
            "case_class": classify_case(payload.get("material_issue_suspected"),
                                        payload.get("weather_suitability_mismatch")),
            "claimed_materials": self.claimed_materials,
            "material_ground_truth": [
                {"material": m,
                 "genuine_feel": spec.get("expected_texture", []),
                 "thermal": spec.get("thermal"),
                 "ideal_weather": spec.get("weather_suitability", []),
                 "common_substitutes": spec.get("substitution_suspects", [])}
                for m in self.claimed_materials
                if (spec := self.ontology.expectations(m))],
            "weaves": self.weaves,
            "weave_ground_truth": [
                {"weave": w,
                 "should_feel": spec.get("expected_texture", []),
                 "feels_wrong_if": spec.get("failing_adjectives", []),
                 "warmth": spec.get("warmth")}
                for w in self.weaves
                if (spec := self.ontology.weave_expectations(w))],
        }
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
