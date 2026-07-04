"""Gemini adapter — same duck-typed interface as BedrockChat, so the concierge
engine runs on it unchanged.

Why Google AI Studio: its free tier needs NO credit card and NO billing
account — immune to the AISPL account-gating that blocks Bedrock invokes on
this AWS account. Get a key at https://aistudio.google.com/apikey and set
GEMINI_API_KEY. Free-tier limits (~10 req/min, ~250 req/day on 2.5-flash)
cover ~80 concierge sessions/day at $0.

Translation layer: Bedrock-Converse-shaped messages/tools in, Bedrock-shaped
responses out. toolChoice {"any"} -> FunctionCallingConfig(mode="ANY");
{"tool": {...}} -> mode="ANY" + allowed_function_names (exact equivalent of
Bedrock's forced tool call).

    GEMINI_MODEL_ID   default gemini-2.5-flash
"""

import os

from google import genai
from google.genai import types

from src.concierge.bedrock_client import SESSION_COST_LIMIT_USD, CostMeter

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-flash")

# Paid-tier USD/MTok for reference — AI Studio free tier actually bills $0.
PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _sanitize_schema(schema: dict) -> dict:
    """JSON Schema -> Gemini Schema: union types like ["string","null"]
    become type + nullable, everything else passes through recursively."""
    out = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, list):
            non_null = [t for t in v if t != "null"]
            out["type"] = non_null[0] if non_null else "string"
            if "null" in v:
                out["nullable"] = True
        elif k == "properties":
            out[k] = {name: _sanitize_schema(s) for name, s in v.items()}
        elif k == "items":
            out[k] = _sanitize_schema(v)
        else:
            out[k] = v
    return out


def _to_contents(messages: list[dict]) -> list[types.Content]:
    contents = []
    last_tool_name = "ask_question"  # toolResult blocks carry no name; track it
    for m in messages:
        parts = []
        for block in m.get("content", []):
            if "text" in block:
                parts.append(types.Part.from_text(text=block["text"]))
            elif "toolUse" in block:
                tu = block["toolUse"]
                last_tool_name = tu["name"]
                parts.append(types.Part.from_function_call(
                    name=tu["name"], args=_plain(tu["input"])))
            elif "toolResult" in block:
                payload = block["toolResult"]["content"][0].get("json", {})
                parts.append(types.Part.from_function_response(
                    name=last_tool_name, response=_plain(payload)))
        contents.append(types.Content(
            role="user" if m["role"] == "user" else "model", parts=parts))
    return contents


def _to_tools(tool_config: dict):
    decls = [types.FunctionDeclaration(
        name=t["toolSpec"]["name"],
        description=t["toolSpec"].get("description", ""),
        parameters=_sanitize_schema(t["toolSpec"]["inputSchema"]["json"]))
        for t in tool_config.get("tools", [])]
    choice = tool_config.get("toolChoice") or {}
    if "tool" in choice:
        fcc = types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=[choice["tool"]["name"]])
    elif "any" in choice:
        fcc = types.FunctionCallingConfig(mode="ANY")
    else:
        fcc = types.FunctionCallingConfig(mode="AUTO")
    return ([types.Tool(function_declarations=decls)],
            types.ToolConfig(function_calling_config=fcc))


class GeminiChat:
    region = "google-ai-studio"

    def __init__(self, model_id: str = DEFAULT_MODEL):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key (no card needed) at "
                "https://aistudio.google.com/apikey then: "
                'setx GEMINI_API_KEY "<key>" (new terminal after setx).')
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.meter = CostMeter(*PRICES.get(model_id, (0.30, 2.50)))

    def converse(self, system: str, messages: list[dict],
                 tool_config: dict | None = None,
                 max_tokens: int = 700, temperature: float = 0.3) -> dict:
        if self.meter.usd >= SESSION_COST_LIMIT_USD:
            raise RuntimeError(f"Session cost limit reached ({self.meter.summary()})")
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
            # Structured interviewing doesn't need extended thinking; budget 0
            # keeps latency and (paid-tier) cost down.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if tool_config:
            config.tools, config.tool_config = _to_tools(tool_config)
        resp = self.client.models.generate_content(
            model=self.model_id, contents=_to_contents(messages), config=config)

        usage = resp.usage_metadata
        self.meter.add({
            "inputTokens": getattr(usage, "prompt_token_count", 0) or 0,
            "outputTokens": ((getattr(usage, "candidates_token_count", 0) or 0)
                             + (getattr(usage, "thoughts_token_count", 0) or 0)),
        })

        content = []
        for i, part in enumerate(resp.candidates[0].content.parts or []):
            if getattr(part, "function_call", None):
                content.append({"toolUse": {
                    "toolUseId": f"gemini-{i}",
                    "name": part.function_call.name,
                    "input": _plain(dict(part.function_call.args or {}))}})
            elif getattr(part, "text", None):
                content.append({"text": part.text})
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": str(resp.candidates[0].finish_reason),
        }
