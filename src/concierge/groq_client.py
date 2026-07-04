"""Groq adapter — same duck-typed interface as BedrockChat, so the concierge
engine runs on it unchanged.

Why Groq: free tier with NO credit card and no billing entity (sign in with
Google/GitHub at https://console.groq.com), OpenAI-compatible API with real
tool calling incl. forced function choice, very fast inference. Free limits on
llama-3.3-70b-versatile (~30 req/min, ~1,000 req/day) cover hundreds of
concierge sessions daily at $0.

Implemented as plain REST against the OpenAI-compatible endpoint via
`requests` — no extra SDK. Translation layer: Bedrock-Converse-shaped
messages/tools in, Bedrock-shaped responses out.
    toolChoice {"any"}          -> tool_choice "required"
    toolChoice {"tool": {...}}  -> tool_choice {"type":"function","function":{...}}

    GROQ_MODEL_ID   default openai/gpt-oss-120b (best tool-calling reasoner on
                    this account's model list; llama-3.3-70b-versatile is the
                    battle-tested fallback)
"""

import json
import os
import re
import time

import requests

from src.concierge.bedrock_client import SESSION_COST_LIMIT_USD, CostMeter

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL_ID", "openai/gpt-oss-120b")

# Paid-tier USD/MTok for reference — the free tier bills $0.
PRICES = {
    "llama-3.3-70b": (0.59, 0.79),
    "llama-3.1-8b": (0.05, 0.08),
    "gpt-oss-120b": (0.15, 0.75),
    "kimi-k2": (1.00, 3.00),
}


def price_for(model_id: str) -> tuple[float, float]:
    for key, price in PRICES.items():
        if key in model_id:
            return price
    return (0.59, 0.79)


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        for block in m.get("content", []):
            if "text" in block:
                out.append({"role": m["role"], "content": block["text"]})
            elif "toolUse" in block:
                tu = block["toolUse"]
                out.append({"role": "assistant", "content": None, "tool_calls": [{
                    "id": tu["toolUseId"], "type": "function",
                    "function": {"name": tu["name"],
                                 "arguments": json.dumps(tu["input"])}}]})
            elif "toolResult" in block:
                tr = block["toolResult"]
                out.append({"role": "tool", "tool_call_id": tr["toolUseId"],
                            "content": json.dumps(tr["content"][0].get("json", {}))})
    return out


def _to_openai_tools(tool_config: dict) -> tuple[list, object]:
    tools = [{"type": "function", "function": {
        "name": t["toolSpec"]["name"],
        "description": t["toolSpec"].get("description", ""),
        "parameters": t["toolSpec"]["inputSchema"]["json"],
    }} for t in tool_config.get("tools", [])]
    choice = tool_config.get("toolChoice") or {}
    if "tool" in choice:
        tool_choice = {"type": "function",
                       "function": {"name": choice["tool"]["name"]}}
    elif "any" in choice:
        tool_choice = "required"
    else:
        tool_choice = "auto"
    return tools, tool_choice


class GroqChat:
    region = "groq-cloud"

    def __init__(self, model_id: str = DEFAULT_MODEL):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key (no card needed) at "
                "https://console.groq.com -> API Keys, then: "
                'setx GROQ_API_KEY "<key>" (new terminal after setx).')
        self.api_key = api_key
        self.model_id = model_id
        self.meter = CostMeter(*price_for(model_id))

    def converse(self, system: str, messages: list[dict],
                 tool_config: dict | None = None,
                 max_tokens: int = 700, temperature: float = 0.3) -> dict:
        if self.meter.usd >= SESSION_COST_LIMIT_USD:
            raise RuntimeError(f"Session cost limit reached ({self.meter.summary()})")
        body = {
            "model": self.model_id,
            "messages": _to_openai_messages(system, messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "gpt-oss" in self.model_id:
            # Reasoning model: keep the hidden reasoning short so small token
            # budgets still leave room for the actual tool call / answer.
            body["reasoning_effort"] = "low"
        if tool_config:
            body["tools"], body["tool_choice"] = _to_openai_tools(tool_config)
        for attempt in range(3):
            resp = requests.post(
                API_URL, json=body, timeout=60,
                headers={"Authorization": f"Bearer {self.api_key}"})
            if resp.status_code == 429 and attempt < 2:
                # Free-tier TPM limit — the error text says how long to wait.
                m = re.search(r"try again in ([\d.]+)s", resp.text)
                wait = float(m.group(1)) + 1.0 if m else 15.0
                print(f"    (Groq rate limit, retrying in {wait:.0f}s ...)", flush=True)
                time.sleep(wait)
                continue
            break
        if not resp.ok:
            raise RuntimeError(f"Groq API {resp.status_code}: {resp.text[:400]}")
        data = resp.json()

        usage = data.get("usage", {})
        self.meter.add({"inputTokens": usage.get("prompt_tokens", 0),
                        "outputTokens": usage.get("completion_tokens", 0)})

        msg = data["choices"][0]["message"]
        content = []
        for tc in msg.get("tool_calls") or []:
            content.append({"toolUse": {
                "toolUseId": tc.get("id", "groq-0"),
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"] or "{}")}})
        if msg.get("content"):
            content.append({"text": msg["content"]})
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": data["choices"][0].get("finish_reason", ""),
        }
