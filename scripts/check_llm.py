"""Verify the ACTIVE LLM provider (per src/concierge/provider.py) with one tiny
call, plus a native tool-use probe.

    uv run python scripts/check_llm.py

Provider selection: LLM_PROVIDER env (gemini | bedrock | mock), defaulting to
gemini when GEMINI_API_KEY is set, else bedrock. For the AWS-specific checks
(STS identity, account-restriction hints) use scripts/check_bedrock.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.concierge.provider import make_chat, provider_name


def main():
    print(f"[..] Active provider: {provider_name()}")
    try:
        chat = make_chat()
    except RuntimeError as e:
        sys.exit(f"[fail] {e}")

    print(f"[..] Pinging {chat.model_id} via {chat.region} ...")
    try:
        resp = chat.converse(
            system="Reply with exactly: OK",
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            max_tokens=128, temperature=0.0)
        text = " ".join(b.get("text", "") for b in
                        resp["output"]["message"]["content"]).strip()
        print(f"[ok] Model replied: {text!r}")
    except Exception as e:
        sys.exit(f"[fail] {e.__class__.__name__}: {e}")

    try:
        chat.converse(
            system="Call the ping tool.",
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            tool_config={"tools": [{"toolSpec": {
                "name": "ping", "description": "connectivity test",
                "inputSchema": {"json": {"type": "object", "properties": {}}}}}],
                "toolChoice": {"any": {}}},
            max_tokens=300)
        print("[ok] Native tool use supported — concierge will force tool calls")
    except Exception as e:
        print(f"[warn] Native tool use rejected ({e.__class__.__name__}: {e})")
        print("       Concierge will automatically use its JSON-protocol fallback.")

    free_tier = {"google-ai-studio": " (AI Studio free tier bills $0)",
                 "groq-cloud": " (Groq free tier bills $0)"}
    print(f"\nTotal cost of this check: {chat.meter.summary()}"
          + free_tier.get(chat.region, ""))
    print("All good — run: uv run python scripts/run_concierge.py")


if __name__ == "__main__":
    main()
