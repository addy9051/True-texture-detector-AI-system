"""Verify the ACTIVE LLM provider (per src/concierge/portkey_llm.py) with one tiny
call, plus a native tool-use probe.

    uv run python scripts/check_llm.py

Provider selection: LLM_PROVIDER env (gemini | bedrock | mock), defaulting to
gemini when GEMINI_API_KEY is set, else bedrock. For the AWS-specific checks
(STS identity, account-restriction hints) use scripts/check_bedrock.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from src.concierge.portkey_llm import make_llm, provider_name


def main():
    print(f"[..] Active provider: {provider_name()}")
    try:
        llm = make_llm()
    except RuntimeError as e:
        sys.exit(f"[fail] {e}")

    model_id = getattr(llm, "model_name", getattr(llm, "model_id", "unknown"))
    print(f"[..] Pinging {model_id} via Portkey ...")
    try:
        msg = llm.invoke([
            SystemMessage(content="Reply with exactly: OK"),
            HumanMessage(content="ping")
        ])
        print(f"[ok] Model replied: {msg.content!r}")
    except Exception as e:
        sys.exit(f"[fail] {e.__class__.__name__}: {e}")

    print("[..] Testing tool use ...")
    try:
        def ping() -> str:
            """connectivity test"""
            return "pong"

        llm_with_tools = llm.bind_tools([ping], tool_choice="required")
        msg = llm_with_tools.invoke([SystemMessage(content="Call the ping tool."), HumanMessage(content="ping")])

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("[ok] Native tool use supported — concierge will use tools seamlessly")
        else:
            print("[warn] Native tool use rejected (no tool call returned)")
    except Exception as e:
        print(f"[warn] Native tool use rejected ({e.__class__.__name__}: {e})")

    print("\nAll good — run: uv run python scripts/run_concierge.py")


if __name__ == "__main__":
    main()
