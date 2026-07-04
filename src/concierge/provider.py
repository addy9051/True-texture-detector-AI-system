"""LLM provider selection — free-tier providers while the AWS account is
gated, Bedrock kept as fallback infrastructure, mock for offline demos.

    LLM_PROVIDER = groq | gemini | bedrock | mock    (explicit override)

Default when LLM_PROVIDER is unset, in order:
    groq    if GROQ_API_KEY is set      (primary — no card, no billing entity)
    gemini  if GEMINI_API_KEY is set    (alternative; user hit Google billing issues)
    bedrock otherwise                   (AWS fallback)

Switch-back plan: once `uv run python scripts/check_bedrock.py` passes without
ValidationException, run `setx LLM_PROVIDER "bedrock"` — no code changes.
"""

import os
import sys


def _env_or_registry(name: str) -> str | None:
    """Read an env var, falling back to the Windows user registry.

    `setx` writes to the registry but does NOT update already-open terminals,
    so a shell opened before `setx GROQ_API_KEY ...` silently lacks the key
    and the factory would wrongly fall back to Bedrock. Reading the registry
    directly makes every script work regardless of terminal age. The value is
    promoted into os.environ so the provider adapters see it too.
    """
    val = os.environ.get(name)
    if val:
        return val
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                val, _ = winreg.QueryValueEx(key, name)
        except OSError:
            return None
        if val:
            os.environ[name] = val
            return val
    return None


def provider_name() -> str:
    explicit = (_env_or_registry("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        return explicit
    if _env_or_registry("GROQ_API_KEY"):
        return "groq"
    if _env_or_registry("GEMINI_API_KEY"):
        return "gemini"
    return "bedrock"


def make_chat():
    name = provider_name()
    if name == "mock":
        from src.concierge.mock_chat import MockBedrockChat
        return MockBedrockChat()
    if name == "groq":
        from src.concierge.groq_client import GroqChat
        return GroqChat()
    if name == "gemini":
        from src.concierge.gemini_client import GeminiChat
        return GeminiChat()
    if name == "bedrock":
        from src.concierge.bedrock_client import BedrockChat
        return BedrockChat()
    raise RuntimeError(
        f"Unknown LLM_PROVIDER {name!r} — use groq | gemini | bedrock | mock")
