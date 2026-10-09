"""LLM provider selection via Portkey AI gateway.

Replaces the old per-provider adapters (bedrock_client.py, groq_client.py,
gemini_client.py, provider.py) with a single unified gateway.

Provider selection keeps the same env-var-driven priority as before::

    LLM_PROVIDER = modal | groq | gemini | bedrock | mock    (explicit override)

Default when LLM_PROVIDER is unset, in order:
    modal    if MODAL_LLM_URL or VLLM_BASE_URL is set
    groq     if GROQ_API_KEY or GROQ_MODEL_ID is set
    gemini   if GEMINI_API_KEY or GEMINI_MODEL_ID is set
    bedrock  otherwise

All live providers are routed through Portkey (PORTKEY_API_KEY required).
The model field uses the ``@slug/model-name`` format — Portkey resolves the
provider from the slug configured in its dashboard.  Mock mode bypasses
Portkey entirely.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

MAX_RESPONSE_TOKENS = 1600

# Provider slug → (env_key_for_model, default_model_with_slug)
# The @slug prefix must match the integration name set in the Portkey dashboard.
_PROVIDERS = {
    "groq":    ("GROQ_MODEL_ID",    "@texture/openai/gpt-oss-120b"),
    "gemini":  ("GEMINI_MODEL_ID",  "@texture/gemini-2.5-flash"),
    "bedrock": ("BEDROCK_MODEL_ID", "@texture/mistral.mistral-large-2407-v1:0"),
}

# Keys that help detect which provider to use (for auto-detection only)
_DETECTION_KEYS = {
    "groq":   ("GROQ_API_KEY", "GROQ_MODEL_ID"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL_ID"),
}


# Alias for backward compatibility if tracer.py imports it
_env_or_registry = os.environ.get


def provider_name() -> str:
    """Return the active provider name using the priority:
    explicit override → Modal → Groq → Gemini → Bedrock."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        return explicit
    if os.environ.get("MODAL_LLM_URL") or os.environ.get("VLLM_BASE_URL"):
        return "modal"
    for name, keys in _DETECTION_KEYS.items():
        if any(os.environ.get(k) for k in keys):
            return name
    return "bedrock"


def make_llm(provider: str | None = None, *, callbacks: list | None = None) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI routed to the active provider.

    Supports:
    - mock: offline deterministic mock
    - modal / vllm: self-hosted serverless vLLM on Modal (direct OpenAI-compatible endpoint)
    - groq / gemini / bedrock: routed through Portkey AI Gateway
    """
    name = provider or provider_name()

    if name == "mock":
        from src.concierge.mock_chat import MockChatModel
        return MockChatModel()

    # Self-hosted Modal vLLM serverless endpoint (OpenAI compatible)
    if name in ("modal", "vllm"):
        modal_url = os.environ.get("MODAL_LLM_URL") or os.environ.get("VLLM_BASE_URL")
        if not modal_url:
            raise RuntimeError(
                "MODAL_LLM_URL not set. Deploy scripts/serve_modal_vllm.py with `modal deploy` "
                "and set MODAL_LLM_URL=https://<endpoint>/v1 in your .env")
        base_url = modal_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        model_id = os.environ.get("MODAL_MODEL_ID") or "Qwen/Qwen2.5-7B-Instruct"
        api_key = os.environ.get("MODAL_API_KEY") or "modal"
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_id,
            temperature=0.3,
            max_tokens=MAX_RESPONSE_TOKENS,
            callbacks=callbacks,
        )

    portkey_key = os.environ.get("PORTKEY_API_KEY")
    if not portkey_key:
        raise RuntimeError(
            "PORTKEY_API_KEY not set. Get a free key at https://portkey.ai "
            "and add it to .env")

    if name not in _PROVIDERS:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER {name!r} — use modal | groq | gemini | bedrock | mock")

    model_env, default_model = _PROVIDERS[name]

    # Model ID: use the env-var override, or the default @slug/model.
    # If the user sets a raw model name (no @), prepend the workspace slug.
    model_id = os.environ.get(model_env) or default_model

    # Build Portkey headers.  api_key in the header is the Portkey key.
    # The @slug in the model field handles provider routing — no virtual key
    # or inline provider header needed.
    headers = createHeaders(
        api_key=portkey_key,
        metadata={"feature": "concierge", "provider": name},
    )

    return ChatOpenAI(
        api_key=portkey_key,          # Portkey key as Bearer token
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        model=model_id,
        temperature=0.3,
        max_tokens=MAX_RESPONSE_TOKENS,
        callbacks=callbacks,
    )
