"""Offline stand-in LLM — no cloud, no network, no cost.

Extends LangChain's BaseChatModel so it slots into the LangGraph agent
seamlessly.  Returns canned AIMessages with tool calls, exercising the same
code paths the real model uses.  Sessions carry model_id
``mock.offline-concierge`` so they are never mistaken for real ones.

Emits native tool calls (AIMessage.tool_calls) — same path as real providers.
"""

import uuid
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MockChatModel(BaseChatModel):
    """Deterministic mock that always asks two questions then submits a
    diagnosis.  Stateful: ``_turn`` increments per call."""

    model_name: str = "mock.offline-concierge"
    _turn: int = 0

    # ------------------------------------------------------ required overrides

    @property
    def _llm_type(self) -> str:
        return "mock-concierge"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._turn += 1
        tool_calls = self._mock_tool_calls(messages)
        msg = AIMessage(content="", tool_calls=tool_calls)
        # Attach synthetic usage metadata so cost-tracking code doesn't break.
        msg.usage_metadata = {"input_tokens": 0, "output_tokens": 0,
                              "total_tokens": 0}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "MockChatModel":
        return self

    # ----------------------------------------------------------- mock helpers

    @property
    def model_id(self) -> str:
        return self.model_name

    def _mock_tool_calls(self, messages: list[BaseMessage]) -> list[dict]:
        if self._turn == 1:
            return [{"name": "ask_question", "args": {
                "question": "What made you decide to return this item?",
                "options": ["How the fabric feels", "How it fits",
                            "How it looks", "Changed my mind"],
            }, "id": f"mock-{uuid.uuid4().hex[:8]}", "type": "tool_call"}]

        if self._turn == 2:
            return [{"name": "ask_question", "args": {
                "question": "How would you describe the fabric?",
                "options": ["Shiny or plastic-like", "Scratchy or rough",
                            "Too thin or see-through", "Too heavy or hot"],
            }, "id": f"mock-{uuid.uuid4().hex[:8]}", "type": "tool_call"}]

        # Turn 3+: submit diagnosis
        # Try to echo the last customer answer if available.
        last_answer = ""
        for m in reversed(messages):
            text = getattr(m, "content", "")
            if isinstance(text, str) and "customer_answer" in text:
                import json
                try:
                    last_answer = json.loads(text).get("customer_answer", "")
                except (json.JSONDecodeError, AttributeError):
                    pass
                break

        return [{"name": "submit_diagnosis", "args": {
            "root_cause_category": "TEXTURE_MISMATCH",
            "material_issue_suspected": True,
            "reported_feel": "shiny, plastic-like",
            "weather_context": None,
            "weather_suitability_mismatch": None,
            "suspected_substitution": "polyester",
            "customer_summary": last_answer or "Fabric felt synthetic, unlike the listed cotton.",
            "seller_action": "SUPPLY_CHAIN_AUDIT",
            "listing_fix_recommendation": (
                "Listing claims cotton but customers report a shiny synthetic feel. "
                "Verify the fiber with the supplier, then either declare the actual "
                "blend or source genuine cotton; if the fiber checks out, source "
                "premium-grade cotton to fix the hand-feel."),
            "customer_closing_message": (
                "Thank you so much for the detailed feedback — it's genuinely "
                "helpful and we're forwarding it to our team to improve this product."),
            "confidence": "MEDIUM",
        }, "id": f"mock-{uuid.uuid4().hex[:8]}", "type": "tool_call"}]
