"""Offline stand-in for BedrockChat — no AWS, no network, no cost.

Lets the concierge pipeline (session engine, CLI, seller_insights.jsonl, and
the Phase-5 dashboard) be exercised end-to-end while the AWS account
restriction is being resolved. Responses are CANNED, not model output — the
insight rows it produces carry model_id "mock.offline-concierge" so they are
never mistaken for real sessions.

Emits native-tool-shaped responses, so it exercises the same code path the
real model will use.
"""

from src.concierge.bedrock_client import CostMeter


class MockBedrockChat:
    model_id = "mock.offline-concierge"
    region = "local"

    def __init__(self):
        self.meter = CostMeter(0.0, 0.0)
        self._turn = 0

    def converse(self, system, messages, tool_config=None, **kwargs):
        self._turn += 1
        if self._turn == 1:
            tool = {"toolUseId": "mock-1", "name": "ask_question", "input": {
                "question": "What made you decide to return this item?",
                "options": ["How the fabric feels", "How it fits",
                            "How it looks", "Changed my mind"]}}
        elif self._turn == 2:
            tool = {"toolUseId": "mock-2", "name": "ask_question", "input": {
                "question": "How would you describe the fabric?",
                "options": ["Shiny or plastic-like", "Scratchy or rough",
                            "Too thin or see-through", "Too heavy or hot"]}}
        else:
            answers = [b["toolResult"]["content"][0]["json"].get("customer_answer", "")
                       for m in messages for b in m.get("content", [])
                       if "toolResult" in b]
            quote = (answers[-1] if answers else "")[:140]
            tool = {"toolUseId": "mock-3", "name": "submit_diagnosis", "input": {
                "root_cause_category": "TEXTURE_MISMATCH",
                "material_issue_suspected": True,
                "reported_feel": "shiny, plastic-like",
                "weather_context": None,
                "weather_suitability_mismatch": None,
                "suspected_substitution": "polyester",
                "customer_summary": quote or "Fabric felt synthetic, unlike the listed cotton.",
                "seller_action": "SUPPLY_CHAIN_AUDIT",
                "listing_fix_recommendation": "Listing claims cotton but customers "
                                              "report a shiny synthetic feel. Verify "
                                              "the fiber with the supplier, then "
                                              "either declare the actual blend or "
                                              "source genuine cotton; if the fiber "
                                              "checks out, source premium-grade "
                                              "cotton to fix the hand-feel.",
                "customer_closing_message": "Thank you so much for the detailed "
                                            "feedback — it's genuinely helpful and "
                                            "we're forwarding it to our team to "
                                            "improve this product.",
                "confidence": "MEDIUM"}}
        return {
            "output": {"message": {"role": "assistant", "content": [{"toolUse": tool}]}},
            "usage": {"inputTokens": 0, "outputTokens": 0},
            "stopReason": "tool_use",
        }
