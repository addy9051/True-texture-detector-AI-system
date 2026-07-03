"""Thin wrapper around the Bedrock Converse API with a running cost meter.

Deliberately uses the plain Converse API — NOT Bedrock Agents or Knowledge
Bases. The KB default vector store (OpenSearch Serverless) alone would burn
the project's $100 budget (PROJECT_PLAN.md §5); the fabric ontology is a few
KB and lives in the system prompt instead.

Model choice: Amazon Nova Pro. Anthropic Claude models on Bedrock are billed
through AWS Marketplace and are NOT covered by promotional credits; Nova is
first-party (credit-eligible), supports tool use on Converse, and is cheap.

Model and region come from env vars so switching is zero-code:

    BEDROCK_REGION    default us-west-2 (where Mistral Large 24.07 is verified working
                      on this account, 2026-07-03)
    BEDROCK_MODEL_ID  default mistral.mistral-large-2407-v1:0
                      (TEMPORARY stand-in while the account's Nova Pro billing/plan
                      issue is resolved — switch back to us.amazon.nova-pro-v1:0 then)

For the India-demo story with Nova, switch in-region to Mumbai:
    $env:BEDROCK_REGION   = "ap-south-1"
    $env:BEDROCK_MODEL_ID = "apac.amazon.nova-pro-v1:0"
"""

import os
from dataclasses import dataclass

import boto3

DEFAULT_REGION = os.environ.get("BEDROCK_REGION", "us-west-2")
DEFAULT_MODEL = os.environ.get("BEDROCK_MODEL_ID", "mistral.mistral-large-2407-v1:0")

# USD per million tokens, keyed by model-id substring (first match wins).
PRICES = {
    "nova-pro": (0.80, 3.20),
    "nova-lite": (0.06, 0.24),
    "nova-micro": (0.035, 0.14),
    "claude-haiku-4-5": (1.00, 5.00),
    "mistral-large-2407": (3.00, 9.00),
    "mistral-large": (4.00, 12.00),
}
FALLBACK_PRICE = (1.00, 5.00)  # conservative estimate for unknown models

# Belt-and-braces budget guard: abort any single session that exceeds this.
SESSION_COST_LIMIT_USD = 0.25


def price_for(model_id: str) -> tuple[float, float]:
    for key, price in PRICES.items():
        if key in model_id:
            return price
    return FALLBACK_PRICE


@dataclass
class CostMeter:
    price_in: float
    price_out: float
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: dict):
        self.input_tokens += usage.get("inputTokens", 0)
        self.output_tokens += usage.get("outputTokens", 0)

    @property
    def usd(self) -> float:
        return (self.input_tokens * self.price_in
                + self.output_tokens * self.price_out) / 1_000_000

    def summary(self) -> str:
        return (f"{self.input_tokens} in / {self.output_tokens} out tokens"
                f" = ${self.usd:.4f}")


class BedrockChat:
    def __init__(self, region: str = DEFAULT_REGION, model_id: str = DEFAULT_MODEL):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.region = region
        self.model_id = model_id
        self.meter = CostMeter(*price_for(model_id))

    def converse(self, system: str, messages: list[dict], tool_config: dict | None = None,
                 max_tokens: int = 700, temperature: float = 0.3) -> dict:
        if self.meter.usd >= SESSION_COST_LIMIT_USD:
            raise RuntimeError(f"Session cost limit reached ({self.meter.summary()})")
        kwargs = dict(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=messages,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        if tool_config:
            kwargs["toolConfig"] = tool_config
        resp = self.client.converse(**kwargs)
        self.meter.add(resp.get("usage", {}))
        return resp
