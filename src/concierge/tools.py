"""Pydantic tool schemas for the concierge agent's two interview actions.

These define the structured inputs the LLM can call via LangGraph's tool-calling
mechanism.  The schemas match the original Bedrock Converse toolSpecs exactly so
the diagnosis output shape is unchanged — downstream consumers (dashboard,
insights_store, seller_escalation) work without modification.
"""

from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ------------------------------------------------------------ ask_question

class AskQuestionInput(BaseModel):
    question: str = Field(
        description="One short question (under 25 words) about the return")
    options: list[str] = Field(
        description="2-4 tappable options. The customer may also type free text.",
        min_length=2, max_length=4)


@tool(args_schema=AskQuestionInput)
def ask_question(question: str, options: list[str]) -> str:
    """Ask the customer ONE short question (under 25 words) about their return,
    with 2-4 tappable options. The customer may also type a free-text answer
    instead of picking an option."""
    # Body is never executed — the graph intercepts tool calls via routing.
    return "question_asked"


# ---------------------------------------------------------- submit_diagnosis

class SubmitDiagnosisInput(BaseModel):
    root_cause_category: Literal[
        "TEXTURE_MISMATCH", "THERMAL_DISCOMFORT", "SIZE_FIT",
        "COLOR_APPEARANCE", "QUALITY_DEFECT", "CHANGED_MIND", "OTHER"
    ]
    material_issue_suspected: bool
    reported_feel: Optional[str] = Field(
        default=None,
        description=("The customer's tactile description in their own adjectives, "
                     "2-6 words (e.g. 'rough and stiff'); null if feel was never discussed"))
    weather_context: Optional[str] = Field(
        default=None,
        description=("Weather/wear context in which the garment disappointed "
                     "(e.g. 'sweltering in humid heat'); null if not mentioned"))
    weather_suitability_mismatch: Optional[bool] = Field(
        default=None,
        description=("true if the customer wore the item in weather OUTSIDE the "
                     "listed material's ideal range WITHOUT compensating precautions; "
                     "false if worn in suitable conditions or precautions were taken; "
                     "null if weather was never discussed"))
    suspected_substitution: Optional[str] = Field(
        default=None,
        description=("Fiber likely substituted for the claimed one, only if the "
                     "customer's report matches a known substitution signature; "
                     "null for quality issues within the genuine fiber"))
    customer_summary: str = Field(
        description="One sentence in the customer's own words")
    seller_action: Literal[
        "SUPPLY_CHAIN_AUDIT", "QUALITY_IMPROVEMENT", "LISTING_FIX",
        "SIZE_CHART_FIX", "NO_ACTION"
    ]
    listing_fix_recommendation: Optional[str] = Field(
        default=None,
        description=("Must name the claimed material, the specific gap, and the "
                     "matching remedy; null only when seller_action is NO_ACTION"))
    customer_closing_message: str = Field(
        description=("A warm, polite 2-4 sentence message shown to THE CUSTOMER. "
                     "Always sincerely thank them and say their feedback is genuinely "
                     "helpful and forwarded to the team to improve the product."))
    confidence: Literal["HIGH", "MEDIUM", "LOW"]


@tool(args_schema=SubmitDiagnosisInput)
def submit_diagnosis(
    root_cause_category: str,
    material_issue_suspected: bool,
    customer_summary: str,
    seller_action: str,
    customer_closing_message: str,
    confidence: str,
    reported_feel: Optional[str] = None,
    weather_context: Optional[str] = None,
    weather_suitability_mismatch: Optional[bool] = None,
    suspected_substitution: Optional[str] = None,
    listing_fix_recommendation: Optional[str] = None,
) -> str:
    """Submit the final structured diagnosis of why the item is being returned."""
    # Body is never executed — the graph intercepts tool calls via routing.
    return "diagnosis_submitted"
